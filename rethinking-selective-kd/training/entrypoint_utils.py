from __future__ import annotations

import gc
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _cleanup_cuda(device_idx: int) -> None:
    """Aggressively free caches on the target CUDA device before loading."""
    if not torch.cuda.is_available():
        return
    try:
        prev = torch.cuda.current_device()
    except Exception:
        prev = None
    try:
        torch.cuda.set_device(device_idx)
    except Exception:
        pass
    try:
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()
    except Exception:
        pass
    finally:
        if prev is not None:
            try:
                torch.cuda.set_device(prev)
            except Exception:
                pass
    gc.collect()


def _get_free_gb(device_idx: int) -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _total_bytes = torch.cuda.mem_get_info(device_idx)
    except RuntimeError:
        try:
            prev_device = torch.cuda.current_device()
        except RuntimeError:
            prev_device = None
        torch.cuda.set_device(device_idx)
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        if prev_device is not None:
            torch.cuda.set_device(prev_device)
    return free_bytes / (1024**3)


def load_teacher_8bit_strict(
    model_name: str,
    prefer_gpus: List[int],
    student_gpu: Optional[int],
    min_free_gb: Optional[float],
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    Load teacher on a single GPU using BF16 weights (when supported).
    """
    teacher_gpus = [g for g in prefer_gpus if g != student_gpu]
    if not teacher_gpus:
        raise RuntimeError("No GPUs available for teacher after excluding the student GPU.")

    gpu = None
    if min_free_gb is None:
        gpu = teacher_gpus[0]
    else:
        insufficient: List[Tuple[int, float]] = []
        for candidate in teacher_gpus:
            free_gb = _get_free_gb(candidate)
            if free_gb is None:
                gpu = candidate
                break
            if free_gb >= min_free_gb:
                gpu = candidate
                break
            insufficient.append((candidate, free_gb))
        if gpu is None:
            detail = ", ".join(f"{idx}:{free:.1f}GB" for idx, free in insufficient)
            raise RuntimeError(
                f"No teacher GPU meets free-memory requirement (need >= {min_free_gb:.1f}GB). "
                f"Candidates: [{detail}]"
            )

    tok = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=False,
        padding_side="left",  # Required for decoder-only models during generation
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    _cleanup_cuda(gpu)
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch_dtype = torch.bfloat16 if bf16_supported else torch.float16
    print(f"[teacher] Loading {model_name} on cuda:{gpu} with dtype={torch_dtype}.", flush=True)

    teacher = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map={"": gpu},  # pin every module to this GPU
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
    )

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    # Sanity: ensure no layer got sharded/offloaded
    devmap = getattr(teacher, "hf_device_map", None)
    if isinstance(devmap, dict):
        bad = [k for k, v in devmap.items() if isinstance(v, str) and ("cpu" in v or "disk" in v)]
        if bad:
            raise RuntimeError(f"Unexpected CPU/offloaded modules in device_map: {bad}")

    print(f"[teacher] Teacher load completed using dtype={torch_dtype}.", flush=True)
    return teacher, tok, torch.device(f"cuda:{gpu}")


def load_teacher_with_fallback(
    model_name: str,
    prefer_gpus: List[int],          # Local GPU indices to use for the teacher, in order of preference
    student_gpu: Optional[int],      # Local GPU index reserved for the student (exclude from teacher)
    min_free_gb: Optional[float] = None,
) -> Tuple[torch.nn.Module, AutoTokenizer, torch.device]:
    """
    Load teacher in BF16 on a single GPU.
    Returns: (teacher_model, tokenizer, teacher_device_for_inputs)
    """
    try:
        return load_teacher_8bit_strict(
            model_name=model_name,
            prefer_gpus=prefer_gpus,
            student_gpu=student_gpu,
            min_free_gb=min_free_gb,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Teacher load failed on GPUs {prefer_gpus} (student GPU={student_gpu})."
        ) from exc


def load_fineweb_subset(
    tokenizer,
    max_tokens: int,
    seed: int = 1337,
    max_seq_len: int = 768,
    packing_enabled: bool = True,
):
    """
    Load FineWeb-Edu subset with automatic caching.
    The first run streams data, tokenizes, and caches to disk; later runs reuse the cache.

    Returns a list of {prompt, answer} examples.
    """
    from sampledkd.data.cache import load_or_create_fineweb_cache

    return load_or_create_fineweb_cache(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        max_seq_len=max_seq_len,
        seed=seed,
        batch_size=512,
        packing_enabled=packing_enabled,
    )
