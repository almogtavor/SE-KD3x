"""Runtime helpers to wrap lm-eval HuggingFace models with PyTorch FSDP."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from importlib import import_module
from typing import Optional

import torch

try:
    import torch.distributed as dist
    from torch.distributed.fsdp import (  # type: ignore[attr-defined]
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import (  # type: ignore[attr-defined]
        transformer_auto_wrap_policy,
    )
except Exception as exc:  # pragma: no cover - handled at runtime
    raise RuntimeError("FSDP requires torch.distributed with FSDP support") from exc

_APPLIED = False


@dataclass
class _FsdpConfig:
    layer_cls_name: Optional[str]
    dtype: torch.dtype
    policy: str


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    key = name.strip().lower()
    if key not in mapping:
        warnings.warn(
            f"[fsdp] Unknown dtype '{name}', defaulting to float16.",
            RuntimeWarning,
        )
        return torch.float16
    return mapping[key]


def _parse_sharding_strategy(policy: str) -> ShardingStrategy:
    tokens = {tok.strip().lower() for tok in policy.split() if tok.strip()}
    if "hybrid_shard" in tokens:
        return getattr(ShardingStrategy, "HYBRID_SHARD", ShardingStrategy.FULL_SHARD)
    if "shard_grad_op" in tokens:
        return ShardingStrategy.SHARD_GRAD_OP
    if "no_shard" in tokens:
        return ShardingStrategy.NO_SHARD
    if "state_shard" in tokens:
        return ShardingStrategy.STATE_SHARD
    return ShardingStrategy.FULL_SHARD


def _ensure_dist_initialized() -> None:
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def _infer_layer_class(module: torch.nn.Module, layer_cls_name: Optional[str]) -> Optional[type]:
    if layer_cls_name is None:
        return None
    candidate_modules = []
    mod_name = module.__class__.__module__
    parts = mod_name.split(".")
    for i in range(len(parts), 0, -1):
        candidate_modules.append(".".join(parts[:i]))
    seen = set()
    for mod_path in candidate_modules:
        if mod_path in seen:
            continue
        seen.add(mod_path)
        try:
            mod = import_module(mod_path)
        except Exception:
            continue
        if hasattr(mod, layer_cls_name):
            obj = getattr(mod, layer_cls_name)
            if isinstance(obj, type) and issubclass(obj, torch.nn.Module):
                return obj
    warnings.warn(
        f"[fsdp] Unable to locate layer class '{layer_cls_name}'. Falling back to whole-model wrap.",
        RuntimeWarning,
    )
    return None


def _resolve_auto_wrap_policy(layer_cls: Optional[type]):
    if layer_cls is None:
        return None
    return transformer_auto_wrap_policy({layer_cls})


def _wrap_with_fsdp(model: torch.nn.Module, cfg: _FsdpConfig) -> torch.nn.Module:
    _ensure_dist_initialized()
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        target_device = torch.device("cuda", local_rank)
    else:
        target_device = torch.device("cpu")

    # Keep the model on its current device (typically CPU) to avoid a full copy
    # spilling onto a single GPU before FSDP shards it. Just ensure dtype matches.
    if cfg.dtype is not None:
        model.to(dtype=cfg.dtype)
    layer_cls = _infer_layer_class(model, cfg.layer_cls_name)
    auto_wrap = _resolve_auto_wrap_policy(layer_cls)

    strategy = _parse_sharding_strategy(cfg.policy)

    mixed_precision = MixedPrecision(  # type: ignore[call-arg]
        param_dtype=cfg.dtype,
        reduce_dtype=cfg.dtype,
        buffer_dtype=cfg.dtype,
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=strategy,
    device_id=target_device if target_device.type == "cuda" else None,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        use_orig_params=True,
    )
    fsdp_model.eval()
    dist.barrier()
    return fsdp_model


def _patch_hf_model(cfg: _FsdpConfig) -> None:
    import lm_eval.models.huggingface as hf_mod

    original_create_model = hf_mod.HFLM._create_model

    def patched_create_model(self, *args, **kwargs):  # type: ignore[override]
        original_create_model(self, *args, **kwargs)
        if getattr(self, "_fsdp_wrapped", False):
            return
        self._model = _wrap_with_fsdp(self._model, cfg)
        self._fsdp_wrapped = True
        print(
            f"[fsdp] Wrapped model with FSDP (layer='{cfg.layer_cls_name}', dtype={cfg.dtype}, policy='{cfg.policy}').",
            flush=True,
        )

    hf_mod.HFLM._create_model = patched_create_model  # type: ignore[assignment]


def maybe_enable_fsdp() -> None:
    """Enable FSDP wrapping if the relevant environment variables are set."""
    global _APPLIED
    if _APPLIED:
        return
    if os.environ.get("SAMPLEDKD_USE_FSDP") != "1":
        return

    layer_cls = os.environ.get("SAMPLEDKD_FSDP_LAYER_CLS") or None
    dtype_name = os.environ.get("SAMPLEDKD_FSDP_DTYPE", "float16")
    policy = os.environ.get("SAMPLEDKD_FSDP_POLICY", "full_shard auto_wrap")
    dtype = _parse_dtype(dtype_name)

    cfg = _FsdpConfig(layer_cls_name=layer_cls, dtype=dtype, policy=policy)
    _patch_hf_model(cfg)
    _APPLIED = True
    print(
        f"[fsdp] FSDP instrumentation enabled (layer_cls={layer_cls}, dtype={dtype}).",
        flush=True,
    )
