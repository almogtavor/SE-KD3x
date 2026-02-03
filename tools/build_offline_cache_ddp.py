#!/usr/bin/env python3
"""DDP-only offline cache builder.

Run with torchrun to build logits caches in parallel and exit.
"""
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Ensure Hugging Face caches stay within the repo tmp to avoid home quota limits.
_repo_root = Path(__file__).resolve().parents[1]
_tmp_root = _repo_root / "tmp"
_tmp_root.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_tmp_root / "xdg_cache"))
os.environ.setdefault("HF_HOME", str(_tmp_root / "hf"))
os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub"))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import torch
import shutil
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from sekd.config import TrainingConfig
from sekd.data.dataset import AIMEJsonl, DistillCollator, PackedTokenDataset, PromptDataset
from sekd.models.loader import load_model
from sekd.distill._mixins.amp_oom import AmpOomMixin
from sekd.training.entrypoint_utils import load_fineweb_subset
from sekd.training.offline_cache import execute_cache_plan, plan_offline_cache
from sekd.training.distributed import setup_distributed_context, destroy_distributed, distributed_barrier


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build offline logits cache with DDP only.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-8B")
    p.add_argument("--student_model", type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--datasets", type=str, nargs="+", default=["fineweb"])
    p.add_argument("--dataset_config", type=str, default=None)
    p.add_argument("--prompt_col", type=str, default=None)
    p.add_argument("--answer_col", type=str, default=None)
    p.add_argument("--fineweb_tokens", type=int, default=80_000_000)
    p.add_argument("--max_seq_len", type=int, default=512)
    p.add_argument("--enable_packing", action="store_true", default=True)
    p.add_argument("--disable_packing", dest="enable_packing", action="store_false")
    p.add_argument("--offline_cache_mode", choices=["entropy_approx", "entropy", "unc", "none"], default="none")
    p.add_argument("--offline_cache_dir", type=str, default=None)
    p.add_argument("--offline_cache_force_hash", type=str, default=os.getenv("OFFLINE_CACHE_FORCE_HASH"))
    p.add_argument("--output_dir", type=str, required=True)
    # Keep small by default to avoid OOM on 24GB cards with 8B teachers.
    p.add_argument("--offline_cache_batch_size", type=int, default=1)
    p.add_argument("--entropy_approx_m", type=int, default=12)
    p.add_argument("--rs_vocab_samples", type=int, default=64)
    p.add_argument("--rs_vocab_beta", type=float, default=1.0)
    p.add_argument("--H_hat_u8", action="store_true", default=True)
    p.add_argument("--kd_temperature", type=float, default=1.0)
    p.add_argument("--distill_type", type=str, default="top-k-tok")
    return p.parse_args()


def _build_dataset(cfg: TrainingConfig, tok, base_seed: int):
    if all(str(p).endswith(".jsonl") for p in cfg.datasets):
        aime = AIMEJsonl([Path(p) for p in cfg.datasets])
        raw_texts = [aime[i]["prompt"] for i in range(len(aime))]
    else:
        if cfg.datasets[0].lower() == "fineweb":
            budget = int(getattr(cfg, "fineweb_tokens", 50_000_000))
            print(f"Loading FineWeb-Edu subset with {budget:,} tokens, seed {base_seed}")
            cached_examples = load_fineweb_subset(
                tok,
                max_tokens=budget,
                seed=base_seed,
                max_seq_len=cfg.max_seq_len,
                packing_enabled=bool(getattr(cfg, "enable_packing", True)),
            )
            raw_texts = [ex["prompt"] for ex in cached_examples]
        else:
            print(f"Loading Hugging Face dataset: {cfg.datasets[0]}")
            hf_dataset = (
                load_dataset(cfg.datasets[0], cfg.dataset_config)["train"]
                if cfg.dataset_config
                else load_dataset(cfg.datasets[0])["train"]
            )
            prompt_col = cfg.prompt_col or "prompt"
            answer_col = cfg.answer_col
            print(f"Using columns - prompt: '{prompt_col}', answer: '{answer_col}'")
            raw_texts = []
            for ex in hf_dataset:
                prompt_text = ex[prompt_col]
                if answer_col is not None and answer_col in ex and ex[answer_col] is not None:
                    raw_texts.append(f"{prompt_text}\n{ex[answer_col]}")
                else:
                    raw_texts.append(prompt_text)

    if getattr(cfg, "enable_packing", True):
        dataset = PackedTokenDataset(raw_texts, tok, cfg.max_seq_len)
    else:
        dataset = PromptDataset(raw_texts)

    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty. Reduce max_seq_len or provide longer input texts.")
    return dataset


def main() -> int:
    args = _parse_args()

    cfg = TrainingConfig(
        seed=args.seed,
        teacher_model=args.teacher_model,
        student_model=args.student_model,
        datasets=args.datasets,
        dataset_config=args.dataset_config,
        prompt_col=args.prompt_col,
        answer_col=args.answer_col,
        fineweb_tokens=args.fineweb_tokens,
        max_seq_len=args.max_seq_len,
        enable_packing=args.enable_packing,
        output_dir=args.output_dir,
        offline_cache=True,
        offline_cache_mode=args.offline_cache_mode,
        offline_cache_dir=args.offline_cache_dir,
        offline_cache_force_hash=args.offline_cache_force_hash,
        offline_cache_batch_size=args.offline_cache_batch_size,
        entropy_approx_m=args.entropy_approx_m,
        rs_vocab_samples=args.rs_vocab_samples,
        rs_vocab_beta=args.rs_vocab_beta,
        H_hat_u8=args.H_hat_u8,
        kd_temperature=args.kd_temperature,
        distill_type=args.distill_type,
        ddp_offline=True,
    )

    ddp_ctx = setup_distributed_context(cfg)
    rank = ddp_ctx.rank
    world_size = ddp_ctx.world_size
    if world_size <= 1:
        raise RuntimeError("This script must be launched with torchrun (WORLD_SIZE>1).")

    torch.cuda.set_device(ddp_ctx.local_rank)
    if ddp_ctx.is_main_rank:
        print(f"[ddp-cache] world_size={world_size} local_rank={ddp_ctx.local_rank}")

    base_seed = int(cfg.seed)
    _seed_everything(base_seed + rank)

    # Ensure a clean cache directory per run when requested.
    if cfg.offline_cache_dir:
        cache_dir = Path(cfg.offline_cache_dir)
        if ddp_ctx.is_main_rank:
            shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
        distributed_barrier()

    tok = AutoTokenizer.from_pretrained(
        cfg.teacher_model,
        use_fast=False,
        trust_remote_code=True,
        local_files_only=False,
        padding_side="left",
    )
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token

    dataset = _build_dataset(cfg, tok, base_seed)
    collate = DistillCollator(tok, cfg.max_seq_len)

    teacherless_modes = {"vanilla", "top-k-tok", "bucket", "random"}
    if getattr(cfg, "offline_cache_mode", "entropy") == "unc":
        teacherless_modes = teacherless_modes | {"atkd"}

    cache_plan = plan_offline_cache(
        cfg,
        tok,
        len(dataset),
        is_main_rank=ddp_ctx.is_main_rank,
        teacherless_modes=teacherless_modes,
    )

    teacher = None
    teacher_device = torch.device("cpu")
    if cache_plan.teacher_required:
        if ddp_ctx.is_main_rank or not cache_plan.teacher_rank0_only:
            print(f"[ddp-cache][rank {rank}] Loading teacher on cuda:{ddp_ctx.local_rank}")
        if (not cache_plan.teacher_rank0_only) or rank == 0:
            teacher_device = torch.device(f"cuda:{ddp_ctx.local_rank}")
            teacher = load_model(cfg.teacher_model, device_map=ddp_ctx.local_rank, quant_bits=cfg.teacher_quant_bits)
            teacher.eval()

    execute_cache_plan(
        cache_plan,
        config=cfg,
        tok=tok,
        packed_dataset=dataset,
        collate_fn=collate,
        teacher=teacher,
        teacher_inputs_device=teacher_device,
        seed_offset=base_seed + rank,
        sanitize_logits_fn=AmpOomMixin._sanitize_logits,
        is_main_rank=ddp_ctx.is_main_rank,
        teacherless_modes=teacherless_modes,
    )

    destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
