import argparse
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.utils.hub import cached_file

from sekd.config import TrainingConfig
from sekd.run_registry import (
    compute_params_hash,
    upsert_run_start,
    mark_trained,
    exists as registry_exists,
    normalize_params,
    get_entry,
)
from sekd.data.dataset import (
    AIMEJsonl,
    DistillCollator,
    PackedTokenDataset,
    PromptDataset,
)
from sekd.models.loader import load_model
from sekd.distill import Distiller
from sekd.distill.trainer import TrainingTimeLimitReached
from sekd.distill._mixins.amp_oom import AmpOomMixin
from sekd.training.entrypoint_utils import (
    load_teacher_with_fallback,
    load_fineweb_subset,
)
from sekd.training.distributed import (
    create_distributed_sampler,
    distributed_barrier,
    distributed_broadcast_object_list,
    destroy_distributed,
    is_rank0,
    setup_distributed_context,
)
from sekd.training.offline_cache import (
    CacheBuildResult,
    CachePlan,
    execute_cache_plan,
    plan_offline_cache,
)

TIME_LIMIT_EXIT_CODE = int(os.environ.get("TIME_LIMIT_EXIT_CODE", "66"))


def _auto_torchrun_cache_enabled() -> bool:
    raw = os.environ.get("AUTO_TORCHRUN_OFFLINE_CACHE", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.strip():
        return len([v for v in visible.split(",") if v.strip() != ""])
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def _run_offline_cache_torchrun(config: TrainingConfig) -> None:
    repo_root = Path(__file__).resolve().parent
    output_dir = (
        Path(getattr(config, "output_dir", repo_root / "results"))
        / "offline_cache_build"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "tools/build_offline_cache_ddp.py",
        "--seed",
        str(getattr(config, "seed", 0)),
        "--teacher_model",
        str(getattr(config, "teacher_model", "")),
        "--student_model",
        str(getattr(config, "student_model", "")),
        "--distill_type",
        str(getattr(config, "distill_type", "")),
        "--max_seq_len",
        str(getattr(config, "max_seq_len", 512)),
        "--output_dir",
        str(output_dir),
    ]

    datasets = getattr(config, "datasets", None)
    if datasets:
        cmd.append("--datasets")
        cmd.extend([str(d) for d in datasets])

    dataset_config = getattr(config, "dataset_config", None)
    if dataset_config:
        cmd.extend(["--dataset_config", str(dataset_config)])

    prompt_col = getattr(config, "prompt_col", None)
    if prompt_col:
        cmd.extend(["--prompt_col", str(prompt_col)])

    answer_col = getattr(config, "answer_col", None)
    if answer_col:
        cmd.extend(["--answer_col", str(answer_col)])

    fineweb_tokens = getattr(config, "fineweb_tokens", None)
    if fineweb_tokens is not None:
        cmd.extend(["--fineweb_tokens", str(fineweb_tokens)])

    offline_cache_mode = getattr(config, "offline_cache_mode", None)
    if offline_cache_mode:
        cmd.extend(["--offline_cache_mode", str(offline_cache_mode)])

    offline_cache_force_hash = getattr(config, "offline_cache_force_hash", None)
    if offline_cache_force_hash:
        cmd.extend(["--offline_cache_force_hash", str(offline_cache_force_hash)])

    offline_cache_batch_size = getattr(config, "offline_cache_batch_size", None)
    if offline_cache_batch_size is not None:
        cmd.extend(["--offline_cache_batch_size", str(offline_cache_batch_size)])

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    if "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = "0,1"

    print("[logits-cache] Auto-building offline cache via torchrun (2 GPUs) ...")
    print("[logits-cache] Command:", " ".join(shlex.quote(c) for c in cmd))
    subprocess.run(cmd, check=True, env=env)


def _format_alpha_ce_suffix(alpha: float) -> str:
    """Format alpha_ce for inclusion in run names (replace '.' with '_')."""
    try:
        alpha_val = float(alpha)
    except (TypeError, ValueError):
        return str(alpha).replace(".", "_")
    if alpha_val.is_integer():
        text = str(int(alpha_val))
    else:
        text = f"{alpha_val:.4f}".rstrip("0").rstrip(".")
        if not text:
            text = "0"
    return text.replace(".", "_")


# Import logging utils with fallback
try:
    from sekd.logging.wandb_utils import create_training_combined_logger
except ImportError:

    def create_training_combined_logger(*args, **kwargs):
        return None


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment variables."""
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_float(name: str) -> Optional[float]:
    """Parse optional float environment variables, returning None on failure."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _env_float_any(*names: str) -> Optional[float]:
    """Return the first successfully parsed float from the provided env names."""
    for name in names:
        value = _env_float(name)
        if value is not None:
            return value
    return None


def _env_int(name: str) -> Optional[int]:
    """Parse optional int environment variables, returning None on failure."""
    value = os.environ.get(name)
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _cli_str_to_bool(value) -> bool:
    """argparse helper that accepts boolean-ish strings like true/false or 1/0."""
    if isinstance(value, bool):
        return value
    if value is None:
        raise argparse.ArgumentTypeError("expected a boolean value")
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


def parse_args_to_config() -> TrainingConfig:
    """Parse command line arguments and create TrainingConfig."""
    parser = argparse.ArgumentParser(description="Entropy-guided KD for LLMs")
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--student_model", required=True)
    parser.add_argument(
        "--distill_category",
        choices=["off_policy", "on_policy"],
        default="off_policy",
        help="High-level training regime: off_policy distillation (default) or on_policy rollouts",
    )
    parser.add_argument(
        "--student_quant_bits",
        type=int,
        choices=[4, 8],
        default=None,
        help="Optionally quantize student for memory (not typical during training)",
    )
    parser.add_argument(
        "--distill_type",
        choices=[
            "vanilla",
            "top-k-tok",
            "top-k-tok-dkd",
            "random",
            "random-dkd",
            "bucket",
            "pos-rs-kd",
            "pos-rs-kd-dkd",
            "atkd",
            "dkd",
        ],
        default="vanilla",
    )
    parser.add_argument(
        "--k_percent", type=int, default=20, help="for top-k-tok and random"
    )
    parser.add_argument(
        "--atkd_hard_percent",
        type=float,
        default=50.0,
        help="For AT-KD: percentage of tokens (by highest teacher uncertainty) treated as hard tokens per batch (default 50%%).",
    )
    parser.add_argument(
        "--atkd_loss_lambda",
        type=float,
        default=0.2,
        help="For AT-KD: λ in L_all = λ*L_easy + (1-λ)*L_hard, λ weight on easy-token KL when combining easy and hard losses (paper default 0.2).",
    )
    parser.add_argument(
        "--normalize_topk_by_length",
        action="store_true",
        default=True,
        help="When set, top-k token quota is based on the batch-average valid length instead of per-example length",
    )
    parser.add_argument(
        "--topk_tok_selection_metric",
        type=str,
        choices=[
            "teacher_entropy",
            "student_entropy",
            "student_ce",
            "kl",
            "reverse-kl",
            "ce_ratio",
            "ce_ratio_entropy",
            "ce_ratio_plus_entropy",
        ],
        default=os.environ.get("TOPK_TOK_SELECTION_METRIC", "teacher_entropy"),
        help="Token ranking metric for top-k selection: teacher_entropy (default), student_entropy, student_ce, kl, reverse-kl, ce_ratio (CE_s/CE_t), ce_ratio_entropy (CE_s/CE_t * H_s), or ce_ratio_plus_entropy (CE_s/CE_t + H_s)",
    )
    parser.add_argument(
        "--selection_curriculum",
        action="store_true",
        default=False,
        help="Enable entropy curriculum that gradually shifts token selection from low-entropy to high-entropy positions",
    )
    parser.add_argument(
        "--no_selection_curriculum",
        dest="selection_curriculum",
        action="store_false",
        help="Disable selection curriculum over entropy-ranked tokens",
    )
    parser.add_argument(
        "--selection_curriculum_steps",
        type=int,
        default=2000,
        help="Number of optimization steps used to progress the selection curriculum",
    )
    parser.add_argument(
        "--selection_curriculum_start",
        type=float,
        default=0.0,
        help="Start fraction within [0,1] for entropy-ranked tokens (0.0 = lowest entropy at curriculum start)",
    )
    parser.add_argument(
        "--selection_curriculum_end",
        type=float,
        default=1.0,
        help="End fraction within [0,1] for entropy-ranked tokens (1.0 = highest entropy by curriculum completion)",
    )
    parser.add_argument(
        "--selection_curriculum_power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized curriculum progress (values >1 keep focus on easy tokens longer)",
    )
    parser.add_argument(
        "--kd_temperature",
        type=float,
        default=1.0,
        help="Unified KD temperature for teacher/student log-softmax and T^2 scaling",
    )

    # Optional gating of distillation based on a frozen pre-distillation student entropy pass
    parser.add_argument(
        "--skip_by_frozen_student",
        dest="skip_by_frozen_student",
        action="store_true",
        default=_env_flag("SKIP_SAMPLES_BY_STUDENT", False),
        help="When set, run a frozen pre-pass of the initial student to compute mean token entropy per sample, then distill only on the top l%% highest-entropy samples during training.",
    )
    # Backwards-compatible alias
    parser.add_argument(
        "--skip_by_frozen_student1",
        dest="skip_by_frozen_student",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--l_percent_samples_to_keep",
        type=float,
        default=_env_float("L_PERCENT_SAMPLES_TO_KEEP") or 20.0,
        help="Percentage (0-100) of samples to keep for distillation (top-entropy) when --skip_by_frozen_student is enabled (default 20).",
    )
    parser.add_argument(
        "--skip_samples_strategy",
        choices=["entropy", "kl", "ce_ratio", "random"],
        default=os.environ.get("SKIP_SAMPLES_STRATEGY", "entropy"),
        help=(
            "Strategy for selecting the l%% samples when --skip_by_frozen_student is enabled: "
            "entropy (default) keeps the top-l%% by frozen-student mean token entropy; "
            "kl keeps the top-l%% by mean token KL divergence between frozen teacher/student "
            "(direction controlled by --kd_objective); "
            "ce_ratio keeps the top-l%% by mean CE_s/(CE_t+eps) between frozen teacher/student; "
            "random keeps a random l%% subset (no pre-pass)."
        ),
    )
    parser.add_argument(
        "--kd_objective",
        choices=["forward", "reverse"],
        default=None,
        help="Direction of KL divergence: forward=teacher||student, reverse=student||teacher",
    )
    dkd_alpha_default = _env_float("DKD_ALPHA")
    dkd_beta_default = _env_float("DKD_BETA")
    parser.add_argument(
        "--unbounded_to_1_loss",
        action="store_true",
        default=_env_flag("UNBOUNDED_TO_1_LOSS", False),
        help="When set, use CE with weight 1.0 instead of (1-alpha_ce)*KD + alpha_ce*CE (exact DKD paper loss).",
    )
    parser.add_argument(
        "--dkd_alpha",
        type=float,
        default=dkd_alpha_default if dkd_alpha_default is not None else 1.0,
        help="Weight for the DKD target-class term (TCKD)",
    )
    parser.add_argument(
        "--dkd_beta",
        type=float,
        default=dkd_beta_default if dkd_beta_default is not None else 8.0,
        help="Weight for the DKD non-target term (NCKD)",
    )
    parser.add_argument(
        "--entropy_approx_temperature",
        type=float,
        default=2.0,
        help="Temperature for offline entropy approximation (and RS-KD proposal)",
    )
    # KD temperature annealing controls
    parser.add_argument(
        "--anneal_kd_temperature",
        action="store_true",
        default=False,
        help="Enable annealing of kd_temperature during training",
    )
    parser.add_argument(
        "--kd_temperature_start",
        type=float,
        default=2.0,
        help="Starting KD temperature when annealing is enabled",
    )
    parser.add_argument(
        "--kd_temperature_end",
        type=float,
        default=1.0,
        help="Final KD temperature when annealing is enabled",
    )
    parser.add_argument(
        "--kd_hold_frac",
        type=float,
        default=0.6,
        help="Fraction of total updates to hold at start temperature before linear decay",
    )
    # RS-KD (position-sampling) hyperparams
    parser.add_argument(
        "--rs_alpha",
        type=float,
        default=1.0,
        help="Exponent on entropy for sampling dist: q(i) ∝ H_i^alpha (alpha∈[0,∞))",
    )
    parser.add_argument(
        "--rs_floor",
        type=float,
        default=1e-6,
        help="Minimum probability floor to avoid huge weights / degeneracy",
    )
    bucket_mode_default = _env_flag("BUCKET_MODE", _env_flag("RS_BUCKET_MODE", False))
    parser.add_argument(
        "--rs_bucket_mode",
        action="store_true",
        default=bucket_mode_default,
        help="When set, restrict pos-rs-kd positions to a percentile bucket before RS sampling",
    )
    parser.add_argument(
        "--rs_bucket_lower_percent",
        type=float,
        default=_env_float_any("BUCKET_LOWER_PERCENT", "RS_BUCKET_LOWER_PERCENT"),
        help="Lower percentile bound (0-100) for pos-rs-kd bucket mode",
    )
    parser.add_argument(
        "--rs_bucket_upper_percent",
        type=float,
        default=_env_float_any("BUCKET_UPPER_PERCENT", "RS_BUCKET_UPPER_PERCENT"),
        help="Upper percentile bound (0-100) for pos-rs-kd bucket mode",
    )
    parser.add_argument(
        "--bucket_lower_percent",
        type=int,
        default=int(float(os.environ.get("BUCKET_LOWER_PERCENT", 70))),
        help="For bucket mode: lower bound percentile (skip bottom X%%)",
    )
    parser.add_argument(
        "--bucket_upper_percent",
        type=int,
        default=int(float(os.environ.get("BUCKET_UPPER_PERCENT", 80))),
        help="For bucket mode: upper bound percentile (skip top Y%%)",
    )
    parser.add_argument(
        "--pos_rs_match_full_kd",
        action="store_true",
        default=False,
        help="When set, positional RS-KD uses unbiased weighting to match full KD gradient expectation",
    )
    parser.add_argument(
        "--topk_debug_dump_path",
        type=str,
        default=None,
        help="Optional JSONL file to log detailed top-k token selection diagnostics",
    )
    parser.add_argument(
        "--topk_debug_dump_limit",
        type=int,
        default=0,
        help="Maximum documents to log for top-k selection debugging (0 disables)",
    )
    parser.add_argument(
        "--cut_after_last_selected",
        action="store_true",
        default=False,
        help="Truncate sequences to (max selected position + 1) before transformer forward, saving compute on trailing tokens",
    )
    parser.add_argument(
        "--logits_on_selected_only",
        action="store_true",
        default=False,
        help="Only compute lm_head projection on selected token positions (avoids full [B,T,V] logits tensor)",
    )
    parser.add_argument(
        "--teacher_selective_lm_head",
        action="store_true",
        default=_env_flag("TEACHER_SELECTIVE_LM_HEAD", False),
        help="Only compute teacher lm_head on positions selected by student entropy (avoids full [B,T,V_teacher] tensor)",
    )
    parser.add_argument(
        "--student_selective_lm_head",
        action="store_true",
        default=_env_flag("STUDENT_SELECTIVE_LM_HEAD", False),
        help="Only compute student lm_head (with grad) on selected positions (avoids full [B,T,V_student] forward+backward)",
    )
    parser.add_argument(
        "--selective_lm_head_same_flow",
        action="store_true",
        default=_env_flag("SELECTIVE_LM_HEAD_SAME_FLOW", False),
        help="Force selective lm_head flow even when both selective flags are false (same-flow control).",
    )
    parser.add_argument(
        "--entropy_streaming_chunk_size",
        type=int,
        default=int(os.environ.get("ENTROPY_STREAMING_CHUNK_SIZE", 128)),
        help="Number of positions per chunk for streaming entropy computation (default 128)",
    )
    parser.add_argument(
        "--log_peak_memory",
        action="store_true",
        default=_env_flag("LOG_PEAK_MEMORY", True),
        help="Log peak GPU memory per step to efficiency CSV and W&B",
    )
    parser.add_argument(
        "--weighted_kd",
        type=_cli_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Weight each token's KL by uncertainty (entropy or unc based on offline_cache_mode)",
    )
    parser.add_argument(
        "--weighted_kd_metric",
        type=str,
        choices=["entropy", "unc", "student_entropy"],
        default=os.environ.get("WEIGHTED_KD_METRIC"),
        help=(
            "Metric used to weight per-token KL when --weighted_kd is enabled. "
            "If unset, defaults to teacher entropy when offline_cache_mode is an entropy mode, otherwise 'unc'."
        ),
    )
    parser.add_argument(
        "--udkd_loss",
        type=_cli_str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help="Use UDKD (Uncertainty-Driven Decoupled KD) loss instead of standard KL divergence",
    )
    parser.add_argument(
        "--udkd_uncertainty_metric",
        type=str,
        choices=["unc", "entropy", "student_entropy", "kl", "reverse_kl"],
        default="unc",
        help="UDKD gate metric: 'unc'=1-p(target), 'entropy'=teacher H/log(V), 'student_entropy'=student H/log(V), 'kl'=KL(teacher||student), 'reverse_kl'=KL(student||teacher)",
    )
    parser.add_argument(
        "--score_token_selection",
        action="store_true",
        default=False,
        help="Rank tokens by composite score (entropy + student CE + KL) when selecting top-k/bucket tokens",
    )
    parser.add_argument(
        "--score_normalize",
        choices=["none", "z", "minmax"],
        default="z",
        help="Normalization applied per example to score components before weighting",
    )
    parser.add_argument(
        "--score_entropy_weight",
        type=float,
        default=1.0,
        help="Weight for teacher entropy component in score-based KD",
    )
    parser.add_argument(
        "--score_ce_weight",
        type=float,
        default=1.0,
        help="Weight for student cross-entropy component in score-based KD",
    )
    parser.add_argument(
        "--score_kl_weight",
        type=float,
        default=1.0,
        help="Weight for teacher-student KL component in score-based KD",
    )
    parser.add_argument(
        "--enable_ce",
        action="store_true",
        default=True,
        help="Enable cross-entropy loss in addition to KD loss",
    )
    parser.add_argument(
        "--enable_ce_on_all_tokens",
        action="store_true",
        default=False,
        help="Apply cross-entropy loss to every valid token even when KD selects a subset",
    )
    parser.add_argument(
        "--alpha_ce",
        type=float,
        default=0.3,
        help="Weight for cross-entropy loss (vs KD loss). Total loss = (1-alpha_ce)*L_KD + alpha_ce*L_CE",
    )
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument(
        "--prompt_col",
        type=str,
        default=None,
        help="name of text prompt column for HF datasets",
    )
    parser.add_argument(
        "--answer_col",
        type=str,
        default=None,
        help="name of answer column for HF datasets",
    )
    parser.add_argument(
        "--fineweb_tokens",
        type=int,
        default=50_000_000,
        help="Token budget when streaming FineWeb-Edu (used when datasets[0] == 'fineweb')",
    )
    parser.add_argument(
        "--disable_packing",
        dest="enable_packing",
        action="store_false",
        help="Disable sequence packing into fixed-length windows",
    )
    parser.add_argument(
        "--enable_packing",
        dest="enable_packing",
        action="store_true",
        help="Enable sequence packing into fixed-length windows",
    )
    parser.set_defaults(enable_packing=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before updating",
    )
    parser.add_argument("--max_seq_len", type=int, default=512)  # to save memory
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="tb",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--log_efficiency_csv",
        action="store_true",
        default=False,
        help="Append total wall-clock, prepass wall-clock, iterations, and FLOPs to a CSV table after training.",
    )
    parser.add_argument(
        "--efficiency_csv_path",
        type=str,
        default=os.environ.get(
            "EFFICIENCY_CSV_PATH", "results/table_efficiency_test.csv"
        ),
        help="CSV output path for efficiency metrics when --log_efficiency_csv is enabled.",
    )
    parser.add_argument(
        "--log_skipping_indices",
        action="store_true",
        default=False,
        help="Append skip-sample indices (per run) to an append-only JSONL file after the prepass.",
    )
    parser.add_argument(
        "--skipping_indices_path",
        type=str,
        default=os.environ.get(
            "SKIPPING_INDICES_PATH", "results/skipping_indices.json"
        ),
        help="JSONL output path for skip-sample indices when --log_skipping_indices is enabled.",
    )
    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=500,
        help="Save checkpoint every N steps (0 to disable)",
    )
    parser.add_argument(
        "--keep_checkpoints",
        type=int,
        default=3,
        help="Number of recent checkpoints to keep",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=os.environ.get("RESUME_FROM_CHECKPOINT"),
        help="Optional path to a checkpoint (.pt) to resume training from.",
    )
    parser.add_argument(
        "--max_train_hours",
        type=float,
        default=_env_float_any("MAX_TRAIN_HOURS", "TRAIN_MAX_HOURS"),
        help="Optional wall-clock limit in hours (checkpoint + exit).",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=None,
        help="(Optional) HF dataset config, e.g. for gsm8k use '--dataset_config main' or 'socratic'",
    )
    parser.add_argument(
        "--offline_cache",
        action="store_true",
        default=True,
        help="Enable offline caching mode: automatically create/use teacher cache for entropy approximation and vocab RS-KD.",
    )
    parser.add_argument(
        "--no_offline_cache",
        dest="offline_cache",
        action="store_false",
        help="Disable offline caching mode (use online teacher forward pass).",
    )
    parser.add_argument(
        "--offline_cache_selected_only",
        action="store_true",
        default=None,
        help="When skip_by_frozen_student is enabled, defer offline cache build and cache only selected samples.",
    )
    parser.add_argument(
        "--no_offline_cache_selected_only",
        dest="offline_cache_selected_only",
        action="store_false",
        help="Disable selected-only offline cache behavior.",
    )
    parser.add_argument(
        "--offline_cache_dir",
        type=str,
        default=None,
        help="Where to store/read the offline teacher cache (defaults under output_dir).",
    )
    parser.add_argument(
        "--offline_cache_force_hash",
        type=str,
        default=os.getenv("OFFLINE_CACHE_FORCE_HASH"),
        help="Force-use a specific offline cache hash under logits_caches/, ignoring signature mismatches.",
    )
    parser.add_argument(
        "--offline_cache_missing_tolerance",
        type=int,
        default=os.getenv("OFFLINE_CACHE_MISSING_TOLERANCE", 100),
        help="Allow up to N missing cache items when force-using a cache hash (default 100).",
    )
    parser.add_argument(
        "--offline_cache_min_hit_rate",
        type=float,
        default=os.getenv("OFFLINE_CACHE_MIN_HIT_RATE", 0.9),
        help="Minimum cache hit rate required when force-using a cache hash (default 0.9).",
    )
    parser.add_argument(
        "--profiler_enabled",
        action="store_true",
        default=bool(_env_flag("PROFILER_ENABLED", False)),
        help="Enable torch.profiler tracing during training.",
    )
    parser.add_argument(
        "--profiler_dir",
        type=str,
        default=os.getenv("PROFILER_DIR"),
        help="Output directory for profiler traces (defaults under results/gpu_util).",
    )
    parser.add_argument(
        "--profiler_wait",
        type=int,
        default=int(os.getenv("PROFILER_WAIT", 1)),
        help="Profiler schedule: wait steps (default 1).",
    )
    parser.add_argument(
        "--profiler_warmup",
        type=int,
        default=int(os.getenv("PROFILER_WARMUP", 1)),
        help="Profiler schedule: warmup steps (default 1).",
    )
    parser.add_argument(
        "--profiler_active",
        type=int,
        default=int(os.getenv("PROFILER_ACTIVE", 3)),
        help="Profiler schedule: active steps (default 3).",
    )
    parser.add_argument(
        "--profiler_repeat",
        type=int,
        default=int(os.getenv("PROFILER_REPEAT", 1)),
        help="Profiler schedule: repeat count (default 1).",
    )
    parser.add_argument(
        "--profiler_record_shapes",
        action="store_true",
        default=bool(_env_flag("PROFILER_RECORD_SHAPES", True)),
        help="Record tensor shapes in profiler (default true).",
    )
    parser.add_argument(
        "--profiler_with_stack",
        action="store_true",
        default=bool(_env_flag("PROFILER_WITH_STACK", True)),
        help="Record stack traces in profiler (default true).",
    )
    parser.add_argument(
        "--profiler_profile_memory",
        action="store_true",
        default=bool(_env_flag("PROFILER_PROFILE_MEMORY", True)),
        help="Profile memory allocations in profiler (default true).",
    )
    parser.add_argument(
        "--profiler_max_steps",
        type=int,
        default=int(os.getenv("PROFILER_MAX_STEPS", 200)),
        help="Max number of profiler steps to record (default 200).",
    )
    parser.add_argument(
        "--offline_cache_batch_size",
        type=int,
        default=4,
        help="Batch size used only during offline cache building (defaults to --batch_size when unset).",
    )
    parser.add_argument(
        "--entropy_cache_approx",
        action="store_true",
        default=False,
        help="Store truncated entropy approximation in the cache (enables Gumbel-Max path). Implies --offline_cache_mode entropy_approx and --offline_cache.",
    )
    parser.add_argument(
        "--no_entropy_cache_approx",
        dest="entropy_cache_approx",
        action="store_false",
        help="Disable truncated entropy approximation (use exact entropy cache).",
    )
    parser.add_argument(
        "--offline_cache_mode",
        choices=["entropy_approx", "entropy", "unc", "none"],
        default="entropy",
        help="Offline cache mode: entropy_approx (truncated), entropy (exact), unc (store target probabilities), or none (store no teacher uncertainty metric).",
    )
    parser.add_argument(
        "--entropy_approx_m",
        type=int,
        default=12,
        help="Top-k for truncated-entropy approximation, m=12 by default.",
    )
    parser.add_argument(
        "--rs_vocab_samples",
        type=int,
        default=64,
        help="How many vocab tokens to sample per position for RS-KD. 36 bytes per position",
    )
    parser.add_argument(
        "--rs_vocab_beta",
        type=float,
        default=1.0,
        help="Proposal exponent: q ∝ p^beta (beta=1 is proportional to p).",
    )

    # On-policy distillation configuration
    parser.add_argument(
        "--on_policy_max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens generated per student rollout (on-policy mode)",
    )
    parser.add_argument(
        "--on_policy_temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for student-generated rollouts (on-policy mode)",
    )
    parser.add_argument(
        "--on_policy_top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling cutoff for student rollouts",
    )
    parser.add_argument(
        "--on_policy_group_size",
        type=int,
        default=1,
        help="Number of student rollouts sampled per prompt in on-policy mode",
    )
    parser.add_argument(
        "--on_policy_reverse_kl_weight",
        type=float,
        default=1.0,
        help="Weight applied to reverse KL (student||teacher) when training on-policy",
    )
    parser.add_argument(
        "--on_policy_forward_kl_weight",
        type=float,
        default=0.0,
        help="Weight applied to forward KL (teacher||student) during on-policy training",
    )
    parser.add_argument(
        "--on_policy_curriculum",
        action="store_true",
        default=False,
        help="Enable curriculum schedule for k_percent during on-policy training",
    )
    parser.add_argument(
        "--on_policy_curriculum_steps",
        type=int,
        default=1000,
        help="Number of optimizer updates to anneal k_percent towards its target value",
    )
    parser.add_argument(
        "--on_policy_curriculum_start_k",
        type=float,
        default=5.0,
        help="Initial k_percent (percentage) when curriculum starts in on-policy mode",
    )
    parser.add_argument(
        "--on_policy_curriculum_power",
        type=float,
        default=1.0,
        help="Exponent applied to normalized curriculum progress (1.0 = linear)",
    )
    parser.add_argument(
        "--on_policy_forward_self_norm",
        dest="on_policy_forward_self_norm",
        action="store_true",
        default=True,
        help="Use self-normalized importance weights for forward KL estimator in on-policy mode",
    )
    parser.add_argument(
        "--on_policy_no_forward_self_norm",
        dest="on_policy_forward_self_norm",
        action="store_false",
        help="Disable self-normalized weights when estimating forward KL in on-policy mode",
    )
    parser.add_argument(
        "--on_policy_do_sample",
        dest="on_policy_do_sample",
        action="store_true",
        default=True,
        help="Enable stochastic sampling for student rollouts (default)",
    )
    parser.add_argument(
        "--on_policy_no_sample",
        dest="on_policy_do_sample",
        action="store_false",
        help="Disable sampling and use greedy decoding for student rollouts",
    )
    parser.add_argument(
        "--enable_cuts_in_the_middle_for_on_policy",
        dest="enable_cuts_in_the_middle_for_on_policy",
        action="store_true",
        help="Sample middle cut points for on-policy FineWeb prompts before generation.",
    )
    parser.add_argument(
        "--disable_cuts_in_the_middle_for_on_policy",
        dest="enable_cuts_in_the_middle_for_on_policy",
        action="store_false",
        help="Disable middle cut sampling for on-policy rollouts.",
    )
    parser.add_argument(
        "--on_policy_cut_min_tokens",
        type=int,
        default=12,
        help="Minimum number of rollout tokens to force after a middle cut (on-policy mode).",
    )
    parser.add_argument(
        "--on_policy_cut_max_tokens",
        type=int,
        default=32,
        help="Maximum number of rollout tokens to force after a middle cut (on-policy mode).",
    )
    parser.add_argument(
        "--on_policy_cut_min_context",
        type=int,
        default=128,
        help="Minimum prefix length retained before the sampled cut (on-policy mode).",
    )
    parser.set_defaults(enable_cuts_in_the_middle_for_on_policy=True)

    # Global-Level Selection (GLS) over tokens - only impacts top-k-tok when enabled
    parser.add_argument(
        "--gls_enabled",
        action="store_true",
        default=False,
        help="Enable global-level selection FIFO queue (only impacts top-k-tok)",
    )
    parser.add_argument(
        "--gls_queue_size",
        type=int,
        default=30000,
        help="Capacity of GLS FIFO queue for computing global threshold",
    )
    parser.add_argument(
        "--gls_log_threshold",
        action="store_true",
        default=False,
        help="Log the GLS threshold each time it's computed",
    )
    # Unified upsert registry controls
    parser.add_argument(
        "--runs_registry_validation",
        type=str,
        default="results/runs_validation.json",
        help="Path to the validation split runs JSON registry.",
    )
    parser.add_argument(
        "--runs_registry_test",
        type=str,
        default="results/runs_test.json",
        help="Path to the test split runs JSON registry.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        default=False,
        help="If set, run even if an identical-params hash already exists in the registry.",
    )
    # Reproducibility
    default_seed = int(os.environ.get("SEED", "1337"))
    default_det = bool(int(os.environ.get("DETERMINISTIC", "0")))
    parser.add_argument(
        "--seed", type=int, default=default_seed, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=default_det,
        help="Enable deterministic algorithms (may slow down, sets cudnn.deterministic and use_deterministic_algorithms)",
    )
    args = parser.parse_args()

    if args.resume_from_checkpoint is not None:
        resume_val = str(args.resume_from_checkpoint).strip()
        args.resume_from_checkpoint = resume_val if resume_val else None

    if args.max_train_hours is not None:
        try:
            if float(args.max_train_hours) <= 0:
                args.max_train_hours = None
        except (TypeError, ValueError):
            args.max_train_hours = None

    if args.offline_cache_selected_only is None:
        env_selected_only = os.environ.get("OFFLINE_CACHE_SELECTED_ONLY")
        if env_selected_only is not None:
            args.offline_cache_selected_only = _env_flag(
                "OFFLINE_CACHE_SELECTED_ONLY", False
            )
        else:
            args.offline_cache_selected_only = bool(args.skip_by_frozen_student)

    if args.kd_objective is None:
        args.kd_objective = (
            "reverse" if args.distill_category == "on_policy" else "forward"
        )

    if args.distill_category == "on_policy":
        # Force settings compatible with on-policy rollouts unless user explicitly overrides later.
        args.offline_cache = False
        if getattr(args, "enable_ce", True):
            args.enable_ce = False
        if getattr(args, "alpha_ce", None) is None or args.alpha_ce > 0:
            args.alpha_ce = 0.0
        if args.on_policy_group_size < 1:
            args.on_policy_group_size = 1
        if args.on_policy_max_new_tokens < 1:
            args.on_policy_max_new_tokens = 1
        args.on_policy_cut_min_tokens = max(1, args.on_policy_cut_min_tokens)
        if args.on_policy_cut_max_tokens < args.on_policy_cut_min_tokens:
            args.on_policy_cut_max_tokens = args.on_policy_cut_min_tokens
        args.on_policy_cut_min_context = max(1, args.on_policy_cut_min_context)
    # Convert argparse Namespace to TrainingConfig
    return TrainingConfig(**vars(args))


class DistillationEntrypoint:
    """Coordinate the end-to-end distillation launch."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.ddp_ctx = setup_distributed_context(config)
        self.ddp_world_size = self.ddp_ctx.world_size
        self.ddp_rank = self.ddp_ctx.rank
        self.ddp_local_rank = self.ddp_ctx.local_rank
        self.is_main_rank = self.ddp_ctx.is_main_rank

        self.params_dict = self._config_to_dict()
        self.launch_seed = _env_int("AUTOPILOT_LAUNCH_SEED")
        self.base_seed = (
            self.launch_seed if self.launch_seed is not None else int(self.config.seed)
        )
        self.seed_offset = self.base_seed + self.ddp_rank

        self.registry_paths_to_update: List[Path] = []
        self.params_hash: Optional[str] = None
        self.experiment_name: Optional[str] = None
        self.display_name: Optional[str] = None
        self.primary_registry_path: Optional[Path] = None
        self.job_id: Optional[str] = None

        self.local_avail: List[int] = []
        self.student_local: Optional[int] = None
        self.student_device = torch.device("cpu")
        self.teacher_locals: List[int] = []
        self.teacher_student_exclusion: Optional[int] = None
        self.available_gpu_priority: List[int] = []
        self.gpu_mem_snapshot: List[tuple[int, float, float]] = []
        self.large_gpu_threshold_gb = 30.0

        self.tok = None
        self.packed_dataset = None
        self.collate = None
        self.dl = None
        self.dataset_size: Optional[int] = None

        self.cache_plan = None
        self.cache_ready = False
        self.teacher_required = False
        self.teacher_rank0_only = False
        self.teacherless_modes = {
            "vanilla",
            "top-k-tok",
            "bucket",
            "random",
        }
        if getattr(self.config, "offline_cache_mode", "entropy") == "unc":
            self.teacherless_modes = self.teacherless_modes | {"atkd"}
        if self.config.distill_category == "on_policy":
            # Always require a live teacher for on-policy rollouts
            self.teacherless_modes = set()

        self.teacher = None
        self.teacher_inputs_device = torch.device("cpu")
        self.teacher_device_str: Optional[str] = None

        self.student = None
        self.student_dtype: Optional[torch.dtype] = None
        self.combined_logger = None
        self.distiller: Optional[Distiller] = None
        self.defer_cache_build = False

    def run(self) -> None:
        self._capture_display_name()
        self._log_launch()
        self._seed_everything()
        self._configure_deterministic_mode()
        self._configure_tmpdirs()
        self._configure_cuda_backend()
        self._prepare_registry()
        self._persist_run_params()
        self._plan_devices()
        self._prepare_data()
        self._plan_teacher_and_cache()
        if self._switch_to_single_process_after_cache_build():
            return
        self._load_student()
        self._initialize_logger()
        self.distiller = self._build_distiller()
        try:
            self._train_distiller()
        except TrainingTimeLimitReached as exc:
            if self.is_main_rank:
                print(f"[timelimit] {exc}")
                if getattr(exc, "checkpoint_path", None):
                    print(f"[timelimit] checkpoint_path={exc.checkpoint_path}")
            self._finalize()
            sys.exit(TIME_LIMIT_EXIT_CODE)
        self._post_training()
        self._finalize()

    def _capture_display_name(self) -> None:
        """Capture optional display name for logging and registry metadata."""
        display_name_raw = os.getenv("RUN_DISPLAY_NAME")
        if display_name_raw is not None:
            cleaned = display_name_raw.strip()
            self.display_name = cleaned if cleaned else None

    def _config_to_dict(self):
        try:
            return self.config.model_dump()
        except Exception:
            return vars(self.config)

    def _estimate_model_weight_bytes(self, model_name: str) -> Optional[int]:
        """Approximate raw weight size by inspecting cached weight index files."""
        index_candidates: Tuple[Tuple[str, str], ...] = (
            ("model.safetensors.index.json", "model.safetensors"),
            ("pytorch_model.bin.index.json", "pytorch_model.bin"),
        )
        for index_name, weight_name in index_candidates:
            index_path: Optional[str] = None
            try:
                index_path = cached_file(model_name, index_name, local_files_only=True)
            except Exception:
                index_path = None
            if not index_path:
                continue
            try:
                data = json.loads(Path(index_path).read_text())
                metadata = data.get("metadata") or {}
                total_size = metadata.get("total_size")
                if total_size:
                    return int(total_size)
            except Exception:
                pass
            weight_path = None
            try:
                weight_path = cached_file(
                    model_name, weight_name, local_files_only=True
                )
            except Exception:
                weight_path = None
            if not weight_path:
                continue
            try:
                return int(Path(weight_path).stat().st_size)
            except Exception:
                continue
        return None

    def _build_training_dataloader(
        self, *, use_distributed_sampler: bool
    ) -> DataLoader:
        if self.packed_dataset is None or self.collate is None:
            raise RuntimeError(
                "Dataset or collator not initialized before building dataloader."
            )

        gen = torch.Generator()
        gen.manual_seed(self.base_seed)

        def _seed_worker(worker_id: int):
            seed = self.base_seed + worker_id + self.ddp_rank * 1000
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        worker_count = min(8, os.cpu_count() or 1)
        persistent_workers = True
        if self.config.distill_category == "on_policy":
            worker_count = 0
            persistent_workers = False

        sampler = None
        if use_distributed_sampler:
            sampler = create_distributed_sampler(
                self.packed_dataset,
                config=self.config,
                seed=self.seed_offset,
                shuffle=True,
                drop_last=False,
            )

        return DataLoader(
            self.packed_dataset,
            batch_size=self.config.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=self.collate,
            num_workers=worker_count,
            pin_memory=True,
            persistent_workers=persistent_workers,
            generator=gen,
            worker_init_fn=_seed_worker,
        )

    def _switch_to_single_process_after_cache_build(self) -> bool:
        if getattr(self, "defer_cache_build", False):
            return False
        if self.ddp_world_size <= 1:
            return False

        # Clean up teacher model on non-rank-0 before barrier to free GPU memory
        if not self.is_main_rank and self.teacher is not None:
            del self.teacher
            self.teacher = None
            import gc

            gc.collect()
            torch.cuda.empty_cache()

        distributed_barrier()
        destroy_distributed()

        if not self.is_main_rank:
            print(
                "[ddp-offline] Cache build complete. Exiting non-main rank.", flush=True
            )
            return True

        # Rank 0 continues with single-process training
        self.config.ddp_world_size = 1
        self.config.ddp_rank = 0
        self.config.ddp_local_rank = 0
        self.ddp_world_size = 1
        self.ddp_rank = 0
        self.ddp_local_rank = 0
        self.is_main_rank = True

        self.dl = self._build_training_dataloader(use_distributed_sampler=False)
        return False

    def _estimate_role_expected_vram(
        self, weight_bytes: Optional[int], role: str
    ) -> Optional[float]:
        """Heuristically expand raw weights into expected peak VRAM."""
        if not weight_bytes or weight_bytes <= 0:
            return None
        weight_gb = weight_bytes / (1024**3)
        if role == "student":
            # Weights + grads + optimizer + activations. 8-bit optimizer reduces overhead slightly.
            multiplier = 3.1
            if getattr(self.config, "student_quant_bits", None) in (4, 8):
                multiplier = 2.4
        else:
            # Teacher runs inference only; allow modest overhead for KV caches and buffers.
            multiplier = 1.3
        return weight_gb * multiplier

    def _determine_heavier_role(self) -> str:
        """Return which role (student/teacher) is expected to require more VRAM."""
        student_bytes = self._estimate_model_weight_bytes(self.config.student_model)
        teacher_bytes = self._estimate_model_weight_bytes(self.config.teacher_model)
        student_vram = self._estimate_role_expected_vram(student_bytes, "student")
        teacher_vram = self._estimate_role_expected_vram(teacher_bytes, "teacher")

        heavier = "student"
        if teacher_vram is not None and (
            student_vram is None or teacher_vram > student_vram * 1.05
        ):
            heavier = "teacher"

        if self.is_main_rank and (student_vram is not None or teacher_vram is not None):
            sv = f"{student_vram:.1f}" if student_vram is not None else "?"
            tv = f"{teacher_vram:.1f}" if teacher_vram is not None else "?"
            print(
                f"[device-planning] VRAM estimate (GB) → student≈{sv}, teacher≈{tv}; treating {heavier} as heavier.",
                flush=True,
            )
        return heavier

    def _log_launch(self) -> None:
        # Print all CLI parameters at startup
        print("[launch] python", " ".join(sys.argv), flush=True)
        if self.is_main_rank:
            print("=" * 80)
            print("TRAINING CONFIGURATION")
            print("=" * 80)
            for key, value in sorted(self.params_dict.items()):
                print(f"  {key:30s} = {value}")
            print("=" * 80)
            print()
            if self.display_name:
                print(f"[display] RUN_DISPLAY_NAME: {self.display_name}", flush=True)
            if self.launch_seed is not None:
                print(
                    f"[seed] AUTOPILOT_LAUNCH_SEED={self.launch_seed} (registry/config seed={self.config.seed})",
                    flush=True,
                )

    def _seed_everything(self) -> None:
        # Global seeding
        random.seed(self.seed_offset)
        np.random.seed(self.seed_offset)
        torch.manual_seed(self.seed_offset)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed_offset)

    def _configure_deterministic_mode(self) -> None:
        # Optional deterministic mode
        if getattr(self.config, "deterministic", False):
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            torch.use_deterministic_algorithms(True)

    def _configure_tmpdirs(self) -> None:
        # Prefer node-local tmp to avoid NFS .nfs tombstones on cleanup
        # If SLURM provides a local scratch (e.g., /dev/shm), use it; fall back to repo tmp
        node_tmp = os.environ.get("TMPDIR")
        if not node_tmp:
            shm_candidate = f"/dev/shm/{os.environ.get('USER', 'user')}.sekd.{os.environ.get('SLURM_JOB_ID', 'local')}"
            try:
                os.makedirs(shm_candidate, exist_ok=True)
                node_tmp = shm_candidate
            except Exception:
                node_tmp = f"/tmp/sekd/{os.environ.get('USER', 'user')}"
            os.environ["TMPDIR"] = node_tmp
        # Point HF caches to tmp (can still hit shared cache via symlink if needed)
        os.environ.setdefault("HF_HOME", os.path.join(node_tmp, "hf"))
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

    def _configure_cuda_backend(self) -> None:
        # Set CUDA memory management settings for better memory efficiency
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Speed optimizations (safe) or deterministic fallback
        if getattr(self.config, "deterministic", False):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    def _prepare_registry(self) -> None:
        # ----------------- runs registry preflight -----------------
        default_test_path = Path("results/runs_test.json")
        default_validation_path = Path("results/runs_validation.json")
        registry_path_test = Path(
            getattr(self.config, "runs_registry_test", str(default_test_path))
        )
        registry_path_validation = Path(
            getattr(
                self.config, "runs_registry_validation", str(default_validation_path)
            )
        )

        if (
            registry_path_validation == default_validation_path
            and registry_path_test != default_test_path
        ):
            registry_path_validation = registry_path_test.with_name(
                "runs_validation.json"
            )

        seen_paths: Set[str] = set()
        for candidate in (registry_path_validation, registry_path_test):
            resolved = Path(candidate).resolve()
            key = str(resolved)
            if key in seen_paths:
                continue
            seen_paths.add(key)
            self.registry_paths_to_update.append(resolved)

        display_name_raw = os.getenv("RUN_DISPLAY_NAME")
        self.display_name = display_name_raw.strip() if display_name_raw else None

        self.params_hash = compute_params_hash(self.params_dict)
        self.primary_registry_path = Path(registry_path_test).resolve()

        if self.is_main_rank:
            if registry_exists(
                self.primary_registry_path, self.params_hash
            ) and not getattr(self.config, "override", False):
                entry = get_entry(self.primary_registry_path, self.params_hash)
                completed_eval = bool(entry and entry.get("completed_eval"))
                runs_info = entry.get("runs", {}) if entry else {}
                existing_output_dir = (
                    (runs_info.get("train") or {}).get("output_dir")
                    if runs_info
                    else None
                )
                print(
                    f"[registry] Run with identical parameters already exists (id={self.params_hash}). Use --override to force rerun. Exiting gracefully."
                )
                needs_eval = not completed_eval
                meta_output_dir = existing_output_dir or ""
                print(
                    f"[registry] duplicate params_hash={self.params_hash} needs_eval={int(needs_eval)} output_dir={meta_output_dir}"
                )
                sys.exit(11 if needs_eval else 10)

        self._create_experiment_name()

        if self.is_main_rank:
            cli_args = " ".join(shlex.quote(arg) for arg in sys.argv)
            for reg_path in self.registry_paths_to_update:
                upsert_run_start(
                    reg_path,
                    self.params_dict,
                    experiment_name=self.experiment_name,
                    job_id=self.job_id,
                    model_output_dir=self.config.output_dir,
                    launch_args=cli_args,
                    display_name=self.display_name,
                )

    def _create_experiment_name(self) -> None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        self.job_id = os.getenv("SLURM_JOB_ID", "local")
        name = f"distill-{self.config.distill_type}-{current_date}_{self.job_id}"
        if self.config.distill_type in {"top-k-tok", "random", "random-dkd"}:
            name += f"_k={self.config.k_percent}"
        elif self.config.distill_type == "bucket":
            name += f"_bucket={self.config.bucket_lower_percent}-{self.config.bucket_upper_percent}"
        suffix_parts = []
        if self.config.distill_category == "on_policy":
            suffix_parts.append("on_policy")
        alpha_value = getattr(self.config, "alpha_ce", None)
        if alpha_value is not None:
            formatted = _format_alpha_ce_suffix(alpha_value)
            suffix_parts.append(f"ce{formatted}")
        if suffix_parts:
            name += "_" + "_".join(suffix_parts)
        self.experiment_name = name

    def _persist_run_params(self) -> None:
        # Persist normalized params and hash alongside model outputs for downstream eval
        if not self.is_main_rank or self.params_hash is None:
            return
        try:
            Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
            params_out = Path(self.config.output_dir) / "run_params.json"
            with open(params_out, "w", encoding="utf-8") as f:
                import json as _json

                payload = {
                    "id": self.params_hash,
                    "params": normalize_params(self.params_dict),
                }
                if self.display_name:
                    payload["display_name"] = self.display_name
                _json.dump(payload, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[registry] Warning: failed to write run_params.json: {e}")

    def _plan_devices(self) -> None:
        # ----------------- teacher / student device planning -----------------
        device_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if device_count == 0:
            raise RuntimeError("No CUDA devices available.")

        total_vram_gb = 0
        for i in range(device_count):
            vram_bytes = torch.cuda.get_device_properties(i).total_memory
            total_vram_gb += vram_bytes / (1024**3)
        if self.is_main_rank:
            print(
                "Success: Detected "
                + str(device_count)
                + " GPUs with "
                + str(round(total_vram_gb, 1))
                + " GB total VRAM available."
            )

        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if cvd:
            physical = [int(x) for x in cvd.split(",") if x != ""]
            local_avail = list(range(len(physical)))
        else:
            physical = list(range(device_count))
            local_avail = list(range(device_count))

        gpu_mem_snapshot: List[tuple[int, float, float]] = []
        for idx in local_avail:
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            except RuntimeError:
                try:
                    prev_device = torch.cuda.current_device()
                except RuntimeError:
                    prev_device = None
                torch.cuda.set_device(idx)
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                if prev_device is not None:
                    torch.cuda.set_device(prev_device)
            free_gb = free_bytes / (1024**3)
            total_gb = total_bytes / (1024**3)
            gpu_mem_snapshot.append((idx, free_gb, total_gb))

        if not gpu_mem_snapshot:
            raise RuntimeError(
                "No CUDA devices discovered after applying visibility filters."
            )

        gpu_mem_snapshot.sort(key=lambda x: x[1], reverse=True)
        self.gpu_mem_snapshot = gpu_mem_snapshot
        self.available_gpu_priority = [idx for idx, _, _ in gpu_mem_snapshot]
        self.local_avail = list(self.available_gpu_priority)

        chosen_gpu = self.available_gpu_priority[0]
        self.student_local = chosen_gpu
        self.student_device = torch.device(f"cuda:{self.student_local}")
        self.teacher_locals = [self.student_local]
        self.teacher_student_exclusion = None

        if self.is_main_rank:
            free_summary = ", ".join(
                f"{idx}:{free:.1f}/{total:.1f}GB"
                for idx, free, total in gpu_mem_snapshot
            )
            print(
                f"[device-planning] GPU free memory snapshot (prioritized): [{free_summary}]",
                flush=True,
            )

        if self.ddp_world_size > 1:
            if self.ddp_local_rank >= len(self.local_avail):
                raise RuntimeError(
                    f"LOCAL_RANK={self.ddp_local_rank} exceeds available GPUs {self.local_avail}"
                )
            self.student_local = self.ddp_local_rank
            self.student_device = torch.device(f"cuda:{self.student_local}")
            self.teacher_locals = [self.student_local]
            self.teacher_student_exclusion = None
        else:
            self.student_local = self.local_avail[0]
            self.student_device = torch.device(f"cuda:{self.student_local}")
            self.teacher_locals = [self.student_local]
            self.teacher_student_exclusion = None

        if self.is_main_rank:
            print(f"CUDA_VISIBLE_DEVICES: {cvd}")
            print(
                f"Available GPUs (local priority order): {self.available_gpu_priority}"
            )
        if self.is_main_rank and len(self.available_gpu_priority) > 1:
            print(
                f"[device-planning] Initial student GPU candidate: {self.student_local} (priority order will be reconciled later)"
            )
        print(f"[rank {self.ddp_rank}] Student GPU (local): {self.student_local}")
        print(f"[rank {self.ddp_rank}] Teacher GPUs (local): {self.teacher_locals}")

    def _apply_device_policy(self) -> None:
        """
        Decide whether to co-locate teacher and student on a single large GPU or split across two GPUs.
        """
        if not self.available_gpu_priority:
            raise RuntimeError("GPU inventory unavailable when applying device policy.")

        mem_map: Dict[int, tuple[float, float]] = {
            idx: (free, total) for idx, free, total in self.gpu_mem_snapshot
        }
        threshold = self.large_gpu_threshold_gb
        teacher_needed = bool(self.teacher_required)
        heavier_role = "student"
        if teacher_needed:
            heavier_role = self._determine_heavier_role()

        # Identify first GPU meeting the large-memory threshold
        large_gpu_idx: Optional[int] = None
        for idx in self.available_gpu_priority:
            free_gb = mem_map.get(idx, (0.0, 0.0))[0]
            if free_gb >= threshold:
                large_gpu_idx = idx
                break

        if teacher_needed:
            if large_gpu_idx is not None:
                self.student_local = large_gpu_idx
                self.student_device = torch.device(f"cuda:{self.student_local}")
                self.teacher_locals = [large_gpu_idx]
                self.teacher_student_exclusion = None
                self.local_avail = [large_gpu_idx]
                if self.is_main_rank:
                    print(
                        f"[device-planning] Teacher required → co-locating teacher and student on cuda:{large_gpu_idx} "
                        f"(>= {threshold:.0f}GB free).",
                        flush=True,
                    )
            else:
                if len(self.available_gpu_priority) < 2:
                    raise RuntimeError(
                        "Teacher required but no GPU meets the large-memory threshold and fewer than two GPUs are available."
                    )
                first_gpu, second_gpu = (
                    self.available_gpu_priority[0],
                    self.available_gpu_priority[1],
                )
                if heavier_role == "teacher":
                    teacher_gpu, student_gpu = first_gpu, second_gpu
                else:
                    student_gpu, teacher_gpu = first_gpu, second_gpu
                self.student_local = student_gpu
                self.student_device = torch.device(f"cuda:{self.student_local}")
                self.teacher_locals = [teacher_gpu]
                self.teacher_student_exclusion = self.student_local
                self.local_avail = [student_gpu, teacher_gpu]
                if self.is_main_rank:
                    print(
                        f"[device-planning] Teacher required with no GPU >= {threshold:.0f}GB free → "
                        f"assigning student to cuda:{student_gpu} and teacher to cuda:{teacher_gpu} (heavier role: {heavier_role}).",
                        flush=True,
                    )
        else:
            student_gpu = self.available_gpu_priority[0]
            self.student_local = student_gpu
            self.student_device = torch.device(f"cuda:{self.student_local}")
            self.teacher_locals = [student_gpu]
            self.teacher_student_exclusion = None
            self.local_avail = [student_gpu]
            if self.is_main_rank:
                print(
                    f"[device-planning] Teacher not required → using single GPU cuda:{student_gpu} for the student.",
                    flush=True,
                )

    def _prepare_data(self) -> None:
        # ---- Tokenizer and dataset preparation (before teacher load) ----
        self.tok = AutoTokenizer.from_pretrained(
            self.config.teacher_model,
            use_fast=False,
            trust_remote_code=True,
            local_files_only=False,
            padding_side="left",  # Required for decoder-only models during generation
        )
        if self.tok.pad_token_id is None and self.tok.eos_token is not None:
            self.tok.pad_token = self.tok.eos_token

        if self.config.distill_category == "on_policy" and getattr(
            self.config, "enable_packing", True
        ):
            try:
                self.config.enable_packing = False
            except Exception:
                object.__setattr__(self.config, "enable_packing", False)

        if all(p.endswith(".jsonl") for p in self.config.datasets):
            aime = AIMEJsonl([Path(p) for p in self.config.datasets])
            raw_texts = [aime[i]["prompt"] for i in range(len(aime))]
        else:
            if self.config.datasets[0].lower() == "fineweb":
                budget = int(getattr(self.config, "fineweb_tokens", 50_000_000))
                print(
                    f"Loading FineWeb-Edu subset with {budget:,} tokens, seed {self.base_seed}"
                )
                cached_examples = load_fineweb_subset(
                    self.tok,
                    max_tokens=budget,
                    seed=self.base_seed,
                    max_seq_len=self.config.max_seq_len,
                    packing_enabled=bool(getattr(self.config, "enable_packing", True)),
                )
                raw_texts = [ex["prompt"] for ex in cached_examples]
            else:
                print(f"Loading Hugging Face dataset: {self.config.datasets[0]}")
                hf_dataset = (
                    load_dataset(self.config.datasets[0], self.config.dataset_config)[
                        "train"
                    ]
                    if self.config.dataset_config
                    else load_dataset(self.config.datasets[0])["train"]
                )
                prompt_col = self.config.prompt_col or "prompt"
                answer_col = self.config.answer_col
                print(f"Using columns - prompt: '{prompt_col}', answer: '{answer_col}'")
                raw_texts = []
                for ex in hf_dataset:
                    prompt_text = ex[prompt_col]
                    if (
                        answer_col is not None
                        and answer_col in ex
                        and ex[answer_col] is not None
                    ):
                        raw_texts.append(f"{prompt_text}\n{ex[answer_col]}")
                    else:
                        raw_texts.append(prompt_text)

        if getattr(self.config, "enable_packing", True):
            dataset = PackedTokenDataset(raw_texts, self.tok, self.config.max_seq_len)
            if len(dataset) == 0:
                raise RuntimeError(
                    "Packed dataset is empty. Reduce max_seq_len or provide longer input texts."
                )
        else:
            dataset = PromptDataset(raw_texts)
            if len(dataset) == 0:
                raise RuntimeError("Dataset is empty. Provide non-empty input texts.")

        self.packed_dataset = dataset

        self.collate = DistillCollator(self.tok, self.config.max_seq_len)
        self.dl = self._build_training_dataloader(use_distributed_sampler=False)

        self.dataset_size = len(dataset)

    def _plan_teacher_and_cache(self) -> None:
        if bool(getattr(self.config, "skip_by_frozen_student", False)) and not bool(
            getattr(self.config, "offline_cache_selected_only", False)
        ):
            try:
                setattr(self.config, "offline_cache_selected_only", True)
            except Exception:
                object.__setattr__(self.config, "offline_cache_selected_only", True)

        selection_hash = getattr(self.config, "offline_cache_selection_hash", None)
        if selection_hash is None:
            selection_hash = getattr(self.config, "_offline_cache_selection_hash", None)
        selection_pending = (
            bool(getattr(self.config, "offline_cache", False))
            and bool(getattr(self.config, "skip_by_frozen_student", False))
            and bool(getattr(self.config, "offline_cache_selected_only", False))
            and not selection_hash
        )

        if selection_pending:
            teacher_required = True
            teacher_rank0_only = bool(
                teacher_required and getattr(self.config, "ddp_world_size", 1) > 1
            )
            cache_plan = CachePlan(
                signature={},
                cache=None,
                cache_ready=False,
                cache_manifest_items=0,
                expected_items=0,
                cache_dir=None,
                parallel_cache_build=False,
                teacher_required=teacher_required,
                teacher_rank0_only=teacher_rank0_only,
            )
        else:
            try:
                cache_plan = plan_offline_cache(
                    self.config,
                    self.tok,
                    self.dataset_size,
                    is_main_rank=self.is_main_rank,
                    teacherless_modes=self.teacherless_modes,
                )
            except RuntimeError as exc:
                msg = str(exc)
                needs_two_gpus = "Offline cache build requires 2 GPUs" in msg
                if (
                    needs_two_gpus
                    and self.is_main_rank
                    and _auto_torchrun_cache_enabled()
                    and _visible_gpu_count() >= 2
                ):
                    print(
                        "[logits-cache] Missing cache; auto-building with torchrun.",
                        flush=True,
                    )
                    _run_offline_cache_torchrun(self.config)
                    cache_plan = plan_offline_cache(
                        self.config,
                        self.tok,
                        self.dataset_size,
                        is_main_rank=self.is_main_rank,
                        teacherless_modes=self.teacherless_modes,
                    )
                else:
                    raise
        self.cache_plan = cache_plan
        self.cache_ready = cache_plan.cache_ready
        self.teacher_required = cache_plan.teacher_required
        self.teacher_rank0_only = cache_plan.teacher_rank0_only

        self._apply_device_policy()

        # When doing DDP cache build (cache not ready), each rank
        # should load its own teacher to build its partition of the cache in parallel.
        # Only force teacher_rank0_only AFTER cache is complete (for training phase).
        ddp_parallel_cache_build = (
            getattr(self.config, "ddp_world_size", 1) > 1 and not cache_plan.cache_ready
        )
        force_rank0_teacher = (
            self.teacher_required
            and getattr(self.config, "ddp_world_size", 1) > 1
            and not ddp_parallel_cache_build  # Don't force rank0-only during cache build
        )
        if force_rank0_teacher and not self.teacher_rank0_only:
            cache_plan.teacher_rank0_only = True
            self.teacher_rank0_only = True
            if self.is_main_rank:
                print(
                    "[device-planning] Hosting teacher on rank 0 only; other ranks will request logits via broadcast."
                )
        elif ddp_parallel_cache_build and self.teacher_required:
            # During parallel cache build, each rank loads its own teacher
            cache_plan.teacher_rank0_only = False
            self.teacher_rank0_only = False
            if self.is_main_rank:
                print(
                    "[device-planning] DDP parallel cache build: each rank loading its own teacher."
                )
            # Avoid multiple ranks loading the teacher on the same GPU during cache build.
            self.teacher_locals = [self.student_local]
            self.teacher_student_exclusion = None
            print(
                f"[device-planning] Rank {self.ddp_rank} will load teacher on cuda:{self.teacher_locals[0]} for cache build.",
                flush=True,
            )

        if self.teacher_required:
            will_load_teacher = (not self.teacher_rank0_only) or (self.ddp_rank == 0)
            if will_load_teacher and not ddp_parallel_cache_build:
                dedicated_candidates: List[int] = []
                if len(self.local_avail) >= 2:
                    dedicated_candidates = [
                        g for g in self.local_avail if g != self.student_local
                    ]
                if dedicated_candidates:
                    if self.teacher_student_exclusion != self.student_local:
                        self.teacher_student_exclusion = self.student_local
                    if self.teacher_locals != dedicated_candidates:
                        self.teacher_locals = dedicated_candidates
                        should_log_pref = (
                            self.teacher_rank0_only and self.ddp_rank == 0
                        ) or (not self.teacher_rank0_only and self.is_main_rank)
                        if should_log_pref:
                            print(
                                f"[device-planning] Preferring dedicated teacher GPU(s) {self.teacher_locals} with student on {self.student_local}",
                                flush=True,
                            )

        self.teacher = None
        self.teacher_inputs_device = torch.device("cpu")
        if self.teacher_required:
            teacher_bytes = self._estimate_model_weight_bytes(self.config.teacher_model)
            teacher_min_free_gb = self._estimate_role_expected_vram(
                teacher_bytes, "teacher"
            )
            load_here = not self.teacher_rank0_only or is_rank0()
            if load_here:
                if self.is_main_rank:
                    print("Loading teacher with GPU-first fallback...", flush=True)
                self.teacher, _, self.teacher_inputs_device = (
                    load_teacher_with_fallback(
                        model_name=self.config.teacher_model,
                        prefer_gpus=self.teacher_locals,
                        student_gpu=self.teacher_student_exclusion,
                        min_free_gb=teacher_min_free_gb,
                    )
                )
            elif self.is_main_rank:
                print(
                    "Teacher will be hosted on rank 0; skipping local load on this rank."
                )

        # Optionally pre-build offline cache before loading students to free teacher VRAM
        defer_cache_build = bool(
            getattr(self.config, "offline_cache_selected_only", False)
        ) and bool(getattr(self.config, "skip_by_frozen_student", False))
        self.defer_cache_build = defer_cache_build
        if defer_cache_build:
            if self.is_main_rank:
                print(
                    "[logits-cache] Deferring cache build until after sample selection.",
                    flush=True,
                )
            cache_result = CacheBuildResult(
                cache_ready=False,
                cache_manifest_items=int(
                    getattr(self.cache_plan, "cache_manifest_items", 0) or 0
                ),
                teacher_required=self.teacher_required,
                teacher_rank0_only=self.teacher_rank0_only,
                teacher_inputs_device=self.teacher_inputs_device,
                teacher=self.teacher,
                cache=getattr(self.cache_plan, "cache", None),
            )
        else:
            cache_result = execute_cache_plan(
                self.cache_plan,
                config=self.config,
                tok=self.tok,
                packed_dataset=self.packed_dataset,
                collate_fn=self.collate,
                teacher=self.teacher,
                teacher_inputs_device=self.teacher_inputs_device,
                seed_offset=self.seed_offset,
                sanitize_logits_fn=AmpOomMixin._sanitize_logits,
                is_main_rank=self.is_main_rank,
                teacherless_modes=self.teacherless_modes,
            )

        self.teacher = cache_result.teacher
        self.teacher_inputs_device = cache_result.teacher_inputs_device
        self.cache_ready = cache_result.cache_ready
        self.teacher_required = cache_result.teacher_required
        self.teacher_rank0_only = cache_result.teacher_rank0_only
        if force_rank0_teacher:
            self.teacher_rank0_only = True

        teacher_device_str = str(self.teacher_inputs_device)

        if getattr(self.config, "ddp_world_size", 1) > 1:
            if self.teacher_rank0_only:
                obj = [teacher_device_str] if is_rank0() else [None]
                obj = distributed_broadcast_object_list(obj, src=0)
                teacher_device_str = obj[0] or teacher_device_str
            self.teacher_inputs_device = torch.device(teacher_device_str)
            distributed_barrier()
        else:
            self.teacher_inputs_device = torch.device(teacher_device_str)

        self.teacher_device_str = teacher_device_str

        setattr(self.config, "_teacher_device_str", teacher_device_str)
        setattr(self.config, "_teacher_rank0_owner", self.teacher_rank0_only)
        setattr(self.config, "_teacher_required", self.teacher_required)

    def _load_student(self) -> None:
        print("Loading student on its own GPU...", flush=True)
        self.student = load_model(
            self.config.student_model,
            device_map=self.student_local,
            quant_bits=self.config.student_quant_bits,
        )
        first_param = next(iter(self.student.parameters()), None)
        if first_param is not None:
            self.student_dtype = first_param.dtype
        else:
            first_buffer = next(iter(self.student.buffers()), None)
            if first_buffer is not None:
                self.student_dtype = first_buffer.dtype
            else:
                self.student_dtype = torch.float32

        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad_(False)
            self.teacher.config.use_cache = False

        self.student.train()
        self.student.config.use_cache = False

        if self.teacher is not None or self.teacher_required:
            print(f"Teacher device: {self.teacher_inputs_device}")
        else:
            if self.is_main_rank:
                print(
                    "Teacher load skipped; using offline cache exclusively for teacher signals."
                )
        print(f"Student device: {self.student_device}")

        if self.student_device.type == "cuda":
            print(
                f"Using GPU: {torch.cuda.get_device_name(self.student_device.index)} (device {self.student_device.index})"
            )
            print(
                f"GPU memory allocated: {torch.cuda.memory_allocated(self.student_device) / 1024**3:.2f} GB"
            )
            print(
                f"GPU memory reserved: {torch.cuda.memory_reserved(self.student_device) / 1024**3:.2f} GB"
            )
        else:
            print(f"Using device: {self.student_device}")

        if self.is_main_rank:

            def _param_count(model):
                if model is None:
                    return 0
                counter = getattr(model, "num_parameters", None)
                if callable(counter):
                    try:
                        return int(counter())
                    except Exception:
                        pass
                return sum(int(p.numel()) for p in model.parameters())

            teacher_params = _param_count(self.teacher)
            student_params = _param_count(self.student)
            print(
                f"Parameter counts → teacher: {teacher_params:,} | student: {student_params:,}",
                flush=True,
            )

    def _initialize_logger(self) -> None:
        # Initialize combined logger (W&B + TensorBoard)
        self.combined_logger = None
        if self.is_main_rank:
            self.combined_logger = create_training_combined_logger(
                self.config,
                self.experiment_name,
                tensorboard_dir=self.config.tensorboard_dir,
                display_name=self.display_name,
            )

    def _build_distiller(self) -> Distiller:
        if self.config.distill_category == "on_policy":
            from sekd.distill import OnPolicyDistiller

            return OnPolicyDistiller(
                teacher_model=self.teacher,
                student_model=self.student,
                tokenizer=self.tok,
                dataloader=self.dl,
                config=self.config,
                teacher_device=self.teacher_inputs_device,
                student_device=self.student_device,
                logger=self.combined_logger,
            )
        return Distiller(
            teacher_model=self.teacher,
            student_model=self.student,
            tokenizer=self.tok,
            dataloader=self.dl,
            config=self.config,
            teacher_device=self.teacher_inputs_device,
            student_device=self.student_device,
            student_dtype=self.student_dtype,
            logger=self.combined_logger,
        )

    def _train_distiller(self) -> None:
        if self.distiller is None:
            raise RuntimeError("Distiller has not been initialized.")
        self.distiller.train(epochs=self.config.epochs)

    def _post_training(self) -> None:
        if getattr(self.config, "ddp_world_size", 1) > 1:
            distributed_barrier()

        if self.is_main_rank:
            if self.combined_logger:
                try:
                    self.combined_logger.log_artifact(
                        self.config.output_dir,
                        f"student_model_{self.experiment_name}",
                        "model",
                    )
                    self.combined_logger.finish()
                except Exception:
                    pass

            model_to_save = getattr(self.distiller, "student", self.student)
            if hasattr(model_to_save, "module"):
                model_to_save = model_to_save.module
            print("Saving student to", self.config.output_dir)
            model_to_save.save_pretrained(self.config.output_dir)
            self.tok.save_pretrained(self.config.output_dir)

            for reg_path in self.registry_paths_to_update:
                try:
                    mark_trained(
                        reg_path,
                        self.params_hash,
                        model_output_dir=self.config.output_dir,
                    )
                except Exception as e:
                    print(f"[registry] Failed to mark trained at {reg_path}: {e}")

        if getattr(self.config, "ddp_world_size", 1) > 1:
            distributed_barrier()

    def _finalize(self) -> None:
        if getattr(self.config, "ddp_world_size", 1) > 1:
            destroy_distributed()


def main():
    config = parse_args_to_config()
    runner = DistillationEntrypoint(config)
    runner.run()


if __name__ == "__main__":
    main()
