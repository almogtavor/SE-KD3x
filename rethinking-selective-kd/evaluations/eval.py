#!/usr/bin/env python3
import argparse
import math
import json
import os
import random
import re
import shutil
import subprocess
import sys
import hashlib
import time
import copy
import yaml
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, Set
from sampledkd.run_registry import compute_params_hash, upsert_eval_results
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ConfigDict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    GenerationConfig,
)
import importlib
from importlib import import_module
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

load_dotenv()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_INFERENCE_MODE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
if "HF_DATASETS_CACHE" not in os.environ:
    _tmp = os.environ.get("TMPDIR", "/tmp")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(_tmp, "hf_datasets")

def _safe_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except Exception:
        return default

IF_EVAL_SAMPLE_LIMIT = _safe_int_env("IF_EVAL_SAMPLE_LIMIT", 0)
IF_EVAL_MAX_NEW_TOKENS = _safe_int_env("IF_EVAL_MAX_NEW_TOKENS", 512)
LM_VALIDATION_SAMPLE_SIZE = _safe_int_env("LM_VALIDATION_SAMPLE_SIZE", 200)

# Fallback tokenizer when model directory is incomplete (all our models use the same tokenizer)
FALLBACK_TOKENIZER = "Qwen/Qwen3-1.7B"


def validate_model_directory(model_dir: Path) -> None:
    """Validate that a model directory contains the files required for evaluation.

    Raises a descriptive error if essential files (config.json, model weights,
    tokenizer) are missing, which typically indicates that training has not
    finished yet.
    """
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Model directory is incomplete (missing config.json): {model_dir}\n"
            "This usually means training has not finished yet. "
            "Check if training is still running or if it crashed before saving the final model."
        )

    # Check for model weights - either safetensors or pytorch format
    has_safetensors = any(model_dir.glob("*.safetensors"))
    has_pytorch = (model_dir / "pytorch_model.bin").exists() or any(
        model_dir.glob("pytorch_model-*.bin")
    )
    has_weights = has_safetensors or has_pytorch
    if not has_weights:
        raise FileNotFoundError(
            f"Model directory is incomplete (missing model weights): {model_dir}\n"
            "Expected to find *.safetensors or pytorch_model*.bin files. "
            "This usually means training has not finished yet."
        )

    # Check for tokenizer files (at least one of the common tokenizer files)
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "spiece.model",
        "tokenizer.model",
    ]
    has_tokenizer = any((model_dir / f).exists() for f in tokenizer_files)
    if not has_tokenizer:
        print(
            f"[warn] Model directory is missing tokenizer files: {model_dir}\n"
            f"Will use fallback tokenizer: {FALLBACK_TOKENIZER}"
        )


def load_tokenizer_with_fallback(model_path: str | Path) -> "AutoTokenizer":
    """Load tokenizer from model path, falling back to FALLBACK_TOKENIZER if not found.
    
    All our distilled models use the same Qwen tokenizer, so we can safely
    use a fallback when the model directory is incomplete.
    """
    model_path = Path(model_path) if isinstance(model_path, str) else model_path
    
    # Check if tokenizer files exist
    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "spiece.model", "tokenizer.model"]
    has_tokenizer = model_path.is_dir() and any((model_path / f).exists() for f in tokenizer_files)
    
    if has_tokenizer:
        try:
            return AutoTokenizer.from_pretrained(str(model_path), use_fast=False, trust_remote_code=True)
        except Exception as e:
            print(f"[warn] Failed to load tokenizer from {model_path}: {e}")
            print(f"[warn] Falling back to {FALLBACK_TOKENIZER}")
    else:
        print(f"[info] No tokenizer files in {model_path}, using fallback: {FALLBACK_TOKENIZER}")
    
    return AutoTokenizer.from_pretrained(FALLBACK_TOKENIZER, use_fast=False, trust_remote_code=True)


# ---------- Logging imports ----------
try:
    from sampledkd.logging.wandb_utils import (
        WandBLogger,
        log_evaluation_to_wandb,
        log_evaluation_to_tensorboard,
    )
except ImportError:
    WandBLogger = None  # type: ignore
    log_evaluation_to_wandb = log_evaluation_to_tensorboard = lambda *args, **kwargs: None

# ---------- Pydantic Models for Configuration ----------
class EvalGenerationConfig(BaseModel):
    max_new_tokens: int = 256
    do_sample: bool = False
    temperature: float = 0.0

class MetricConfig(BaseModel):
    metric: str
    aggregation: str = "mean"
    higher_is_better: bool = True

class TaskConfig(BaseModel):
    task: str  # lm-eval expects "task" field
    dataset_path: str
    dataset_name: Optional[str] = None
    test_split: Optional[str] = None
    validation_split: Optional[str] = None
    train_split: Optional[str] = None
    output_type: str = "generate_until"
    generation_kwargs: EvalGenerationConfig = Field(default_factory=EvalGenerationConfig)
    metric_list: List[MetricConfig] = Field(default_factory=list)
    process_docs: Optional[str] = None
    process_results: Optional[str] = None
    num_fewshot: int = 0

class BenchmarkConfig(BaseModel):
    task: List[TaskConfig] = Field(alias="manual_tasks")
    model_config = ConfigDict(populate_by_name=True)

class _BenchmarkSafeLoader(yaml.SafeLoader):
    """YAML loader that gracefully handles custom tags like !function."""

def _construct_unknown_tag(loader: _BenchmarkSafeLoader, tag_suffix: str, node: yaml.Node):
    if isinstance(node, yaml.ScalarNode):
        return loader.construct_scalar(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    raise TypeError(f"Unsupported YAML node type for custom tag: {type(node)}")

_BenchmarkSafeLoader.add_multi_constructor("!", _construct_unknown_tag)

def load_benchmark_config(config_path: Path) -> BenchmarkConfig:
    """Load and validate benchmark configuration from YAML."""
    with open(config_path, "r") as f:
        data = yaml.load(f, Loader=_BenchmarkSafeLoader)
    return BenchmarkConfig(**data)


def _bool_to_yaml(value: bool) -> str:
    return "true" if value else "false"

def _normalize_func_ref(ref: str) -> str:
    # Accept "path/to/file.py:fn" or "pkg.mod:fn". Convert the latter to a file path.
    if ":" not in ref:
        return ref
    head, func = ref.split(":", 1)
    if head.endswith(".py") or "/" in head:
        return f"{head}:{func}"
    return f"{head.replace('.', '/')}.py:{func}"

def _render_manual_task(task: TaskConfig) -> str:
    lines = [f"task: {task.task}", f"dataset_path: {task.dataset_path}"]
    if task.dataset_name:
        lines.append(f"dataset_name: {task.dataset_name}")
    if task.test_split:
        lines.append(f"test_split: {task.test_split}")
    if task.validation_split:
        lines.append(f"validation_split: {task.validation_split}")
    if task.train_split:
        lines.append(f"train_split: {task.train_split}")
    g = task.generation_kwargs
    lines.append(f"output_type: {task.output_type}")
    lines.append('doc_to_text: "{{input}}"')
    lines.append('doc_to_target: "{{answer}}"')
    lines.append("generation_kwargs:")
    lines.append(f"  max_new_tokens: {g.max_new_tokens}")
    lines.append(f"  do_sample: {_bool_to_yaml(g.do_sample)}")
    # Only include sampling-related fields if sampling is enabled to avoid
    # Transformers warnings about ignored flags (temperature/top_p/top_k) when do_sample=false.
    if g.do_sample:
        lines.append(f"  temperature: {g.temperature}")
    lines.append("  until:")
    lines.append('    - "\\\\n\\\\n"')
    lines.append('    - "</s>"')
    lines.append('    - "<|im_end|>"')
    if task.process_docs:
        lines.append(f"process_docs: !function {_normalize_func_ref(task.process_docs)}")
    if task.metric_list:
        lines.append("metric_list:")
        for metric in task.metric_list:
            lines.extend([
                f"  - metric: {metric.metric}",
                f"    aggregation: {metric.aggregation}",
                f"    higher_is_better: {_bool_to_yaml(metric.higher_is_better)}",
            ])
    if task.process_results:
        lines.append(f"process_results: !function {_normalize_func_ref(task.process_results)}")
    lines.extend([f"num_fewshot: {task.num_fewshot}", ""])
    return "\n".join(lines)


def _normalize_dtype_name(value: Optional[str]) -> Optional[str]:
    """Map user/config dtype strings onto canonical HF dtype names."""
    if not value:
        return None
    normalized = value.strip().lower()
    normalized = normalized.replace("torch.", "")
    alias_map = {
        "float16": "float16",
        "fp16": "float16",
        "half": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    return alias_map.get(normalized)


def _infer_model_dtype_from_config(model_dir: Union[str, Path]) -> Optional[str]:
    """Inspect config.json to recover the dtype used during training/export."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[models] Warning: unable to parse {config_path}: {e}")
        return None
    for key in ("torch_dtype", "dtype"):
        raw_val = cfg.get(key)
        if isinstance(raw_val, str):
            normalized = _normalize_dtype_name(raw_val)
            if normalized:
                return normalized
    return None


def materialize_manual_tasks(config: BenchmarkConfig, cache_root: Path) -> Path:
    """Create per-task YAML files lm-eval can consume from the benchmark config."""
    manual_dir = cache_root / "_manual_tasks"
    ensure_dir(manual_dir)
    for existing in manual_dir.glob("*.yaml"):
        existing.unlink()
    for task in config.task:
        (manual_dir / f"{task.task}.yaml").write_text(_render_manual_task(task))
    src_utils = Path(__file__).with_name("utils.py")
    if src_utils.exists():
        # Copy as a flattened module name for lm-eval to find
        shutil.copy2(src_utils, manual_dir / "sampledkd.evaluations.utils.py")
    return manual_dir


# ---------- Utility ----------
def run(cmd: List[str], env: Optional[Dict[str, str]] = None, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Tuple[int, str]:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] $ {' '.join(cmd)}")
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, env=env, cwd=cwd, text=True, timeout=timeout)
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Command completed:")
        print(out)
        return 0, out
    except subprocess.CalledProcessError as e:
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Command failed:")
        print(e.output)
        return e.returncode, e.output
    except subprocess.TimeoutExpired as e:
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Command timed out after {timeout} seconds: {e}")
        return 124, f"Timeout after {timeout} seconds"

def run_async(cmd: List[str], env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] $ (async) {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)

def wait_with_timeout(proc: subprocess.Popen, timeout: int) -> Tuple[int, str]:
    try:
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Waiting for process (timeout={timeout}s)...")
        out, _ = proc.communicate(timeout=timeout)
        rc = proc.returncode
        if rc is None:
            rc = 0
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Process completed with code {rc}")
        print(out or "")
        return rc, out or ""
    except subprocess.TimeoutExpired:
        end_timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{end_timestamp}] Process timed out after {timeout}s, killing...")
        proc.kill()
        out, _ = proc.communicate()
        print(out or "")
        return 124, out or ""

def find_latest_checkpoint(dir_path: Path) -> Optional[Path]:
    if not dir_path.exists():
        return None
    candidates = sorted(dir_path.glob("checkpoint_epoch*_step*.pt"))
    if not candidates:
        return None
    def key(p: Path):
        m = re.search(r"epoch(\d+)_step(\d+)", p.name)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        return (-1, -1)
    candidates.sort(key=key)
    return candidates[-1]

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def export_hf_model(base_model_dir: str, ckpt_path: Path, export_dir: Path) -> None:
    """Export a HF-ready directory from base weights + checkpoint.
       Skips work if kd_export_meta.json matches the checkpoint hash.
    """
    export_dir.mkdir(parents=True, exist_ok=True)
    meta_path = export_dir / "kd_export_meta.json"
    ckpt_hash = sha256_file(ckpt_path)

    # Skip if unchanged
    if meta_path.exists():
        try:
            old = json.load(open(meta_path))
            if old.get("ckpt_sha256") == ckpt_hash:
                print(f"[export] Cache hit for {export_dir} (ckpt unchanged). Skipping export.")
                return
        except Exception:
            pass

    print(f"Exporting model from base '{base_model_dir}' with state_dict '{ckpt_path.name}' -> '{export_dir}'")

    # Always load tokenizer (with fallback to Qwen if not found in model dir)
    tok = load_tokenizer_with_fallback(base_model_dir)

    # ---- Try 8-bit first, then fallback ----
    model = None
    try:
        print("[export] Trying 8-bit quantized load...", flush=True)
        q8 = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            quantization_config=q8,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        for p in model.parameters():
            p.requires_grad_(False)
        print("[export] Loaded model in 8-bit.", flush=True)
    except Exception as e:
        print(f"[export] 8-bit load failed ({e}); falling back to float16.", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            base_model_dir,
            torch_dtype=torch.float16,
            device_map=None,
            trust_remote_code=True,
        )

    # ---- Load checkpoint weights ----
    chk = torch.load(ckpt_path, map_location="cpu")
    state = chk.get("model_state_dict")
    if state is None:
        raise ValueError(f"No 'model_state_dict' in {ckpt_path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

    # ---- Save final HF export ----
    model.save_pretrained(export_dir, safe_serialization=True)
    tok.save_pretrained(export_dir)

    # Copy run_params.json if present
    try:
        candidate = ckpt_path.parent.parent / "run_params.json"
        if candidate.exists():
            shutil.copy2(candidate, export_dir / "run_params.json")
    except Exception:
        pass

    meta = {
        "source_checkpoint": str(ckpt_path),
        "epoch": chk.get("epoch"),
        "step": chk.get("step"),
        "global_step": chk.get("global_step"),
        "distill_type": chk.get("distill_type"),
        "k_percent": chk.get("k_percent"),
        "export_time_utc": datetime.utcnow().isoformat() + "Z",
        "ckpt_sha256": ckpt_hash,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[export] Wrote {meta_path}")

def ensure_dir(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)

def which(bin_name: str) -> bool:
    return shutil.which(bin_name) is not None


def _ensure_allocator_safe_env(env: Dict[str, str]) -> None:
    """Force CUDA allocator to avoid expandable segments which trip PyTorch bugs."""
    key = "PYTORCH_CUDA_ALLOC_CONF"
    current = env.get(key, "")
    if not current:
        env[key] = "expandable_segments:False"
        return
    parts = [p.strip() for p in current.split(",") if p.strip()]
    updated: List[str] = []
    seen_flag = False
    for part in parts:
        if part.startswith("expandable_segments"):
            seen_flag = True
            if part != "expandable_segments:False":
                updated.append("expandable_segments:False")
            else:
                updated.append(part)
        else:
            updated.append(part)
    if not seen_flag:
        updated.append("expandable_segments:False")
    env[key] = ",".join(updated)

# ---------- GPU helpers ----------
def visible_gpu_ids() -> List[int]:
    """Return the list of visible GPU ids from CUDA_VISIBLE_DEVICES or torch."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        try:
            return [int(x) for x in cvd.split(",") if x != ""]
        except Exception:
            pass
    if torch.cuda.is_available():
        return list(range(torch.cuda.device_count()))
    return []

def pick_gpu_pool(max_workers: Optional[int] = None) -> List[int]:
    ids = visible_gpu_ids()
    if max_workers is not None:
        ids = ids[:max_workers]
    if not ids:
        print("No CUDA devices visible, defaulting to CPU (will be slow).")
    return ids

# Map HF model families -> default decoder layer class for FSDP auto-wrapping
FSDP_LAYER_CLS_BY_MODEL_FAMILY: Dict[str, str] = {
    "llama": "LlamaDecoderLayer",
    "mistral": "MistralDecoderLayer",
    "mixtral": "MixtralDecoderLayer",
    "qwen2": "Qwen2DecoderLayer",
    "qwen3": "Qwen3DecoderLayer",
    "gemma": "GemmaDecoderLayer",
    "yi": "YiDecoderLayer",
    "phi": "PhiDecoderLayer",
    "deepseek": "DeepseekDecoderLayer",
    "glm": "GLMBlock",
    "baichuan": "DecoderLayer",
}


def _normalize_family(model_type: str) -> str:
    """Collapse model_type variants (e.g., qwen2_moe, llama3) into a family key."""
    match = re.match(r"([a-zA-Z]+)(\d+(?:\.\d+)*)?", model_type)
    base = match.group(1).lower() if match else model_type.lower()
    return re.split(r"[_\-.]", base)[0]


def _try_dynamic_discovery(model_type: str, cfg: Optional[Any]) -> Optional[str]:
    """Inspect the modeling module for plausible decoder layer classes."""
    import torch.nn as nn

    normalized_type = model_type.lower()
    families = [_normalize_family(normalized_type)]
    if normalized_type not in families:
        families.insert(0, normalized_type)
    candidate_modules = []
    for key in families:
        lib = CONFIG_MAPPING_NAMES.get(key) or key
        candidate_modules.append(f"transformers.models.{lib}.modeling_{lib}")
    if cfg is not None:
        cfg_module = getattr(cfg.__class__, "__module__", "")
        if cfg_module:
            base_mod = cfg_module.rsplit(".", 1)[0]
            suffix = base_mod.split(".")[-1]
            candidate_modules.extend(
                [
                    f"{base_mod}.modeling_{suffix}",
                    f"{base_mod}.modeling",
                ]
            )
    seen: Set[str] = set()
    for module_name in candidate_modules:
        if module_name in seen:
            continue
        seen.add(module_name)
        try:
            module = import_module(module_name)
        except Exception:
            continue
        candidates = []
        for attr in dir(module):
            if not (attr.endswith("DecoderLayer") or attr.endswith("Block")):
                continue
            obj = getattr(module, attr)
            try:
                if isinstance(obj, type) and issubclass(obj, nn.Module):
                    candidates.append(attr)
            except Exception:
                continue
        if not candidates:
            continue
        family = _normalize_family(model_type)
        family_upper = family.upper()
        preferred = [
            name
            for name in candidates
            if family.capitalize() in name or family.title() in name or family_upper in name
        ]
        return preferred[0] if preferred else candidates[0]
    return None


def _try_meta_sniff(cfg: Any) -> Optional[str]:
    """Instantiate the model without weights and peek at the first decoder block."""
    try:
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    except Exception:
        return None
    candidate_paths = [
        "model.layers",
        "model.decoder.layers",
        "transformer.h",
    ]
    for chain in candidate_paths:
        obj = model
        valid = True
        for part in chain.split("."):
            if not hasattr(obj, part):
                valid = False
                break
            obj = getattr(obj, part)
        if not valid:
            continue
        layers = None
        if isinstance(obj, (list, tuple)) and obj:
            layers = obj
        else:
            try:
                if len(obj) > 0:
                    layers = obj
            except Exception:
                layers = None
        if layers:
            first = layers[0]
            return first.__class__.__name__
    return None


def detect_fsdp_layer_cls(model_identifier: Union[str, Path]) -> Optional[str]:
    """Best-effort detection of the transformer block class to wrap when enabling FSDP."""
    target = str(model_identifier)
    try:
        cfg = AutoConfig.from_pretrained(target, trust_remote_code=True)
    except Exception as exc:
        print(f"[fsdp] Warning: failed to load config for '{target}': {exc}")
        return None

    model_type = getattr(cfg, "model_type", None)
    if not model_type:
        print(f"[fsdp] Warning: config for '{target}' does not expose model_type.")
        return None

    model_type_lc = str(model_type).lower()
    family = _normalize_family(model_type_lc)
    mapped = FSDP_LAYER_CLS_BY_MODEL_FAMILY.get(family)
    if mapped:
        print(f"[fsdp] Using mapped transformer layer '{mapped}' for model_type='{model_type}'.")
        return mapped

    dynamic = _try_dynamic_discovery(model_type_lc, cfg)
    if dynamic:
        print(f"[fsdp] Auto-discovered transformer layer '{dynamic}' for model_type='{model_type}'.")
        return dynamic

    meta_cls = _try_meta_sniff(cfg)
    if meta_cls:
        print(f"[fsdp] Meta-sniffed transformer layer '{meta_cls}' for model_type='{model_type}'.")
        return meta_cls

    print(f"[fsdp] Warning: could not identify a decoder layer class for model_type='{model_type}'.")
    return None

# ==========================================================
#                      SUITE DEFINITIONS
# ==========================================================

# NOTE: Task names follow lm-eval harness conventions. Some tasks may be unavailable
# in older harness versions; failures are tolerated and reported.
# LIGHT suite (coffee-break): each tuple is (task, sample_count). We attempt to
# draw a SEEDED RANDOM subset of that many samples per task (when the lm-eval CLI
# supports random limiting). If the installed lm-eval version lacks such a flag,
# we fall back to the harness default (typically first-N) and print a warning.
_BASE_LIGHT_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = [
    # accuracy (percentage of correct answers)
    ("arc_easy", None),
    ("gsm8k", None),
    ("hellaswag", None),
    ("piqa", None),
    # ("svamp", 250),
    ("lambada_openai", None), 
    # normalized accuracy - multiple-choice datasets.raw accuracy can mislead so normalization accounts for imbalanced choices
    # ("arc_challenge", None),
    # exact-match
    # ("aime25", None),
    # ("ifeval", None)
]
LIGHT_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = list(_BASE_LIGHT_LMEVAL_TASKS)

IF_TASK_SPECS = [
    {
        "name": "ifeval_deterministic",
        "display_name": "IFEval",
        "metric_key": "strict_accuracy",
    }
]
IF_TABLE_TASKS: List[Tuple[str, Optional[int]]] = [(spec["name"], None) for spec in IF_TASK_SPECS]
IF_TASK_NAMES = {spec["name"] for spec in IF_TASK_SPECS}
# Optional tiny adds (off by default): BoolQ 200, HumanEval full
LIGHT_ENABLE_OPTIONALS = os.environ.get("LIGHT_EXTRAS", "0") == "1"
LIGHT_OPTIONALS: List[Tuple[str, Optional[int]]] = [
    ("boolq", 200),
]

# HEAVY suite (paper): broad coverage; full sets (no --limit) unless noted
HEAVY_LMEVAL_TASKS: List[Tuple[str, Optional[int]]] = [
    # Reasoning & math
    ("gsm8k", None),
    ("svamp", None),
    ("asdiv", None), # arithmetic subset (ASDiv-A handled inside task)
    ("hendrycks_math", None),  # MATH
    ("aime25", None),
    ("olympiadbench", None),
    ("piqa", None),
    ("lambada_openai", None),
     # General reasoning
    ("agieval", None),
    ("bbh", None), # BIG-Bench Hard
    ("agieval", None),
    # QA / multi-hop
    ("squadv2", None),
    ("hotpotqa", None),
    ("nq_open", None),
     # Commonsense
    ("hellaswag", None),
    ("arc_easy", None),
    ("arc_challenge", None),
]

# Per-task timeouts (seconds).
TASK_TIMEOUTS = {
    # light-ish baselines
    "gsm8k": 30000,
    "hellaswag": 30000,
    "piqa": 30000,
    "lambada_openai": 30000,
    "arc_challenge": 6000,
    "arc_easy": 6000,
    "svamp": 6000,
    "aime25": 6000,
    "boolq": 600,
    # heavy add-ons
    "asdiv": 1200,
    "hendrycks_math": 3600,
    "olympiadbench": 5400,
    "bbh": 5400,
    "agieval": 5400,
    "squadv2": 1200,
    "hotpotqa": 3600,
    "nq_open": 3600,
    # "ifeval": 10
}

# Summarization (Lighteval)
LIGHTEVAL_TASKS = ["helm|summarization:cnn_dailymail", "helm|summarization:xsum"]

# ==========================================================
#     Split overrides for lm-eval tasks
# ==========================================================
# NOTE: LAMBADA OpenAI and IFEval are excluded from validation suite
# and run in full on test suite only.

# Tasks to exclude from validation suite (no native validation split)
VALIDATION_EXCLUDED_TASKS = {"lambada_openai", "ifeval", "ifeval_deterministic"}

# Split overrides for lm-eval tasks when forcing validation/test passes
LM_EVAL_VALIDATION_SPLITS: Dict[str, str] = {
    "hellaswag": "validation",
    "piqa": "validation",
    "gsm8k": "train",  # dataset lacks a dedicated validation split
    "arc_easy": "validation",
}
LM_EVAL_TEST_SPLITS: Dict[str, str] = {
    "hellaswag": "test",
    "piqa": "test",
    "gsm8k": "test",
    "lambada_openai": "test",
    "arc_easy": "test",
}

# ==========================================================
#                  LM-Eval runner with suites
# ==========================================================
def run_lmeval_suite(
    model_dir: Path,
    tag: str,
    results_dir: Path,
    gpu_ids: List[int],
    suite: str,
    batch_size: str = "auto",
    generate_batch_size: str = "1",
    max_batch_size: int = 0,
    generate_max_batch_size: int = 4,
    fallback_batch_size: int = 1,
    model_dtype: Optional[str] = None,
    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    device_map: Optional[str] = None,
    share_gpu_pool: bool = False,
    use_fsdp: bool = False,
    fsdp_layer_cls: Optional[str] = None,
    fsdp_policy: str = "full_shard auto_wrap",
    phase: str = "default",
    task_split_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Path], Dict[str, str], Dict[str, float]]:
    """Run the requested LM-Eval suite (light/heavy).

    Executes tasks across GPUs in waves; when share_gpu_pool is True, a single task uses the full pool.
    """
    out_dir = results_dir / f"lmeval_{suite}_{phase}_{tag}"
    ensure_dir(out_dir)

    tasks_with_limits: List[Tuple[str, Optional[int]]] = list(
        LIGHT_LMEVAL_TASKS if suite == "light" else HEAVY_LMEVAL_TASKS
    )
    if suite == "light" and LIGHT_ENABLE_OPTIONALS:
        tasks_with_limits += list(LIGHT_OPTIONALS)
    tasks_with_limits = [
        (name, limit)
        for name, limit in tasks_with_limits
        if name.split(",", 1)[0] not in IF_TASK_NAMES
    ]

    if not tasks_with_limits:
        print("No LM-Eval tasks selected.")
        return None

    # Validate benchmark config if it exists
    include_root = Path(__file__).parent
    manual_include_dir: Optional[Path] = None
    benchmark_config_path = include_root / "benchmark_tasks.yaml"
    if benchmark_config_path.exists():
        try:
            config = load_benchmark_config(benchmark_config_path)
            manual_task_names = [task.task for task in config.task]
            manual_include_dir = materialize_manual_tasks(config, results_dir)
            print(
                f"✅ Validated benchmark config with {len(manual_task_names)} custom tasks: {manual_task_names}\n"
                f"➡️  Materialized manual tasks at {manual_include_dir}"
            )
        except Exception as e:
            print(f"⚠️ Warning: Invalid benchmark config: {e}")

    include_path = (manual_include_dir or include_root).as_posix()
    repo_root = Path(__file__).resolve().parents[2].as_posix()

    # Exclude LAMBADA and IFEval from validation suite (they run in full on test only)
    if phase == "validation":
        excluded = [name for name, _ in tasks_with_limits if name.split(",", 1)[0] in VALIDATION_EXCLUDED_TASKS]
        if excluded:
            print(f"[lm-eval] Excluding from validation suite (test-only): {excluded}")
        tasks_with_limits = [
            (name, limit)
            for name, limit in tasks_with_limits
            if name.split(",", 1)[0] not in VALIDATION_EXCLUDED_TASKS
        ]

    if not gpu_ids:
        print("[lm-eval] No GPUs detected/selected; skipping suite.")
        return None

    if use_fsdp and (load_in_8bit or load_in_4bit):
        print("[fsdp] Warning: --use-fsdp cannot be combined with bitsandbytes quantization flags; disabling FSDP for this run.")
        use_fsdp = False

    if use_fsdp and len(gpu_ids) < 2:
        print("[fsdp] Warning: fewer than 2 GPUs available; disabling FSDP request.")
        use_fsdp = False

    if use_fsdp and not share_gpu_pool:
        print("[fsdp] Enabling GPU pool sharing for FSDP.")
        share_gpu_pool = True

    if share_gpu_pool and len(gpu_ids) > 1:
        print(f"[lm-eval] Treating GPUs {gpu_ids} as a single tensor-parallel pool.")

    # preflight: load task registry to filter unknown tasks (but allow external include_path)
    try:
        from lm_eval import tasks as _tasks_mod  # type: ignore
        _available = set(getattr(_tasks_mod, "TASK_REGISTRY", {}).keys())
    except Exception:
        _available = set()
    sampling_overrides: Dict[str, int] = {}
    if phase == "validation" and task_split_overrides:
        for base_task, split_name in task_split_overrides.items():
            if split_name in {"train", "test"}:
                sampling_overrides[base_task] = LM_VALIDATION_SAMPLE_SIZE
    if _available:
        filtered_tasks: List[Tuple[str, Optional[int]]] = []
        skipped_tasks: List[str] = []
        for task_name, limit in tasks_with_limits:
            base_task = task_name.split(",", 1)[0]
            if base_task in IF_TASK_NAMES:
                skipped_tasks.append(task_name)
                continue
            if base_task in _available:
                new_limit = sampling_overrides.get(base_task, limit)
                filtered_tasks.append((task_name, new_limit))
            else:
                skipped_tasks.append(task_name)
        if skipped_tasks:
            print(f"[lm-eval] Skipping non-registry tasks (handled separately): {skipped_tasks}")
        tasks_with_limits = filtered_tasks
    else:
        tmp_tasks: List[Tuple[str, Optional[int]]] = []
        for task_name, limit in tasks_with_limits:
            base_task = task_name.split(",", 1)[0]
            if base_task in IF_TASK_NAMES:
                continue
            tmp_tasks.append((task_name, sampling_overrides.get(base_task, limit)))
        tasks_with_limits = tmp_tasks

    # Include local task YAMLs if present
    # Seed for downstream tools; default to env or EVAL_SEED set by main
    seed = int(os.environ.get("EVAL_SEED", "42"))
    # Probe lm-eval CLI support for seed and random limiting (version-dependent)
    _help_code, _help_out = run(["lm-eval", "--help"])
    _has_seed_flag = _help_code == 0 and ("--seed" in _help_out)
    _has_fewshot_seed_flag = _help_code == 0 and ("--fewshot-seed" in _help_out)
    _lm_eval_supports_seed = _has_seed_flag or _has_fewshot_seed_flag
    # Random limiting flags have changed across versions; detect both variants
    _has_limit_type = _help_code == 0 and ("--limit-type" in _help_out)
    _has_limit_mode = _help_code == 0 and ("--limit_mode" in _help_out)
    # Per-sample logging (needed for ECE)
    _has_log_samples = _help_code == 0 and ("--log_samples" in _help_out)

    def _detect_flag(help_blob: str, *candidates: str) -> Optional[str]:
        if help_blob is None:
            return None
        for cand in candidates:
            if cand in help_blob:
                return cand
        return None

    _max_batch_flag = _detect_flag(_help_out, "--max_batch_size", "--max-batch-size")
    _generate_max_batch_flag = _detect_flag(
        _help_out,
        "--generate_max_batch_size",
        "--generate-max-batch-size",
    )
    _task_args_flag = _detect_flag(_help_out, "--task_args", "--task-args")

    default_batch_size = str(batch_size)
    default_generate_batch_size = str(generate_batch_size)
    effective_max_batch_size = max(0, int(max_batch_size))
    effective_generate_max_batch_size = max(0, int(generate_max_batch_size))
    effective_fallback_batch_size = max(0, int(fallback_batch_size))

    resolved_model_dtype: Optional[str] = None
    dtype_request = model_dtype.strip() if isinstance(model_dtype, str) else ""
    auto_requested = dtype_request == "" or dtype_request.lower() in {"auto", "detect", "default", "none"}
    if auto_requested:
        resolved_model_dtype = _infer_model_dtype_from_config(model_dir)
        if resolved_model_dtype:
            print(f"[models] Auto-detected dtype '{resolved_model_dtype}' from {model_dir}/config.json")
    else:
        resolved_model_dtype = _normalize_dtype_name(dtype_request)
        if resolved_model_dtype is None and dtype_request:
            print(f"[models] Warning: unknown dtype '{model_dtype}'; letting HF defaults apply.")
    model_dtype = resolved_model_dtype

    model_arg_parts = [
        f"pretrained={model_dir}",
        "trust_remote_code=True",
    ]
    if load_in_4bit:
        model_arg_parts.append("load_in_4bit=True")
    elif load_in_8bit:
        model_arg_parts.append("load_in_8bit=True")
    else:
        if model_dtype:
            model_arg_parts.append(f"dtype={model_dtype}")
    effective_device_map = None if use_fsdp else device_map
    if not effective_device_map and not use_fsdp and share_gpu_pool and len(gpu_ids) > 1:
        effective_device_map = "balanced_low_0"
    if effective_device_map:
        model_arg_parts.append(f"device_map={effective_device_map}")

    entry_prefix = ["lm-eval"]
    if use_fsdp:
        entry_prefix = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={len(gpu_ids)}",
            "--module",
            "sampledkd.evaluations.lm_eval_entry",
        ]

    base_args = [
        "--model",
        "hf",
        "--model_args",
        ",".join(model_arg_parts),
        "--output_path",
        str(out_dir),
        "--include_path",
        include_path,
    ]
    # Enable per-sample logging if supported (helps compute ECE later)
    if _has_log_samples:
        base_args += ["--log_samples"]
    else:
        print("[lm-eval] This lm-eval version lacks --log_samples; will skip ECE metrics.")
    if _has_seed_flag:
        base_args += ["--seed", str(seed)]
    elif _has_fewshot_seed_flag:
        base_args += ["--fewshot-seed", str(seed)]

    _warned_no_random_limit = False
    _warned_no_max_batch_flag = False
    _warned_no_generate_max_batch_flag = False
    _warned_no_task_args_flag = False

    def _task_cmd(
        task: str,
        limit: Optional[int],
        device: str,
        override_batch: Optional[str] = None,
    ) -> Tuple[List[str], str, bool]:
        nonlocal _warned_no_random_limit, _warned_no_max_batch_flag, \
            _warned_no_generate_max_batch_flag, _warned_no_task_args_flag
        cmd = entry_prefix + base_args + ["--tasks", task, "--device", device]
        GENERATE_BS1 = {"gsm8k", "svamp", "aime25"}
        is_generate = task in GENERATE_BS1
        batch_value = str(override_batch) if override_batch is not None else (
            default_generate_batch_size if is_generate else default_batch_size
        )
        cmd += ["--batch_size", batch_value]
        target_max = effective_generate_max_batch_size if is_generate else effective_max_batch_size
        if target_max > 0:
            flag = None
            if is_generate and _generate_max_batch_flag:
                flag = _generate_max_batch_flag
            elif _max_batch_flag:
                flag = _max_batch_flag
            if flag:
                cmd += [flag, str(target_max)]
            else:
                if is_generate and not _warned_no_generate_max_batch_flag:
                    print("[lm-eval] Warning: CLI lacks a generate max batch size flag; skipping cap.")
                    _warned_no_generate_max_batch_flag = True
                elif not is_generate and not _warned_no_max_batch_flag:
                    print("[lm-eval] Warning: CLI lacks a max batch size flag; skipping cap.")
                    _warned_no_max_batch_flag = True
        if isinstance(limit, (int, float)):
            cmd += ["--limit", str(limit)]
            # Prefer randomized subset selection when supported by this lm-eval version
            if _has_limit_type:
                cmd += ["--limit-type", "random"]
            elif _has_limit_mode:
                cmd += ["--limit_mode", "random"]
            else:
                if not _warned_no_random_limit:
                    print("[lm-eval] Warning: CLI lacks a random subset flag; using first-N for --limit.")
                    _warned_no_random_limit = True
        if task_split_overrides:
            base_task = task.split(",", 1)[0]
            override_split = task_split_overrides.get(task) or task_split_overrides.get(base_task)
            if override_split:
                if _task_args_flag:
                    cmd += [_task_args_flag, f"{base_task}:split={override_split}"]
                elif not _warned_no_task_args_flag:
                    print("[lm-eval] Warning: CLI lacks a task args flag; cannot override split.")
                    _warned_no_task_args_flag = True
        return cmd, batch_value, is_generate

    # Parallel across physical GPUs (mask each process to one GPU)
    task_status: Dict[str, str] = {}
    task_durations: Dict[str, float] = {}
    good_any = False
    i = 0
    shared_gpu_env = ",".join(str(x) for x in gpu_ids)
    while i < len(tasks_with_limits):
        procs = []
        if share_gpu_pool:
            wave = tasks_with_limits[i : i + 1]
            gpu_assignments = [shared_gpu_env]
        else:
            wave = tasks_with_limits[i : i + len(gpu_ids)]
            gpu_assignments = [str(g) for g in gpu_ids[: len(wave)]]
        # If the parent environment is already masked (common under Slurm), the only
        # valid device ordinals are 0..N-1 inside the process, so we must use cuda:0.
        parent_has_cuda_mask = bool(os.environ.get("CUDA_VISIBLE_DEVICES", "").strip())
        wave_names = [t for (t, _) in wave]
        timestamp = datetime.now().strftime("%H:%M:%S")
        if share_gpu_pool:
            print(f"[{timestamp}] [lm-eval] Launching wave: {wave_names} on shared GPUs {shared_gpu_env}")
        else:
            print(f"[{timestamp}] [lm-eval] Launching wave: {wave_names} on GPUs {gpu_ids}")
        for (task, limit), gpu_assignment in zip(wave, gpu_assignments):
            env = os.environ.copy()
            # IMPORTANT: When a single GPU is available/selected, do not override
            # CUDA_VISIBLE_DEVICES here. This lets Slurm (or the caller) control
            # which physical GPU is exposed, while the subprocess always uses cuda:0.
            #
            # When multiple GPUs are used, we still mask each subprocess to its assigned GPU
            # (or the shared pool) so tasks can run concurrently.
            # When running multiple GPUs concurrently, mask each subprocess to its assigned GPU/pool.
            # When running a single GPU, prefer the parent's CUDA mask if present; otherwise, use the
            # physical GPU index directly via --device cuda:N.
            if len(gpu_ids) > 1:
                env["CUDA_VISIBLE_DEVICES"] = gpu_assignment  # inside, cuda:0 maps within this pool
            py_path_parts = [include_path, repo_root]
            if env.get("PYTHONPATH"):
                py_path_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = ":".join(py_path_parts)
            # Propagate seed-related env for deterministic child runs
            env.setdefault("PYTHONHASHSEED", str(seed))
            env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
            env.setdefault("EVAL_SEED", str(seed))
            env.setdefault("OMP_NUM_THREADS", "1")
            _ensure_allocator_safe_env(env)
            device = "cuda:0" if parent_has_cuda_mask else f"cuda:{gpu_assignment.split(',')[0]}"
            if use_fsdp:
                env.setdefault("MASTER_ADDR", "127.0.0.1")
                env.setdefault("MASTER_PORT", str(20000 + random.randint(0, 20000)))
                env["SAMPLEDKD_USE_FSDP"] = "1"
                env["SAMPLEDKD_FSDP_LAYER_CLS"] = fsdp_layer_cls or ""
                env["SAMPLEDKD_FSDP_DTYPE"] = model_dtype or "float16"
                env["SAMPLEDKD_FSDP_POLICY"] = fsdp_policy or "full_shard auto_wrap"
                device = "cuda"
            cmd, bs_used, _ = _task_cmd(task, limit, device)
            start = time.time()
            p = run_async(cmd, env=env)
            timeout = TASK_TIMEOUTS.get(task, 3600)
            procs.append((task, p, timeout, start, env, limit, device, bs_used))
        for task, p, to, start, env, limit, device, bs_used in procs:
            rc, out = wait_with_timeout(p, timeout=to)
            duration = max(0.0, time.time() - start)
            fallback_used = False
            if (
                rc not in (0, 124)
                and effective_fallback_batch_size > 0
                and "auto" in str(bs_used).lower()
            ):
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] 🔁 Retrying {task} with fallback batch size {effective_fallback_batch_size} after failure at batch_size={bs_used}"
                )
                fallback_cmd, fb_batch, _ = _task_cmd(
                    task,
                    limit,
                    device,
                    override_batch=str(effective_fallback_batch_size),
                )
                fb_start = time.time()
                p_fb = run_async(fallback_cmd, env=env)
                rc, out = wait_with_timeout(p_fb, timeout=to)
                duration += max(0.0, time.time() - fb_start)
                bs_used = fb_batch
                fallback_used = True
            timestamp = datetime.now().strftime("%H:%M:%S")
            if rc == 0:
                status_suffix = " (fallback)" if fallback_used else ""
                print(f"[{timestamp}] ✅ Task {task} completed successfully in {duration:.1f}s{status_suffix}")
                task_status[task] = "ok"
                good_any = True
            elif rc == 124:
                print(f"[{timestamp}] ⏰ Task {task} timed out after {to} seconds (wall {duration:.1f}s)")
                task_status[task] = "timeout"
            else:
                print(f"[{timestamp}] ❌ Task {task} failed with code {rc} in {duration:.1f}s")
                task_status[task] = f"failed:{rc}"
            task_durations[task] = duration
        i += len(wave)
    return out_dir if good_any else None, task_status, task_durations

# ---------- ECE (Expected Calibration Error) from LM-Eval samples ----------
def _softmax(xs: List[float]) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]

def compute_ece(confidences: List[float], correctness: List[int], n_bins: int = 15) -> float:
    """Compute Expected Calibration Error for top-1 predictions.

    Args:
        confidences: list of predicted top-1 probabilities in [0,1]
        correctness: list of 0/1 indicating if prediction was correct
        n_bins: number of equal-width bins over [0,1]
    Returns:
        Scalar ECE value in [0,1].
    """
    assert len(confidences) == len(correctness)
    n = len(confidences)
    if n == 0:
        return float("nan")
    # Initialize bins
    bins: List[List[int]] = [[] for _ in range(n_bins)]
    bin_confs: List[List[float]] = [[] for _ in range(n_bins)]
    for c, y in zip(confidences, correctness):
        # Clamp to [0,1]
        c = 0.0 if c < 0 else (1.0 if c > 1 else c)
        # Rightmost-inclusive for c==1.0
        idx = min(int(c * n_bins), n_bins - 1)
        bins[idx].append(int(y))
        bin_confs[idx].append(float(c))
    ece = 0.0
    for i in range(n_bins):
        m = len(bins[i])
        if m == 0:
            continue
        acc_i = sum(bins[i]) / m
        conf_i = sum(bin_confs[i]) / m
        ece += (m / n) * abs(acc_i - conf_i)
    return ece

def _parse_sample_line_for_mc(line: Dict[str, Any]) -> Optional[Tuple[List[float], int]]:
    """Attempt to extract (choice_scores, gold_index) from an lm-eval sample JSONL line.

    Returns list of scores (higher is better, e.g., log-likelihoods) and gold index.
    Returns None if the line does not look like a multiple-choice sample with usable data.
    """

    def _maybe_int(value: Any) -> Optional[int]:
        if isinstance(value, int):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    def _maybe_score(entry: Any) -> Optional[float]:
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, str):
            try:
                return float(entry)
            except ValueError:
                return None
        if isinstance(entry, dict):
            for key in ("loglikelihood", "logprob", "score", "value"):
                if key in entry:
                    val = _maybe_score(entry[key])
                    if val is not None:
                        return val
        if isinstance(entry, list) and entry:
            for item in entry:
                val = _maybe_score(item)
                if val is not None:
                    return val
        return None

    # Try common fields for label index, including modern lm-eval layouts
    gold_idx: Optional[int] = None
    for k in ("label", "gold_idx", "gold_index", "target"):
        cand = _maybe_int(line.get(k))
        if cand is not None:
            gold_idx = cand
            break
    if gold_idx is None:
        doc = line.get("doc")
        if isinstance(doc, dict):
            for k in ("label", "gold", "gold_idx"):
                cand = _maybe_int(doc.get(k))
                if cand is not None:
                    gold_idx = cand
                    break
    if gold_idx is None:
        # Some tasks log gold as letter (A/B/C/D)
        gold = line.get("gold") or line.get("answer") or (line.get("doc") or {}).get("gold")
        if isinstance(gold, str) and len(gold) == 1 and gold in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            gold_idx = ord(gold) - ord('A')

    # Collect choice scores
    scores: Optional[List[float]] = None
    raw_choice_scores = line.get("choice_scores")
    if isinstance(raw_choice_scores, list) and all(isinstance(x, (int, float)) for x in raw_choice_scores):
        scores = [float(x) for x in raw_choice_scores]

    if scores is None and isinstance(line.get("choices"), list) and line["choices"]:
        cand_keys = ("loglikelihood", "logprob", "score")
        first = line["choices"][0]
        if isinstance(first, dict):
            for k in cand_keys:
                if k in first and all(isinstance(c.get(k), (int, float)) for c in line["choices"]):
                    scores = [float(c[k]) for c in line["choices"]]
                    break

    if scores is None:
        # Modern lm-eval emits nested lists under filtered_resps/resps with stringified scores
        for resp_key in ("filtered_resps", "resps"):
            resp_blob = line.get(resp_key)
            if isinstance(resp_blob, list) and resp_blob:
                extracted: List[float] = []
                for entry in resp_blob:
                    val = _maybe_score(entry)
                    if val is None:
                        extracted = []
                        break
                    extracted.append(float(val))
                if extracted:
                    scores = extracted
                    break

    if scores is None or gold_idx is None:
        return None
    if not (0 <= gold_idx < len(scores)):
        return None
    return scores, gold_idx

def collect_ece_from_lmeval_samples(lmeval_dir: Optional[Path], bins: int = 15) -> Dict[str, Dict[str, float]]:
    """Parse lm-eval sample logs (if present) and compute ECE per multiple-choice task.

    Looks for samples/*.jsonl under the lm-eval output directory, extracts per-example
    choice scores and gold labels, converts scores to probabilities with softmax,
    and computes top-1 ECE.
    """
    if not lmeval_dir or not lmeval_dir.exists():
        print(f"[ECE] lmeval_dir does not exist: {lmeval_dir}")
        return {}
    # Gather all sample files
    sample_files = list(lmeval_dir.rglob("*.jsonl"))
    print(f"[ECE] Found {len(sample_files)} .jsonl files in {lmeval_dir}")
    # Filter to the typical samples/ subdir first if present
    preferred = [p for p in sample_files if "samples" in p.as_posix().split("/")]
    files = preferred or sample_files
    print(f"[ECE] Using {len(files)} sample files (preferred={len(preferred)})")
    per_task: Dict[str, Dict[str, float]] = {}

    def _normalize_task_from_filename(stem: str) -> str:
        # Drop common suffixes like "_samples" or ".samples"
        if stem.endswith("_samples"):
            stem = stem[: -len("_samples")]
        if stem.endswith(".samples"):
            stem = stem[: -len(".samples")]
        if stem.startswith("samples_"):
            stem = stem[len("samples_") :]
        # Drop trailing ISO-like timestamps that lm-eval appends after the task name
        if "_" in stem:
            prefix, suffix = stem.rsplit("_", 1)
            if suffix.startswith("20") and "T" in suffix:
                stem = prefix
        # If there is a normalization variant like ",none", trim to base task
        if "," in stem:
            stem = stem.split(",", 1)[0]
        return stem
    for p in files:
        try:
            # Task name from filename (strip extension and normalize)
            task_name = _normalize_task_from_filename(p.stem)
            confidences: List[float] = []
            correctness: List[int] = []
            with open(p, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    try:
                        obj = json.loads(ln)
                    except Exception:
                        continue
                    parsed = _parse_sample_line_for_mc(obj)
                    if parsed is None:
                        continue
                    scores, gold_idx = parsed
                    probs = _softmax(scores)
                    if not probs:
                        continue
                    pred_idx = int(max(range(len(probs)), key=lambda i: probs[i]))
                    conf = float(probs[pred_idx])
                    confidences.append(conf)
                    correctness.append(1 if pred_idx == gold_idx else 0)
            if len(confidences) >= 10:
                ece = compute_ece(confidences, correctness, n_bins=bins)
                per_task.setdefault(task_name, {})["ece"] = float(ece)
                per_task[task_name]["ece_n"] = float(len(confidences))
                print(f"[ECE] {task_name}: ECE={ece:.4f} (n={len(confidences)})")
            else:
                print(f"[ECE] {task_name}: Skipped (only {len(confidences)} MC samples, need >=10)")
        except Exception as e:
            print(f"Failed to compute ECE from {p}: {e}")
    return per_task

def _map_ece_to_existing_tasks(ece_metrics: Dict[str, Dict[str, float]],
                               lmeval_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Map ECE task keys to match the keys present in lm-eval metrics when possible.

    lm-eval often uses task keys like "arc_challenge,none" while sample files may yield
    base names like "arc_challenge". This function attempts to align ECE entries to the
    exact keys used in lmeval_metrics so they appear under the same task in the JSON.
    """
    if not ece_metrics or not lmeval_metrics:
        return ece_metrics

    # Build index from base -> full task keys found in lmeval
    base_to_full: Dict[str, List[str]] = {}
    for full_key in lmeval_metrics.keys():
        base_key = full_key.split(",", 1)[0]
        base_to_full.setdefault(base_key, []).append(full_key)

    remapped: Dict[str, Dict[str, float]] = {}
    for ece_key, metrics in ece_metrics.items():
        base_key = ece_key.split(",", 1)[0]
        candidates = base_to_full.get(base_key, [])
        if len(candidates) == 1:
            remapped[candidates[0]] = metrics
        else:
            # If ambiguous or no match, keep original key
            remapped[ece_key] = metrics
    return remapped

# ---------- Lighteval (summarization) ----------
def run_lighteval(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    """Run summarization only for HEAVY suite (CNN/DM, XSum)."""
    if suite != "heavy":
        print("[lighteval] Skipping summarization for light suite.")
        return None
    out_path = results_dir / f"lighteval_{tag}.json"
    tasks = [t.replace("-", "_") for t in LIGHTEVAL_TASKS]  # normalize hyphen → underscore
    code = 1
    for cmd in (
        [
            "lighteval", "--model", "hf", str(model_dir),
            "--tasks", ",".join(tasks), "--results", str(out_path),
        ],
        [
            sys.executable, "-m", "lighteval",
            "--model", "hf", str(model_dir),
            "--tasks", ",".join(tasks), "--results", str(out_path),
        ],
    ):
        code, _ = run(cmd)
        if code == 0:
            break
    return out_path if code == 0 and out_path.exists() else None

# ---------- EvalPlus (code: HumanEval/MBPP) ----------
def run_evalplus(model_dir: Path, tag: str, datasets: List[str], suite: str) -> Dict[str, Optional[Path]]:
    """Run code-gen only for HEAVY suite by default."""
    if suite != "heavy":
        print("[evalplus] Skipping code-gen for light suite.")
        return {}
    out = {}
    evalplus_home = Path(os.environ.get("TMPDIR", "/tmp")) / "evalplus_home"
    (evalplus_home / ".cache").mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["HOME"] = str(evalplus_home)
    env["XDG_CACHE_HOME"] = str(evalplus_home / ".cache")
    env.setdefault("EVALPLUS_TRUST_REMOTE_CODE", "1")
    for ds_name in datasets:
        cmd = [
            "evalplus.evaluate",
            "--model", str(model_dir),
            "--dataset", ds_name,
            "--backend", "hf",
            "--greedy",
        ]
        code, _ = run(cmd, env=env)
        results_root = Path("evalplus_results") / ds_name
        out[ds_name] = results_root if results_root.exists() and code == 0 else None
    return out

# ---------- AlpacaEval 2 (LC win-rates) ----------
def run_alpacaeval(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    if suite != "heavy":
        print("[alpacaeval] Skipping for light suite.")
        return None
    if os.getenv("OPENAI_API_KEY") is None:
        print("Skipping AlpacaEval 2: OPENAI_API_KEY not set.")
        return None
    out_dir = results_dir / f"alpacaeval_{tag}"
    ensure_dir(out_dir)
    cmd = [
        "alpaca_eval",
        "evaluate_from_model",
        "--model", str(model_dir),
        "--output_path", str(out_dir),
    ]
    code, _ = run(cmd)
    return out_dir if code == 0 else None

# ---------- IF-Eval (deterministic, strict) ----------
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _sample_rows_seeded(rows: List[Dict[str, Any]], *, limit: int, seed: int) -> List[Dict[str, Any]]:
    if limit <= 0 or limit >= len(rows):
        return rows
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    keep = sorted(indices[:limit])
    return [rows[i] for i in keep]


def _generate_det_response(model: AutoModelForCausalLM, tok: AutoTokenizer, prompt: str) -> str:
    encoded = tok(prompt, return_tensors="pt")
    input_ids = encoded.get("input_ids")
    if input_ids is None:
        return ""
    try:
        first_device = next(model.parameters()).device
    except Exception:
        first_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoded = {k: v.to(first_device) for k, v in encoded.items() if hasattr(v, "to")}

    # Some pretrained models ship `generation_config.json` with sampling params set
    # (e.g., temperature/top_p/top_k). When `do_sample=False` this triggers warnings.
    # Create an explicit deterministic GenerationConfig for this call.
    gen_cfg = copy.deepcopy(getattr(model, "generation_config", GenerationConfig()))
    gen_cfg.do_sample = False
    gen_cfg.temperature = None
    gen_cfg.top_p = None
    gen_cfg.top_k = None

    with torch.no_grad():
        out = model.generate(
            **encoded,
            max_new_tokens=IF_EVAL_MAX_NEW_TOKENS,
            generation_config=gen_cfg,
            pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
        )

    gen_ids = out[0]
    prompt_len = int(encoded["input_ids"].shape[-1])
    completion_ids = gen_ids[prompt_len:]
    text = tok.decode(completion_ids, skip_special_tokens=True)
    return (text or "").strip()


def run_ifeval_deterministic(
    model_dir: Path,
    tag: str,
    results_dir: Path,
    *,
    suite: str,
    seed: int,
    phase: str = "test",
) -> Tuple[Dict[str, Dict[str, float]], Optional[Path]]:
    """Run deterministic IF-Eval strict mode using the vendored evaluator.

    Produces artifacts under results_dir/ifeval_deterministic_{tag}/.
    Returns metrics in 0-100 scale as {"ifeval_deterministic": {"strict_accuracy": score}}.

    Always runs the full IFEval benchmark (no validation/test splitting).

    Hard-fails on missing prompts/responses or mismatched counts.
    """
    _ = suite  # reserved
    _ = phase  # always run full benchmark
    input_data = Path(__file__).parent / "ifeval" / "data" / "input_data.jsonl"
    if not input_data.exists():
        raise FileNotFoundError(f"Missing IF-Eval input file: {input_data}")

    out_dir = results_dir / f"ifeval_deterministic_{tag}"
    ensure_dir(out_dir)

    rows = _read_jsonl(input_data)
    print(f"[ifeval] Running full IFEval benchmark ({len(rows)} samples)")
    
    # Apply additional sampling limit if configured
    rows = _sample_rows_seeded(rows, limit=IF_EVAL_SAMPLE_LIMIT, seed=seed)
    sampled_input_path = out_dir / "input_data_used.jsonl"
    _write_jsonl(sampled_input_path, rows)

    model_ref = str(model_dir)
    tok = load_tokenizer_with_fallback(model_dir)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_ref,
        trust_remote_code=True,
        dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))

    response_rows: List[Dict[str, Any]] = []
    for ex in rows:
        prompt = ex.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("IF-Eval input contains a missing/invalid 'prompt' field")
        response = _generate_det_response(model, tok, prompt)
        response_rows.append({"prompt": prompt, "response": response})

    input_response_path = out_dir / "input_response_data.jsonl"
    _write_jsonl(input_response_path, response_rows)

    code, _ = run(
        [
            sys.executable,
            "-m",
            "sampledkd.evaluations.ifeval.evaluation_main",
            "--input_data",
            str(sampled_input_path),
            "--input_response_data",
            str(input_response_path),
            "--output_dir",
            str(out_dir),
        ]
    )
    if code != 0:
        raise RuntimeError(f"Deterministic IF-Eval failed with exit code {code}")

    strict_path = out_dir / "eval_results_strict.jsonl"
    if not strict_path.exists():
        raise FileNotFoundError(f"Missing IF-Eval strict output: {strict_path}")

    strict_rows = _read_jsonl(strict_path)
    if len(strict_rows) != len(rows):
        raise RuntimeError(
            f"IF-Eval strict outputs count mismatch: got {len(strict_rows)} rows, expected {len(rows)}"
        )

    followed = 0
    for r in strict_rows:
        if r.get("follow_all_instructions") is True:
            followed += 1
    acc = followed / max(len(strict_rows), 1)
    score = float(acc) * 100.0
    metrics = {"ifeval_deterministic": {"strict_accuracy": score}}
    return metrics, out_dir

# ---------- Safety (JailbreakBench + HarmBench) ----------
def run_jailbreakbench(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    if suite != "heavy":
        print("[jbb] Skipping for light suite.")
        return None
    base_url = os.getenv("JBB_BASE_URL")
    model_name = os.getenv("JBB_MODEL")
    if not which("python") or base_url is None or model_name is None:
        print("Skipping JailbreakBench: set JBB_BASE_URL and JBB_MODEL to use an OpenAI-style endpoint.")
        return None
    out_dir = results_dir / f"jbb_{tag}"
    ensure_dir(out_dir)
    shim = f"""
import json, os
import jailbreakbench as jbb
base_url = os.environ.get("JBB_BASE_URL")
model = os.environ.get("JBB_MODEL")
prompts = jbb.load_default_prompts()
evaluation = jbb.evaluate_prompts(prompts, llm_provider="litellm", base_url=base_url, model=model)
with open("{(out_dir / 'jbb_results.json').as_posix()}", "w") as f:
    json.dump(evaluation, f)
print("Wrote JBB results")
"""
    code, _ = run([sys.executable, "-c", shim])
    return out_dir if code == 0 else None

def run_harmbench(model_dir: Path, tag: str, results_dir: Path, suite: str) -> Optional[Path]:
    if suite != "heavy":
        print("[harmbench] Skipping for light suite.")
        return None
    if importlib.util.find_spec("harmbench") is None:
        print("Skipping HarmBench: package 'harmbench' not installed.")
        return None
    config = os.getenv("HARMBENCH_CONFIG")
    if config is None:
        print("Skipping HarmBench: set HARMBENCH_CONFIG to a YAML that points to your HF model.")
        return None
    out_dir = results_dir / f"harmbench_{tag}"
    ensure_dir(out_dir)
    code, _ = run([sys.executable, "-m", "harmbench", "--config", config])
    return out_dir if code == 0 else None


# ---------- Results aggregation ----------
def collect_lmeval_metrics(lmeval_dir: Path) -> Dict[str, Dict[str, float]]:
    if not lmeval_dir or not lmeval_dir.exists():
        return {}
    results: Dict[str, Dict[str, float]] = {}
    # Look for both per-task and aggregated outputs
    patterns = [
        "results.json",
        "aggregated_results.json",
        "*results*.json",
        "*aggregated*.json",
        "*results*.yaml",
    ]
    targets: List[Path] = []
    for pat in patterns:
        targets.extend(lmeval_dir.rglob(pat))
    if not targets:
        return {}
    if not targets:
        return {}
    for p in targets:
        try:
            if p.suffix == ".yaml":
                blob = yaml.safe_load(open(p))
            else:
                blob = json.load(open(p))
            r = blob.get("results", blob if isinstance(blob, dict) else {})
            if not isinstance(r, dict):
                continue
            for task, metrics in r.items():
                if not isinstance(metrics, dict):
                    continue
                results.setdefault(task, {})
                for mk, mv in metrics.items():
                    if isinstance(mv, dict) and "value" in mv:
                        v = mv.get("value")
                        if isinstance(v, (int, float)):
                            results[task][mk] = float(v)
                        se = mv.get("stderr")
                        if isinstance(se, (int, float)):
                            results[task][f"{mk}_stderr"] = float(se)
                    elif isinstance(mv, (int, float)):
                        results[task][mk] = float(mv)
        except Exception as e:
            print(f"Failed to parse {p}: {e}")
    return results

def collect_lighteval_metrics(out_file: Path) -> Dict[str, Dict[str, float]]:
    if not out_file or not out_file.exists():
        return {}
    try:
        blob = json.load(open(out_file))
        return {
            task: {mk: float(mv) for mk, mv in task_res.items() if isinstance(mv, (int, float))}
            for task, task_res in blob.items()
        }
    except Exception as e:
        print(f"Failed to parse {out_file}: {e}")
    return {}

def collect_evalplus_metrics(root: Optional[Path], ds_name: str) -> Dict[str, Dict[str, float]]:
    if not root or not root.exists():
        return {}
    candidates = [p for name in ("summary.json", "report.json", "scores.json") for p in root.rglob(name)]
    if not candidates:
        return {}
    p = max(candidates, key=lambda x: x.stat().st_mtime)
    try:
        blob = json.load(open(p))
        metrics = {
            k: float(blob[k])
            for k in ["pass@1", "pass@5", "pass@10", "pass@100", "mbpp_score", "humaneval_score"]
            if isinstance(blob.get(k), (int, float))
        }
        metrics.update(
            {
                k: float(v)
                for k, v in blob.items()
                if k not in metrics and isinstance(v, (int, float)) and ("pass" in k or "score" in k or "acc" in k)
            }
        )
        return {ds_name: metrics}
    except Exception as e:
        print(f"Failed to parse EvalPlus summary at {p}: {e}")
    return {}

def collect_alpacaeval_metrics(out_dir: Optional[Path]) -> Dict[str, Dict[str, float]]:
    if not out_dir or not out_dir.exists():
        return {}
    for p in out_dir.rglob("*.json"):
        try:
            blob = json.load(open(p))
            if isinstance(blob, dict):
                vals = {
                    k: float(blob[k])
                    for k in ["length_controlled_win_rate", "win_rate", "pairwise_win_rate", "length_controlled"]
                    if isinstance(blob.get(k), (int, float))
                }
                if vals:
                    return {"AlpacaEval2": vals}
        except Exception:
            continue
    return {}

def collect_simple_json(root: Optional[Path], task_name: str, filename: str) -> Dict[str, Dict[str, float]]:
    if not root:
        return {}
    p = root / filename
    if not p.exists():
        return {}
    try:
        blob = json.load(open(p))
        if isinstance(blob, dict):
            return {
                task_name: {k: float(v) for k, v in blob.items() if isinstance(v, (int, float))}
            }
    except Exception as e:
        print(f"Failed to parse {p}: {e}")
    return {}

def merge_model_results(per_source: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    merged = {}
    for src in per_source:
        for task, metrics in src.items():
            merged.setdefault(task, {})
            for mk, mv in metrics.items():
                merged[task][mk] = mv
    return merged

def print_latex_table(all_models_metrics: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    all_metrics = sorted({m for model in all_models_metrics.values() for task in model.values() for m in task.keys()})
    tasks = sorted({t for model in all_models_metrics.values() for t in model.keys()})
    if not all_metrics or not tasks:
        print("\n% No metrics found — check timeouts/include_path. Skipping table.\n")
        return
    def _disp_metric(m: str) -> str:
        # Clean up filter suffixes like ",none" in lm-eval keys for nicer LaTeX headers
        if "," in m:
            head, tail = m.split(",", 1)
            return head if tail == "none" else m
        return m
    print("\n% ---- LaTeX table (booktabs) ----")
    print(r"\begin{table}")
    print(r"\centering")
    print(r"\caption{Evaluation across public benchmarks (same decoding params across systems).}")
    print(r"\begin{tabular}{l" + "c" * (len(all_metrics) * len(all_models_metrics)) + r"}")
    print(r"\toprule")
    header = ["Task"]
    model_names = list(all_models_metrics.keys())
    for mname in model_names:
        for met in all_metrics:
            header.append(f"{mname}:{_disp_metric(met)}")
    print(" & ".join(header) + r" \\")
    print(r"\midrule")
    for t in tasks:
        row = [t]
        for mname in model_names:
            mm = all_models_metrics[mname].get(t, {})
            for met in all_metrics:
                val = mm.get(met, None)
                if isinstance(val, float):
                    # Avoid printing NaN/inf in LaTeX
                    row.append(f"{val:.2f}" if math.isfinite(val) else "-")
                elif isinstance(val, int):
                    row.append(str(val))
                else:
                    row.append("-")
        print(" & ".join(row) + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

def save_json(obj: dict, path: Path) -> None:
    """Persist a Python dict to pretty-printed UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    print(f"[json] Wrote {path}")

DISPLAY_NAME_ENV_VARS = (
    "RUN_DISPLAY_NAME",
    "EVAL_DISPLAY_NAME",
    "DISPLAY_NAME",
    "METHOD_NAME",
)

LIGHT_TASK_HEADERS = {
    "arc_easy": "Arc-E",
    "gsm8k": "GSM8K (%)",
    "hellaswag": "HellaSwag (%)",
    "piqa": "PIQA (%)",
    "lambada_openai_acc": "LAMBADA OAI acc. (%)",
    "lambada_openai_ppl": "LAMBADA OAI perplexity",
    "avg": "avg.",
    "avg_ece": "Avg ECE (%)",
    "ifeval_deterministic": "IFEval",
}

PERPLEXITY_KEYS = (
    "perplexity,none",
    "perplexity",
    "word_perplexity,none",
    "byte_perplexity,none",
)


def compute_averages(merged: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Compute macro-averages across tasks for each numeric metric.

    - Excludes keys ending with '_stderr'.
    - Excludes obvious count-like fields such as 'ece_n'.
        - Returns per-metric averages (excluding IF tasks) and specialized averages:
      * 'avg_all' is the mean across non-IF tasks of the primary metric (prefers 'exact_match,none', falls back to 'acc,none').
            * 'if_avg' is the mean across IF tasks of the same primary metric.
    """
    regular_metric_values: Dict[str, List[float]] = {}
    regular_primary: List[float] = []
    if_primary: List[float] = []

    for task, task_metrics in merged.items():
        if not isinstance(task_metrics, dict):
            continue
        for mk, mv in task_metrics.items():
            if not isinstance(mv, (int, float)):
                continue
            if mk.endswith("_stderr"):
                continue
            if mk in {"ece_n"}:
                continue
            mv_f = float(mv)
            if not math.isfinite(mv_f):
                continue
            base_task = task.split(",", 1)[0]
            if base_task in IF_TASK_NAMES:
                continue
            regular_metric_values.setdefault(mk, []).append(mv_f)
        base_task = task.split(",", 1)[0]
        primary_val = _extract_primary_metric(task_metrics)
        if primary_val is None:
            continue
        if base_task in IF_TASK_NAMES:
            if_primary.append(primary_val)
        else:
            regular_primary.append(primary_val)

    averages: Dict[str, float] = {}
    for mk, vals in regular_metric_values.items():
        if vals:
            averages[f"avg_{mk}"] = sum(vals) / len(vals)

    if regular_primary:
        averages["avg_all"] = sum(regular_primary) / len(regular_primary)
    if if_primary:
        averages["if_avg"] = sum(if_primary) / len(if_primary)
    return averages


def _extract_primary_metric(metrics: Dict[str, Any]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    for key in (
        "strict_accuracy",
        "exact_match,strict-match",
        "exact_match,flexible-extract",
        "exact_match,none",
        "acc,none",
        "acc_norm,none",
        "accuracy",
        "acc",
    ):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            val_f = float(val)
            if math.isfinite(val_f):
                return val_f
    return None


def _extract_perplexity(metrics: Dict[str, Any]) -> Optional[float]:
    if not isinstance(metrics, dict):
        return None
    for key in PERPLEXITY_KEYS:
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            val_f = float(val)
            if math.isfinite(val_f):
                return val_f
    return None


def _format_table_value(value: Optional[float], *, as_percent: bool = False) -> str:
    if value is None:
        return "---"
    if not math.isfinite(value):
        return "---"
    display_value = value * 100.0 if as_percent else value
    return f"{display_value:.1f}"


def _normalize_ece_value(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value * 100.0 if value < 1.0 else value


def _build_table_header_columns(
    tasks: List[Tuple[str, Optional[int]]],
    *,
    include_avg_ece: bool = False,
) -> List[str]:
    columns = ["Method"]
    for task_name, _ in tasks:
        if task_name == "lambada_openai":
            columns.append(LIGHT_TASK_HEADERS["lambada_openai_acc"])
            columns.append(LIGHT_TASK_HEADERS["lambada_openai_ppl"])
        else:
            columns.append(LIGHT_TASK_HEADERS.get(task_name, task_name))
    columns.append(LIGHT_TASK_HEADERS["avg"])
    if include_avg_ece:
        columns.append(LIGHT_TASK_HEADERS["avg_ece"])
    return columns


def _ensure_table_header(
    path: Path,
    tasks: List[Tuple[str, Optional[int]]],
    *,
    include_avg_ece: bool = False,
) -> None:
    needs_header = not path.exists() or path.stat().st_size == 0
    if not needs_header:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _build_table_header_columns(tasks, include_avg_ece=include_avg_ece)
    with open(path, "w", encoding="utf-8") as f:
        f.write(" & ".join(columns) + " \\\\\n")


def _parse_table_line(line: str) -> Optional[List[str]]:
    stripped = line.strip()
    if not stripped:
        return None
    core = stripped
    if core.endswith("\\"):
        core = core[:-1].rstrip()
    if core.endswith("\\"):
        core = core[:-1].rstrip()
    placeholder = "\x00AMP\x00"
    protected = core.replace("\\&", placeholder)
    columns = [part.strip().replace(placeholder, "&") for part in protected.split("&")]
    return columns if columns else None


def _normalize_row_length(row: List[str], expected_len: int, fill: str = "---") -> List[str]:
    if len(row) < expected_len:
        row = row + [fill] * (expected_len - len(row))
    return row[:expected_len]


def _format_table_rows(rows: List[List[str]]) -> List[str]:
    if not rows:
        return []
    num_cols = len(rows[0])
    widths: List[int] = []
    for idx in range(num_cols):
        col_values = [row[idx] for row in rows if idx < len(row)]
        longest = max((len(v) for v in col_values), default=0)
        min_width = 100 if idx == 0 else 50
        widths.append(max(longest, min_width))

    formatted_lines: List[str] = []
    for ridx, row in enumerate(rows):
        formatted_cols: List[str] = []
        is_header = ridx == 0
        for cidx, col in enumerate(row):
            target = col
            if not is_header:
                placeholder = "\x00AMP\x00"
                target = target.replace("\\&", placeholder)
                target = target.replace("&", "\\&")
                target = target.replace(placeholder, "\\&")
            width = widths[cidx]
            if is_header or cidx == 0:
                formatted_cols.append(target.ljust(width))
            else:
                formatted_cols.append(target.rjust(width))
        formatted_lines.append(" & ".join(formatted_cols) + " \\\\\n")
    return formatted_lines


def _write_csv_table(path: Path, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for row in rows:
            writer.writerow(row)


def append_table_row(
    path: Path,
    method: str,
    tasks: List[Tuple[str, Optional[int]]],
    merged: Dict[str, Dict[str, float]],
    averages: Dict[str, float],
) -> None:
    base_task_names = [task.split(",", 1)[0] for task, _ in tasks]
    all_if_tasks = bool(base_task_names) and all(
        base in IF_TASK_NAMES for base in base_task_names
    )
    if all_if_tasks:
        filtered_tasks = list(tasks)
    else:
        filtered_tasks = [
            (task_name, limit)
            for task_name, limit in tasks
            if task_name.split(",", 1)[0] not in IF_TASK_NAMES
        ]

    if not filtered_tasks:
        return

    include_avg_ece = not all_if_tasks
    _ensure_table_header(path, filtered_tasks, include_avg_ece=include_avg_ece)
    columns: List[str] = [method]
    as_percent = not all_if_tasks
    for task_name, _ in filtered_tasks:
        task_metrics = merged.get(task_name, {})
        if task_name == "lambada_openai":
            acc_val = _extract_primary_metric(task_metrics)
            ppl_val = _extract_perplexity(task_metrics)
            columns.append(_format_table_value(acc_val, as_percent=as_percent))
            columns.append(_format_table_value(ppl_val))
        else:
            val = _extract_primary_metric(task_metrics)
            columns.append(_format_table_value(val, as_percent=as_percent))
    avg_key = "if_avg" if all_if_tasks else "avg_all"
    columns.append(_format_table_value(averages.get(avg_key), as_percent=as_percent))
    if include_avg_ece:
        ece_val = averages.get("avg_ece")
        if ece_val is None:
            ece_values: List[float] = []
            for metrics in merged.values():
                if not isinstance(metrics, dict):
                    continue
                val = metrics.get("ece")
                if isinstance(val, (int, float)) and math.isfinite(float(val)):
                    ece_values.append(float(val))
            if ece_values:
                ece_val = sum(ece_values) / len(ece_values)
        columns.append(_format_table_value(_normalize_ece_value(ece_val)))
    csv_path = path.with_suffix(".csv")
    existing_rows: List[List[str]] = []
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.reader(f):
                    if row:
                        existing_rows.append(row)
        except Exception:
            existing_rows = []

    if not existing_rows and path.exists():
        existing_lines: List[str] = []
        with open(path, "r", encoding="utf-8") as f:
            existing_lines = [line for line in f.readlines() if line.strip()]
        for line in existing_lines:
            parsed = _parse_table_line(line)
            if parsed is not None:
                existing_rows.append(parsed)

    desired_header = _build_table_header_columns(filtered_tasks, include_avg_ece=include_avg_ece)
    existing_header = existing_rows[0] if existing_rows else []
    existing_body = existing_rows[1:] if len(existing_rows) > 1 else []
    header = desired_header

    expected_len = len(header)
    normalized_body: List[List[str]] = []
    if existing_body:
        key_mapping = {
            existing_header[idx]: idx
            for idx in range(len(existing_header))
        }
        for row in existing_body:
            projected_row = []
            for col_name in header:
                src_idx = key_mapping.get(col_name)
                value = row[src_idx] if src_idx is not None and src_idx < len(row) else "---"
                projected_row.append(value)
            normalized_body.append(_normalize_row_length(projected_row, expected_len))
    else:
        normalized_body = []

    # Append-only: do NOT remove existing rows with the same method name.
    # This preserves history and allows duplicates (user preference).
    new_row = _normalize_row_length(columns, expected_len)
    normalized_body.append(new_row)

    all_rows = [header] + normalized_body
    formatted = _format_table_rows(all_rows)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(formatted)

    _write_csv_table(csv_path, all_rows)


def append_table_row_csv_json(
    *,
    csv_path: Path,
    json_path: Path,
    method: str,
    tasks: List[Tuple[str, Optional[int]]],
    merged: Dict[str, Dict[str, float]],
    averages: Dict[str, float],
) -> None:
    """Append a row to a CSV table and keep a JSON mirror.

    Used for deterministic IF-Eval where only CSV+JSON artifacts are desired.
    """
    filtered_tasks = list(tasks)
    if not filtered_tasks:
        return

    header = _build_table_header_columns(filtered_tasks, include_avg_ece=False)
    expected_len = len(header)

    columns: List[str] = [method]
    # IF-Eval metrics are stored in 0-100 already.
    for task_name, _ in filtered_tasks:
        val = _extract_primary_metric(merged.get(task_name, {}))
        columns.append(_format_table_value(val, as_percent=False))
    columns.append(_format_table_value(averages.get("if_avg"), as_percent=False))
    new_row = _normalize_row_length(columns, expected_len)

    existing_rows: List[List[str]] = []
    if csv_path.exists():
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as handle:
                for row in csv.reader(handle):
                    if row:
                        existing_rows.append(row)
        except Exception:
            existing_rows = []

    if not existing_rows:
        all_rows = [header, new_row]
    else:
        existing_header = existing_rows[0]
        existing_body = existing_rows[1:]
        if existing_header != header:
            key_mapping = {existing_header[i]: i for i in range(len(existing_header))}
            projected_body: List[List[str]] = []
            for row in existing_body:
                projected = []
                for col_name in header:
                    src_idx = key_mapping.get(col_name)
                    projected.append(row[src_idx] if src_idx is not None and src_idx < len(row) else "---")
                projected_body.append(_normalize_row_length(projected, expected_len))
            all_rows = [header] + projected_body + [new_row]
        else:
            normalized_body = [_normalize_row_length(r, expected_len) for r in existing_body]
            all_rows = [header] + normalized_body + [new_row]

    _write_csv_table(csv_path, all_rows)
    json_obj = {
        "header": header,
        "rows": [
            {header[i]: row[i] for i in range(expected_len)}
            for row in all_rows[1:]
        ],
    }
    save_json(json_obj, json_path)

# ---------- Main pipeline ----------
def main():
    parser = argparse.ArgumentParser()
    # Single input: trained student model path (either HF dir or .pt checkpoint). We'll auto-detect.
    parser.add_argument("model", type=str, help="Path to trained student model: HF directory (preferred) or a .pt checkpoint. If --from_hf is set, this may be a HF hub ID (e.g., 'Qwen/Qwen3-8B').")
    # Parallel/GPU controls
    parser.add_argument("--gpu_ids", type=str, default=None,
                        help="Comma-separated physical GPU ids for parallel lm-eval (e.g., '0,1,2'). Defaults to visible GPUs.")
    parser.add_argument("--lm_eval_share_gpu_pool", action="store_true",
                        help="Treat provided gpu_ids as a single pool per lm-eval subprocess (tensor-parallel sharding for large models).")
    parser.add_argument("--use-fsdp", "--use_fsdp", dest="use_fsdp", action="store_true",
                        help="Wrap the evaluation model with PyTorch Fully Sharded Data Parallel (torch.distributed.run).")
    parser.add_argument("--fsdp_layer_cls", type=str, default=None,
                        help="Override the transformer layer class to wrap when using FSDP (e.g., 'Qwen3DecoderLayer').")
    parser.add_argument("--fsdp_policy", type=str, default="full_shard auto_wrap",
                        help="Value forwarded as fsdp=... in HF model_args when --use-fsdp is set (default: 'full_shard auto_wrap').")
    parser.add_argument("--max_parallel", type=int, default=None,
                        help="Cap the number of parallel lm-eval workers (<= number of provided GPUs).")
    parser.add_argument("--work_dir", type=str, default="eval_runs")
    # Some wrappers (e.g., Slurm scripts) may accidentally emit a bare `--output_dir`
    # when a variable is empty. Accept that and fall back to the default.
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_json_results",
        nargs="?",
        const="evaluation_json_results",
    )
    # Unified runs registry integration
    parser.add_argument("--runs_registry_validation", type=str, default="results/runs_validation.json",
                        help="Path to the validation split runs JSON registry.")
    parser.add_argument("--runs_registry_test", type=str, default="results/runs_test.json",
                        help="Path to the test split runs JSON registry.")
    parser.add_argument("--params_json", type=str, default=None,
                        help="Optional path to a JSON file containing the original training parameters. If not provided, attempts to infer from model dir (config.json).")
    # Logging configuration (default project updated per request)
    parser.add_argument("--wandb_project", type=str, default="selective-entropy-knowledge-distillation",
                        help="W&B project slug (e.g., 'selective-entropy-knowledge-distillation').")
    parser.add_argument("--disable_wandb", action="store_true", help="Disable W&B logging.")
    parser.add_argument("--disable_tensorboard", action="store_true", help="Disable TensorBoard logging.")
    parser.add_argument(
        "--include_test_split",
        action="store_true",
        help="Include the test split evaluation in addition to validation.",
    )
    # Suite selection
    parser.add_argument("--suite", type=str, choices=["light", "heavy"], default="light",
                        help="Evaluation suite to run: 'light' (quick) or 'heavy' (paper).")
    parser.add_argument(
        "--only_ifeval",
        action="store_true",
        help="Run only deterministic IF-Eval (strict, test-only) and skip all other benchmarks.",
    )
    # Source override: treat positional model arg as HF hub ID
    parser.add_argument("--from_hf", action="store_true", help="Interpret 'model' as a HF hub ID even if it is not a local path.")
    # LM-Eval batching controls
    parser.add_argument(
        "--lm_eval_batch_size",
        type=str,
        default="auto",
        help="Default batch size passed to lm-eval (--batch_size). Use 'auto' to probe GPU capacity.",
    )
    parser.add_argument(
        "--lm_eval_generate_batch_size",
        type=str,
        default="1",
        help="Batch size for long-form generation tasks (gsm8k, svamp, aime25).",
    )
    parser.add_argument(
        "--lm_eval_max_batch_size",
        type=int,
        default=0,
        help="Cap auto-detected batch sizes when lm-eval supports --max_batch_size (0 disables the cap).",
    )
    parser.add_argument(
        "--lm_eval_generate_max_batch_size",
        type=int,
        default=4,
        help="Cap auto-detected batch sizes on generation tasks when supported (0 disables the cap).",
    )
    parser.add_argument(
        "--lm_eval_fallback_batch_size",
        type=int,
        default=1,
        help="Fallback batch size to retry with if an auto batch run fails or times out.",
    )
    parser.add_argument(
        "--lm_eval_model_dtype",
        type=str,
        default=None,
        help="Torch dtype passed to HF model_args when not loading quantized weights (default: auto-detect from the model config).",
    )
    parser.add_argument(
        "--lm_eval_load_in_8bit",
        action="store_true",
        help="Load the evaluation model in 8-bit using bitsandbytes (reduces VRAM).",
    )
    parser.add_argument(
        "--lm_eval_load_in_4bit",
        action="store_true",
        help="Load the evaluation model in 4-bit using bitsandbytes (minimizes VRAM).",
    )
    parser.add_argument(
        "--lm_eval_device_map",
        type=str,
        default=None,
        help="Optional device_map string to forward to HF model_args (e.g., 'auto', 'balanced_low_0').",
    )
    # No vanilla/ekd distinction; one model at a time
    args = parser.parse_args()

    if args.lm_eval_load_in_8bit and args.lm_eval_load_in_4bit:
        print("Error: --lm_eval_load_in_8bit and --lm_eval_load_in_4bit are mutually exclusive.")
        sys.exit(2)
    
    # Print all CLI parameters at startup
    print("=" * 80)
    print("EVALUATION CONFIGURATION")
    print("=" * 80)
    args_dict = vars(args)
    for key, value in sorted(args_dict.items()):
        print(f"  {key:30s} = {value}")
    print("=" * 80)
    print()

    resolved_display_name: Optional[str] = None
    for env_key in DISPLAY_NAME_ENV_VARS:
        candidate = os.getenv(env_key)
        if candidate and candidate.strip():
            resolved_display_name = candidate.strip()
            break

    work_dir = Path(args.work_dir)
    exports_dir = work_dir / "exports"
    results_dir = work_dir / "results"
    json_results_dir = Path(args.output_dir)
    ensure_dir(exports_dir)
    ensure_dir(results_dir)
    ensure_dir(json_results_dir)

    default_test_registry = Path("results/runs_test.json")
    default_validation_registry = Path("results/runs_validation.json")
    runs_registry_test = Path(args.runs_registry_test or default_test_registry)
    runs_registry_validation = Path(args.runs_registry_validation or default_validation_registry)

    if runs_registry_validation == default_validation_registry and runs_registry_test != default_test_registry:
        runs_registry_validation = runs_registry_test.with_name("runs_validation.json")

    args.include_test_split = True  # always evaluate both validation and test splits

    evaluation_phases = [
        {
            "name": "validation",
            "registry_path": runs_registry_validation,
            "split_overrides": LM_EVAL_VALIDATION_SPLITS,
        },
    ]
    if args.include_test_split:
        evaluation_phases.append(
            {
                "name": "test",
                "registry_path": runs_registry_test,
                "split_overrides": LM_EVAL_TEST_SPLITS,
            }
        )

    # IF-Eval is deterministic and should always run on the test split.
    # If the user didn't request running the full test suite, append a synthetic
    # "test" phase that runs only deterministic IF-Eval so every eval.py run
    # produces IF-Eval artifacts/metrics.
    has_test_phase = any(p.get("name") == "test" for p in evaluation_phases)
    if not has_test_phase:
        evaluation_phases.append(
            {
                "name": "test",
                "registry_path": runs_registry_test,
                "split_overrides": LM_EVAL_TEST_SPLITS,
                "ifeval_only": True,
            }
        )

    # If the user explicitly asked for only IF-Eval, run only the test phase.
    if args.only_ifeval:
        evaluation_phases = [p for p in evaluation_phases if p.get("name") == "test"]

    # IF-Eval is deterministic and only runs on the test split (no validation mode).

    # Build GPU pool
    if args.gpu_ids:
        gpu_pool = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    else:
        gpu_pool = visible_gpu_ids()
    if args.max_parallel is not None and args.max_parallel > 0:
        gpu_pool = gpu_pool[: args.max_parallel]
    if args.lm_eval_share_gpu_pool and len(gpu_pool) < 2:
        print("[gpu] Warning: --lm_eval_share_gpu_pool requested but fewer than 2 GPUs available; running without sharding.")
        args.lm_eval_share_gpu_pool = False
    print(f"[gpu] Using GPUs: {gpu_pool}" if gpu_pool else "[gpu] No GPUs detected/selected.")

    # Resolve the single model dir to evaluate
    model_specs: List[Tuple[str, Path]] = []
    # If explicitly flagged as HF hub, skip local checks and create a temp export alias
    tag = None
    base_model_dir = None
    if args.from_hf:
        hub_id = args.model
        # Sanitize tag for filenames/logs
        tag = re.sub(r"[^A-Za-z0-9_.-]", "_", hub_id)
        # Use a pseudo-path that downstream tools accept (lm-eval accepts hub ids directly via --model_args)
        # For consistency with the rest of the pipeline that expects a Path, use the hub id as-is
        model_specs.append((tag, Path(hub_id)))
        base_model_dir = hub_id
    else:
        in_path = Path(args.model)
        tag = in_path.stem if in_path.is_file() else in_path.name
        if in_path.is_dir():
            print(f"[models] Using HF model directory: {in_path}")
            # Validate the directory contains required model files before proceeding
            try:
                validate_model_directory(in_path)
            except FileNotFoundError as e:
                print(f"Error: {e}")
                sys.exit(2)
            model_specs.append((tag, in_path))
        elif in_path.is_file() and in_path.suffix == ".pt":
            # Export HF dir based on base_model_dir recorded in checkpoint
            try:
                chk = torch.load(in_path, map_location="cpu")
            except Exception as e:
                print(f"Error loading checkpoint {in_path}: {e}")
                sys.exit(2)
            base_model_dir = chk.get("base_model_dir") or chk.get("base_model") or chk.get("student_model")
            if not base_model_dir:
                print("Error: checkpoint does not record base_model_dir. Re-train with newer code or evaluate a ready HF model directory.")
                sys.exit(2)
            export_dir = exports_dir / f"export_{in_path.stem}"
            print(f"[export] Exporting HF model from base='{base_model_dir}' and ckpt='{in_path.name}' -> '{export_dir}'")
            export_hf_model(base_model_dir, in_path, export_dir)
            model_specs.append((tag, export_dir))
        else:
            print(f"Error: provided model path is neither an HF directory nor a .pt file: {in_path}")
            sys.exit(2)
    if not model_specs:
        print("No models exported. Exiting.")
        sys.exit(1)

    fsdp_layer_cls = args.fsdp_layer_cls
    if args.use_fsdp and not fsdp_layer_cls:
        # Prefer the base HF dir if known; otherwise fall back to the resolved model path
        detect_target: Union[str, Path] = base_model_dir or model_specs[0][1]
        fsdp_layer_cls = detect_fsdp_layer_cls(detect_target)
    if args.use_fsdp and fsdp_layer_cls and not args.fsdp_layer_cls:
        args.fsdp_layer_cls = fsdp_layer_cls
    if args.use_fsdp and not fsdp_layer_cls:
        print("[fsdp] Warning: unable to identify a transformer layer class; wrapping the full model may be slower.")

    metrics_by_phase: Dict[str, Dict[str, Dict[str, float]]] = {
        phase["name"]: {} for phase in evaluation_phases
    }

    for tag, model_dir in model_specs:
        model_display_name = resolved_display_name
        for phase in evaluation_phases:
            phase_name = phase["name"]
            phase_registry_path = Path(phase["registry_path"])
            phase_tag = f"{tag}-{phase_name}"
            print(f"\n=== Running benchmarks for {tag} (suite={args.suite}, split={phase_name}) ===")
            phase_start = time.time()

            phase_results_dir = results_dir / phase_name
            phase_json_dir = json_results_dir / phase_name
            ensure_dir(phase_results_dir)
            ensure_dir(phase_json_dir)

            if args.only_ifeval or phase.get("ifeval_only"):
                lmeval_root = None
                task_status = {}
                task_durations = {}
                lmeval_wall = 0.0
                lmeval_metrics = {}
                ece_metrics = {}
                lighteval_file = None
                lighteval_wall = 0.0
                lighteval_metrics = {}
                evalplus_roots = {}
                evalplus_wall = 0.0
                he_metrics = {}
                mbpp_metrics = {}
            else:
                lmeval_start = time.time()
                lmeval_root, task_status, task_durations = run_lmeval_suite(
                    model_dir=model_dir,
                    tag=tag,
                    results_dir=phase_results_dir,
                    gpu_ids=gpu_pool,
                    suite=args.suite,
                    batch_size=args.lm_eval_batch_size,
                    generate_batch_size=args.lm_eval_generate_batch_size,
                    max_batch_size=args.lm_eval_max_batch_size,
                    generate_max_batch_size=args.lm_eval_generate_max_batch_size,
                    fallback_batch_size=args.lm_eval_fallback_batch_size,
                    model_dtype=args.lm_eval_model_dtype,
                    load_in_8bit=args.lm_eval_load_in_8bit,
                    load_in_4bit=args.lm_eval_load_in_4bit,
                    device_map=args.lm_eval_device_map,
                    share_gpu_pool=args.lm_eval_share_gpu_pool,
                    use_fsdp=args.use_fsdp,
                    fsdp_layer_cls=fsdp_layer_cls,
                    fsdp_policy=args.fsdp_policy,
                    phase=phase_name,
                    task_split_overrides=phase["split_overrides"],
                )
                lmeval_wall = time.time() - lmeval_start
                lmeval_metrics = collect_lmeval_metrics(lmeval_root) if lmeval_root else {}
                raw_ece_metrics = collect_ece_from_lmeval_samples(lmeval_root) if lmeval_root else {}
                ece_metrics = _map_ece_to_existing_tasks(raw_ece_metrics, lmeval_metrics)

                lighteval_start = time.time()
                lighteval_file = run_lighteval(model_dir, phase_tag, phase_results_dir, suite=args.suite)
                lighteval_wall = time.time() - lighteval_start if lighteval_file else 0.0
                lighteval_metrics = collect_lighteval_metrics(lighteval_file)

                evalplus_start = time.time()
                evalplus_roots = run_evalplus(model_dir, phase_tag, ["humaneval", "mbpp"], suite=args.suite)
                evalplus_wall = time.time() - evalplus_start if evalplus_roots else 0.0
                he_metrics = collect_evalplus_metrics(evalplus_roots.get("humaneval"), "HumanEval+")
                mbpp_metrics = collect_evalplus_metrics(evalplus_roots.get("mbpp"), "MBPP+")

            if args.only_ifeval or phase.get("ifeval_only"):
                alpaca_wall = 0.0
                alpaca_metrics = {}
                jbb_wall = 0.0
                jbb_metrics = {}
                hb_wall = 0.0
                hb_metrics = {}
            else:
                alpaca_start = time.time()
                alpaca_dir = run_alpacaeval(model_dir, phase_tag, phase_results_dir, suite=args.suite)
                alpaca_wall = time.time() - alpaca_start if alpaca_dir else 0.0
                alpaca_metrics = collect_alpacaeval_metrics(alpaca_dir)

                jbb_start = time.time()
                jbb_dir = run_jailbreakbench(model_dir, phase_tag, phase_results_dir, suite=args.suite)
                jbb_wall = time.time() - jbb_start if jbb_dir else 0.0
                jbb_metrics = collect_simple_json(jbb_dir, "JailbreakBench", "jbb_results.json")

                hb_start = time.time()
                hb_dir = run_harmbench(model_dir, phase_tag, phase_results_dir, suite=args.suite)
                hb_wall = time.time() - hb_start if hb_dir else 0.0
                hb_metrics = collect_simple_json(hb_dir, "HarmBench", "harmbench_results.json")

            # Run deterministic IF-Eval (with proper validation/test splits)
            ifeval_det_metrics: Dict[str, Dict[str, float]] = {}
            ifeval_det_wall = 0.0
            # IF-Eval now supports both validation (20%) and test (80%) phases
            ifeval_start = time.time()
            eval_seed = int(os.environ.get("EVAL_SEED", "42"))
            ifeval_det_metrics, _ = run_ifeval_deterministic(
                model_dir,
                phase_tag,
                phase_results_dir,
                suite=args.suite,
                seed=eval_seed,
                phase=phase_name,
            )
            ifeval_det_wall = time.time() - ifeval_start

            merged = merge_model_results([
                lmeval_metrics,
                ece_metrics,
                lighteval_metrics,
                he_metrics,
                mbpp_metrics,
                ifeval_det_metrics,
                alpaca_metrics,
                jbb_metrics,
                hb_metrics,
            ])
            metrics_by_phase[phase_name][tag] = merged

            total_wall = time.time() - phase_start
            print(f"\n=== Timing summary ({phase_name}) ===")
            print(f"lm-eval total: {lmeval_wall:.1f}s")
            if args.suite == "heavy":
                print(
                    "lighteval: "
                    f"{lighteval_wall:.1f}s, evalplus: {evalplus_wall:.1f}s, "
                    f"ifeval_det: {ifeval_det_wall:.1f}s, alpaca: {alpaca_wall:.1f}s, "
                    f"jbb: {jbb_wall:.1f}s, harmbench: {hb_wall:.1f}s"
                )
            print(f"overall wall ({phase_name}): {total_wall:.1f}s")

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            json_result_file = phase_json_dir / f"eval_{phase_name}_{args.suite}_{tag}_{ts}.json"
            averages = compute_averages(merged)
            method_label = model_display_name or tag

            per_task_ece: Dict[str, float] = {}
            ece_values: List[float] = []
            per_task_perplexity: Dict[str, float] = {}
            perplexity_values: List[float] = []

            for task, metrics in merged.items():
                if not isinstance(metrics, dict):
                    continue
                if isinstance(metrics.get("ece"), (int, float)):
                    ece_val = float(metrics["ece"])
                    per_task_ece[task] = ece_val
                    ece_values.append(ece_val)
                for ppl_key in PERPLEXITY_KEYS:
                    if isinstance(metrics.get(ppl_key), (int, float)):
                        ppl_val = float(metrics[ppl_key])
                        per_task_perplexity[task] = ppl_val
                        perplexity_values.append(ppl_val)
                        break

            avg_ece_val = averages.get("avg_ece")
            if avg_ece_val is None and ece_values:
                avg_ece_val = sum(ece_values) / len(ece_values)

            avg_ppl_val = averages.get("avg_perplexity,none") or averages.get("avg_perplexity")
            if avg_ppl_val is None and perplexity_values:
                avg_ppl_val = sum(perplexity_values) / len(perplexity_values)

            calibration = {
                "avg_ece": avg_ece_val,
                "per_task_ece": per_task_ece,
                "avg_perplexity": avg_ppl_val,
                "per_task_perplexity": per_task_perplexity,
            }

            if avg_ece_val is not None:
                print(f"\n[ECE Summary] ({phase_name}) Average ECE across {len(ece_values)} tasks: {avg_ece_val:.4f}")
                for task, ece in per_task_ece.items():
                    print(f"  {task}: {ece:.4f}")

            if avg_ppl_val is not None:
                print(f"\n[Perplexity Summary] ({phase_name}) Average perplexity across {len(perplexity_values)} tasks: {avg_ppl_val:.2f}")
                for task, ppl in per_task_perplexity.items():
                    print(f"  {task}: {ppl:.2f}")

            timings = {
                "lm_eval_total_sec": lmeval_wall,
                "lm_eval_task_sec": task_durations,
                "lighteval_sec": lighteval_wall,
                "evalplus_sec": evalplus_wall,
                "ifeval_deterministic_sec": ifeval_det_wall,
                "alpaca_sec": alpaca_wall,
                "jailbreakbench_sec": jbb_wall,
                "harmbench_sec": hb_wall,
                "overall_wall_sec": total_wall,
            }
            payload = {
                "tag": tag,
                "suite": args.suite,
                "split": phase_name,
                "results": merged,
                "averages": averages,
                "task_status": task_status,
                "calibration": calibration,
                "timings": timings,
                "display_name": method_label,
            }
            save_json(payload, json_result_file)

            try:
                params_blob: Optional[Dict[str, Any]] = None
                model_path_str = str(model_dir) if not args.from_hf else base_model_dir

                if args.params_json and Path(args.params_json).exists():
                    with open(args.params_json, "r", encoding="utf-8") as f:
                        params_candidate = json.load(f)
                    if isinstance(params_candidate, dict):
                        params_blob = params_candidate.get("params") or params_candidate
                        name_candidate = params_candidate.get("display_name")
                        if isinstance(name_candidate, str) and name_candidate.strip():
                            model_display_name = name_candidate.strip()
                        if isinstance(params_candidate.get("id"), str):
                            params_hash = params_candidate["id"]
                            upsert_eval_results(
                                phase_registry_path,
                                params_hash,
                                args.suite,
                                merged,
                                averages,
                                task_status,
                                model_path=model_path_str,
                                calibration=calibration,
                                display_name=model_display_name,
                            )
                            raise StopIteration
                else:
                    rp = model_dir / "run_params.json"
                    if rp.exists():
                        try:
                            rp_blob = json.load(open(rp))
                            if isinstance(rp_blob, dict):
                                params_blob = rp_blob.get("params") or rp_blob
                                name_candidate = rp_blob.get("display_name")
                                if isinstance(name_candidate, str) and name_candidate.strip():
                                    model_display_name = name_candidate.strip()
                                if isinstance(rp_blob.get("id"), str):
                                    params_hash = rp_blob["id"]
                                    upsert_eval_results(
                                        phase_registry_path,
                                        params_hash,
                                        args.suite,
                                        merged,
                                        averages,
                                        task_status,
                                        model_path=model_path_str,
                                        calibration=calibration,
                                        display_name=model_display_name,
                                    )
                                    raise StopIteration
                        except Exception:
                            pass
                    cfg_path = model_dir / "config.json"
                    if cfg_path.exists():
                        try:
                            cfg_blob = json.load(open(cfg_path))
                            probable = {}
                            for k in [
                                "teacher_model", "student_model", "distill_type", "k_percent",
                                "enable_ce", "alpha_ce", "kd_temperature", "entropy_approx_temperature",
                                "anneal_kd_temperature", "kd_temperature_start", "kd_temperature_end", "kd_hold_frac",
                                "rs_alpha", "rs_floor", "bucket_lower_percent", "bucket_upper_percent",
                                "score_token_selection", "score_normalize", "score_entropy_weight", "score_ce_weight", "score_kl_weight",
                                "bandit_alpha", "bandit_lambda", "bandit_threshold", "bandit_min_tokens", "bandit_max_tokens", "bandit_device", "bandit_reward_clip",
                                "datasets", "prompt_col", "answer_col", "dataset_config", "fineweb_tokens",
                                "epochs", "batch_size", "gradient_accumulation_steps", "max_seq_len", "lr",
                                "seed", "deterministic",
                                "offline_cache", "entropy_approx_m", "rs_vocab_samples", "rs_vocab_beta", "H_hat_u8",
                            ]:
                                if k in cfg_blob:
                                    probable[k] = cfg_blob[k]
                            if probable:
                                params_blob = probable
                        except Exception:
                            pass
                if params_blob is None:
                    params_blob = {}
                params_hash = compute_params_hash(params_blob)
                upsert_eval_results(
                    phase_registry_path,
                    params_hash,
                    args.suite,
                    merged,
                    averages,
                    task_status,
                    model_path=model_path_str,
                    calibration=calibration,
                    display_name=model_display_name,
                )
            except StopIteration:
                pass
            except Exception as e:
                print(f"[registry] Failed to upsert eval results at {phase_registry_path}: {e}")

            if args.suite == "light" and not args.only_ifeval:
                table_path = phase_registry_path.with_name(f"table_{phase_name}.txt")
                append_table_row(table_path, method_label, LIGHT_LMEVAL_TASKS, merged, averages)
            if phase_name == "test" and ifeval_det_metrics:
                if_csv_path = phase_registry_path.with_name("if_table_determi_test.csv")
                if_json_path = phase_registry_path.with_name("if_table_determi_test.json")
                append_table_row_csv_json(
                    csv_path=if_csv_path,
                    json_path=if_json_path,
                    method=method_label,
                    tasks=IF_TABLE_TASKS,
                    merged=ifeval_det_metrics,
                    averages=compute_averages(ifeval_det_metrics),
                )

            if not resolved_display_name and model_display_name:
                resolved_display_name = model_display_name

            try:
                if not args.disable_wandb:
                    if WandBLogger is not None:
                        phase_run = WandBLogger(
                            project=args.wandb_project,
                            entity=os.getenv("WANDB_ENTITY"),
                            name=f"eval-{phase_tag}",
                            config={
                                "suite": args.suite,
                                "split": phase_name,
                                "model": str(base_model_dir) if base_model_dir else str(model_dir),
                            },
                            tags=["evaluation", args.suite, phase_name],
                            group=os.getenv("WANDB_GROUP"),
                            job_type=f"eval-{phase_name}",
                            notes=os.getenv("WANDB_NOTES"),
                            resume=os.getenv("WANDB_RESUME", "allow"),
                            run_id=None,
                        )
                        if getattr(phase_run, "enabled", False):
                            flat = {}
                            for task, metrics in merged.items():
                                if isinstance(metrics, dict):
                                    for k, v in metrics.items():
                                        if isinstance(v, (int, float)):
                                            flat[f"{task}/{k}"] = float(v)
                            for k, v in averages.items():
                                if isinstance(v, (int, float)):
                                    flat[f"avg/{k}"] = float(v)
                            flat["meta/num_tasks"] = float(len(merged))
                            flat["meta/split"] = phase_name
                            phase_run.log(flat)
                            phase_run.finish()
                        else:
                            log_evaluation_to_wandb(phase_tag, merged, args.wandb_project)
                    else:
                        log_evaluation_to_wandb(phase_tag, merged, args.wandb_project)
                if not args.disable_tensorboard:
                    log_evaluation_to_tensorboard(phase_tag, merged, str(work_dir / "tb_logs"))
            except Exception as e:
                print(f"Error logging {phase_tag} metrics: {e}")

    for phase in evaluation_phases:
        phase_name = phase["name"]
        print(f"\n=== Summary table ({phase_name}) ===")
        print_latex_table(metrics_by_phase[phase_name])

if __name__ == "__main__":
    main()
