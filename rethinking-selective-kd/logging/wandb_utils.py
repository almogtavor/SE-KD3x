#!/usr/bin/env python3
"""
Logging Utilities for EKD Project

This module provides utilities for logging training runs and evaluations to both
Weights & Biases and TensorBoard.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING, Tuple, List, Sequence, Literal, cast
from datetime import datetime
import re

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = cast(Any, None)

try:
    from torch.utils.tensorboard.writer import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    SummaryWriter = cast(Any, None)

    if TYPE_CHECKING:
        # For type checkers/linting only; avoids runtime import cycles
        from sampledkd.config import TrainingConfig


WandbResumeSetting = Literal["allow", "never", "must", "auto"]
_VALID_WANDB_RESUME: Tuple[WandbResumeSetting, ...] = ("allow", "never", "must", "auto")


def _coerce_resume(value: Optional[str], fallback: str = "allow") -> WandbResumeSetting:
    if value in _VALID_WANDB_RESUME:
        return cast(WandbResumeSetting, value)
    if fallback in _VALID_WANDB_RESUME:
        return cast(WandbResumeSetting, fallback)
    return "allow"


DEFAULT_TRAIN_STEP_METRIC = "train/global_step"
DEFAULT_TRAIN_METRIC_PATTERNS: Tuple[str, ...] = (
    "train/*",
    "bandit/*",
    "model/*",
    "*",
)


class WandBLogger:
    """W&B logging utility for EKD project."""
    
    def __init__(
        self,
        project: str = "selective-entropy-knowledge-distillation",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[list] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        notes: Optional[str] = None,
        resume: str = "allow",
        run_id: Optional[str] = None,
        display_name: Optional[str] = None,
        default_step_metric: Optional[str] = None,
        step_metric_patterns: Optional[Sequence[str]] = None,
    ):
        # Allow environment to override
        self.project = os.getenv("WANDB_PROJECT", project)
        self.entity  = os.getenv("WANDB_ENTITY", entity or "") or None
        self.group   = os.getenv("WANDB_GROUP", group)
        self.job_type= os.getenv("WANDB_JOB_TYPE", job_type)
        self.notes   = os.getenv("WANDB_NOTES", notes)
        resume_env   = os.getenv("WANDB_RESUME", resume)
        resume_mode  = _coerce_resume(resume_env, resume)
        run_id_env   = os.getenv("WANDB_RUN_ID", run_id)
        self.run = None
        self.display_name = display_name
        self.default_step_metric = default_step_metric
        self.step_metric_patterns = tuple(step_metric_patterns or ())
        self._step_metric_configured = False

        def _is_rank0():
            return os.getenv("RANK") in (None, "0") and os.getenv("LOCAL_RANK") in (None, "0") and os.getenv("SLURM_PROCID") in (None, "0")

        offline = os.getenv("WANDB_MODE", "online") == "offline" or os.getenv("WANDB_DISABLED", "").lower() in ("true", "1")
        self.enabled = WANDB_AVAILABLE and _is_rank0() and not offline

        if self.enabled:
            try:
                # Login if key present; else rely on ~/.netrc
                if os.getenv("WANDB_API_KEY"):
                    wandb.login(key=os.getenv("WANDB_API_KEY"))
                settings = wandb.Settings(start_method=os.getenv("WANDB_START_METHOD", "thread"))
                self.run = wandb.init(
                    project=self.project,
                    entity=self.entity,
                    name=name,
                    config=config or {},
                    tags=tags or [],
                    group=self.group,
                    job_type=self.job_type,
                    notes=self.notes,
                    resume=resume_mode,
                    id=run_id_env,
                    settings=settings,
                    reinit=True,
                )
                print(f"W&B logging initialized: {self.run.get_url()}")
                self._configure_default_step_metric()
                if self.display_name:
                    try:
                        if hasattr(self.run, "config"):
                            self.run.config.update({"display_name": self.display_name}, allow_val_change=True)
                    except Exception as e:
                        print(f"Failed to set W&B config display_name: {e}")
                    try:
                        if hasattr(self.run, "display_name"):
                            self.run.display_name = self.display_name
                        elif hasattr(self.run, "name"):
                            # Fallback: update run name so UI reflects display name.
                            self.run.name = self.display_name
                    except Exception as e:
                        print(f"Failed to set W&B run display_name: {e}")
                    try:
                        self.run.summary["display_name"] = self.display_name
                    except Exception:
                        pass
            except Exception as e:
                print(f"Failed to initialize W&B: {e}")
                self.enabled = False
        else:
            if not WANDB_AVAILABLE:
                print("W&B not available: pip install wandb")
            else:
                print("W&B disabled (non-rank0 or WANDB_MODE=offline/WANDB_DISABLED=true)")

    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        commit: Optional[bool] = None,
    ) -> None:
        """Log metrics to W&B."""
        if self.enabled and self.run:
            try:
                # Prevent logging to past steps to avoid W&B warnings/errors
                if step is not None:
                    current_wandb_step = getattr(self.run, "step", 0)
                    if step < current_wandb_step:
                        return

                # Explicit commit control keeps W&B's internal step counter in sync when
                # we log multiple times for the same training step.
                self.run.log(metrics, step=step, commit=commit)
            except Exception as e:
                print(f"Error logging to W&B: {e}")

    def _configure_default_step_metric(self) -> None:
        if not self.enabled or not self.run or not self.default_step_metric or self._step_metric_configured:
            return
        try:
            wandb.define_metric(self.default_step_metric)
        except Exception as exc:
            print(f"Failed to register W&B step metric '{self.default_step_metric}': {exc}")
            return
        patterns = self.step_metric_patterns or ("*",)
        for pattern in patterns:
            try:
                wandb.define_metric(pattern, step_metric=self.default_step_metric)
            except Exception as exc:
                print(f"Failed to bind W&B metric pattern '{pattern}' to '{self.default_step_metric}': {exc}")
        self._step_metric_configured = True

    def log_artifact(self, artifact_path: str, name: str, type: str, description: str = "") -> None:
        """Log an artifact to W&B."""
        if type.lower() == "model":
            print(f"Skipping W&B upload for model artifact '{name}' to avoid storing weights.")
            return
        if self.enabled and self.run:
            try:
                # W&B artifact name may only contain [A-Za-z0-9_.-]
                safe_name = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
                artifact = wandb.Artifact(name=safe_name, type=type, description=description)
                if os.path.isdir(artifact_path):
                    artifact.add_dir(artifact_path)
                else:
                    artifact.add_file(artifact_path)
                self.run.log_artifact(artifact)
                print(f"Artifact '{safe_name}' logged to W&B")
            except Exception as e:
                print(f"Error logging artifact to W&B: {e}")

    def log_table(self, table_name: str, columns: list, data: list) -> None:
        """Log a table to W&B."""
        if self.enabled and self.run and WANDB_AVAILABLE:
            try:
                table = wandb.Table(columns=columns, data=data)
                self.run.log({table_name: table})
            except Exception as e:
                print(f"Error logging table to W&B: {e}")

    def finish(self) -> None:
        """Finish the W&B run."""
        if self.enabled and self.run:
            try:
                self.run.finish()
                print("W&B run finished successfully")
            except Exception as e:
                print(f"Error finishing W&B run: {e}")


class TensorBoardLogger:
    """TensorBoard logging utility for EKD project."""
    
    def __init__(self, log_dir: str):
        """Initialize TensorBoard logger.
        
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = TENSORBOARD_AVAILABLE
        self.writer = None
        
        if self.enabled:
            try:
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
                print(f"TensorBoard logging enabled at {self.log_dir}")
            except Exception as e:
                print(f"Failed to initialize TensorBoard: {e}")
                self.enabled = False
        else:
            print("TensorBoard not available")
    
    def log_scalar(self, name: str, value: float, step: int = 0) -> None:
        """Log a scalar value to TensorBoard."""
        if self.enabled and self.writer:
            try:
                self.writer.add_scalar(name, value, step)
            except Exception as e:
                print(f"Error logging scalar to TensorBoard: {e}")
    
    def log_scalars(self, metrics: Dict[str, float], step: int = 0) -> None:
        """Log multiple scalar values to TensorBoard."""
        if self.enabled and self.writer:
            try:
                for name, value in metrics.items():
                    self.writer.add_scalar(name, value, step)
            except Exception as e:
                print(f"Error logging scalars to TensorBoard: {e}")
    
    def flush(self) -> None:
        """Flush TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.flush()
            except Exception as e:
                print(f"Error flushing TensorBoard: {e}")
    
    def close(self) -> None:
        """Close TensorBoard writer."""
        if self.enabled and self.writer:
            try:
                self.writer.close()
                print("TensorBoard logger closed successfully")
            except Exception as e:
                print(f"Error closing TensorBoard: {e}")


class CombinedLogger:
    """Combined logger that handles both W&B and TensorBoard logging."""
    
    def __init__(
        self,
        wandb_logger: Optional[WandBLogger] = None,
        tensorboard_logger: Optional[TensorBoardLogger] = None,
    ):
        """Initialize combined logger.
        
        Args:
            wandb_logger: W&B logger instance
            tensorboard_logger: TensorBoard logger instance
        """
        self.wandb_logger = wandb_logger
        self.tensorboard_logger = tensorboard_logger
        self.global_step = 0
        self._pending_step: Optional[int] = None
        self._pending_queue: List[Dict[str, Any]] = []
        self._last_warned_step: Optional[int] = None
    
    def _normalize_step(self, step: Optional[int]) -> int:
        if step is None:
            step = self.global_step
        if step < 0:
            step = 0
        if step > self.global_step:
            self.global_step = step
            self._last_warned_step = None
        elif step < self.global_step:
            if self._last_warned_step != step:
                print(
                    f"[logger] Received non-monotonic step {step} (last={self.global_step}); clamping to {self.global_step}.",
                    flush=True,
                )
                self._last_warned_step = step
            step = self.global_step
        return step

    def _queue_wandb_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        if not self.wandb_logger or not getattr(self.wandb_logger, "enabled", False):
            return
        if self._pending_step is None or step != self._pending_step:
            self._flush_pending_wandb()
            self._pending_step = step
        payload = dict(metrics)
        payload.setdefault(DEFAULT_TRAIN_STEP_METRIC, int(step))
        self._pending_queue.append(payload)

    def _flush_pending_wandb(self) -> None:
        if not self.wandb_logger or not getattr(self.wandb_logger, "enabled", False):
            self._pending_queue.clear()
            self._pending_step = None
            return
        if self._pending_step is None or not self._pending_queue:
            return
        last_idx = len(self._pending_queue) - 1
        for idx, payload in enumerate(self._pending_queue):
            commit = idx == last_idx
            self.wandb_logger.log(payload, self._pending_step, commit=commit)
        self._pending_queue.clear()
        self._pending_step = None

    def log(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log metrics to both W&B and TensorBoard."""
        step = self._normalize_step(step)
        self._queue_wandb_metrics(metrics, step)
        
        if self.tensorboard_logger:
            self.tensorboard_logger.log_scalars(metrics, step)
    
    def log_scalar(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a single scalar to both W&B and TensorBoard."""
        step = self._normalize_step(step)
        self._queue_wandb_metrics({name: value}, step)
        
        if self.tensorboard_logger:
            self.tensorboard_logger.log_scalar(name, value, step)
    
    def log_artifact(self, artifact_path: str, name: str, artifact_type: str = "model") -> None:
        """Log artifact to W&B."""
        self._flush_pending_wandb()
        if artifact_type == "model":
            print(f"Skipping W&B upload for model artifact '{name}' to avoid storing weights.")
            return
        if self.wandb_logger:
            self.wandb_logger.log_artifact(artifact_path, name, artifact_type)
    
    def log_table(self, table_name: str, columns: list, data: list) -> None:
        """Log table to W&B."""
        self._flush_pending_wandb()
        if self.wandb_logger:
            self.wandb_logger.log_table(table_name, columns, data)
    
    def increment_step(self) -> None:
        """Increment the global step counter."""
        self.global_step += 1
    
    def flush(self) -> None:
        """Flush both loggers."""
        self._flush_pending_wandb()
        if self.tensorboard_logger:
            self.tensorboard_logger.flush()
    
    def finish(self) -> None:
        """Finish and close both loggers."""
        self._flush_pending_wandb()
        if self.wandb_logger:
            self.wandb_logger.finish()
        
        if self.tensorboard_logger:
            self.tensorboard_logger.close()


def _format_alpha_suffix(alpha: float) -> str:
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


def create_training_logger(
    config,
    experiment_name: Optional[str] = None,
    display_name: Optional[str] = None,
) -> WandBLogger:
    """Create a W&B logger for training runs."""
    if experiment_name is None:
        current_date = datetime.now().strftime("%Y%m%d_%H%M")
        job_id = os.getenv("SLURM_JOB_ID", "local")
        experiment_name = f"distill-{config.distill_type}-{current_date}_{job_id}"
        if config.distill_type in {"top-k-tok", "random", "random-dkd"}:
            experiment_name += f"_k={config.k_percent}"
        elif config.distill_type == "bucket":
            experiment_name += f"_bucket={config.bucket_lower_percent}-{config.bucket_upper_percent}"
        elif config.distill_type == "pos-rs-kd" and getattr(config, "rs_bucket_mode", False):
            lo = getattr(config, "rs_bucket_lower_percent", None)
            hi = getattr(config, "rs_bucket_upper_percent", None)
            if lo is not None and hi is not None:
                experiment_name += f"_rsbucket={lo}-{hi}"
        suffix_parts = []
        if getattr(config, "distill_category", None) == "on_policy":
            suffix_parts.append("on_policy")
        alpha_value = getattr(config, "alpha_ce", None)
        if alpha_value is not None:
            suffix_parts.append(f"ce{_format_alpha_suffix(alpha_value)}")
        if suffix_parts:
            experiment_name += "_" + "_".join(suffix_parts)

    wandb_config = {
        # Training config
        "teacher_model": config.teacher_model,
        "student_model": config.student_model,
        "distill_type": config.distill_type,
        "k_percent": config.k_percent,
        "bucket_lower_percent": getattr(config, 'bucket_lower_percent', None),
        "bucket_upper_percent": getattr(config, 'bucket_upper_percent', None),
        "rs_bucket_mode": getattr(config, 'rs_bucket_mode', False),
        "rs_bucket_lower_percent": getattr(config, 'rs_bucket_lower_percent', None),
        "rs_bucket_upper_percent": getattr(config, 'rs_bucket_upper_percent', None),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_seq_len": config.max_seq_len,
        "lr": config.lr,
        "datasets": config.datasets,
        # System info
        "job_id": os.getenv("SLURM_JOB_ID", "local"),
        "experiment_name": experiment_name,
    }
    if display_name:
        wandb_config["display_name"] = display_name

    # If training uses FineWeb, log the token budget when available
    try:
        if isinstance(getattr(config, 'datasets', None), list) and 'fineweb' in config.datasets:
            fw_tokens = getattr(config, 'fineweb_tokens', None)
            if fw_tokens is not None:
                wandb_config["fineweb_tokens"] = int(fw_tokens)
    except Exception:
        pass
    
    tags = [
        config.distill_type,
        f"k={config.k_percent}" if config.distill_type != "vanilla" else "vanilla",
        "training"
    ]
    
    # Add mode-specific tags
    if config.distill_type == "top-k-tok":
        tags.append(f"k={config.k_percent}")
    elif config.distill_type == "bucket":
        tags.append(f"bucket={config.bucket_lower_percent}-{config.bucket_upper_percent}")
    elif config.distill_type == "pos-rs-kd" and getattr(config, "rs_bucket_mode", False):
        lo = getattr(config, "rs_bucket_lower_percent", None)
        hi = getattr(config, "rs_bucket_upper_percent", None)
        if lo is not None and hi is not None:
            tags.append(f"rs-bucket={lo}-{hi}")
    elif config.distill_type == "vanilla":
        tags.append("all-tokens")
        
    # Add offline cache tag
    if getattr(config, 'offline_cache', False):
        tags.append("cache")
    
    # Optional tag to surface FineWeb token budget in the UI
    try:
        if isinstance(getattr(config, 'datasets', None), list) and 'fineweb' in config.datasets:
            fw_tokens = getattr(config, 'fineweb_tokens', None)
            if fw_tokens is not None:
                tags.append(f"fineweb_tokens={int(fw_tokens)}")
    except Exception:
        pass

    if getattr(config, "enable_ce_on_all_tokens", False):
        tags.append("ce_all_tokens")

    alpha_ce = getattr(config, "alpha_ce", None)
    if alpha_ce is not None:
        tags.append(f"alpha_ce={alpha_ce}")

    wandb_run_id = os.getenv("WANDB_RUN_ID")
    if wandb_run_id:
        tags.append(f"run_id={wandb_run_id}")
    if getattr(config, "distill_category", None) == "on_policy":
        tags.append("on_policy")
    
    return WandBLogger(
        project=getattr(config, 'wandb_project', 'selective-entropy-knowledge-distillation'),
        entity=getattr(config, 'wandb_entity', None),
        name=experiment_name,
        config=wandb_config,
        tags=tags,
        group=os.getenv("WANDB_GROUP"),
        job_type="train",
        notes=os.getenv("WANDB_NOTES"),
        resume=os.getenv("WANDB_RESUME", "allow"),
        run_id=os.getenv("WANDB_RUN_ID"),
        display_name=display_name,
        default_step_metric=DEFAULT_TRAIN_STEP_METRIC,
        step_metric_patterns=DEFAULT_TRAIN_METRIC_PATTERNS,
    )


def create_evaluation_logger(base_model: str, models_evaluated: list) -> WandBLogger:
    """Create a W&B logger for evaluation runs."""
    experiment_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M')}"
    
    wandb_config = {
        "base_model": base_model,
        "evaluation_date": datetime.now().isoformat(),
        "models_evaluated": models_evaluated,
    }
    
    tags = ["evaluation", "benchmarks"]
    
    return WandBLogger(
        name=experiment_name,
        config=wandb_config,
        tags=tags
    )


def log_evaluation_results(logger: WandBLogger, model_tag: str, results: Dict[str, Dict[str, float]]) -> None:
    """Log evaluation results for a specific model."""
    if not logger.enabled:
        return
        
    try:
        # Flatten metrics for W&B logging
        wandb_metrics = {}
        for suite_name, suite_metrics in results.items():
            for metric_name, value in suite_metrics.items():
                wandb_metrics[f"{model_tag}/{suite_name}/{metric_name}"] = value
        
        # Also log summary metrics
        wandb_metrics[f"{model_tag}/total_metrics_count"] = sum(len(suite) for suite in results.values())
        
        logger.log(wandb_metrics)
        
        # Create a results table for this model
        results_data = [
            [suite_name, metric_name, value]
            for suite_name, suite_metrics in results.items()
            for metric_name, value in suite_metrics.items()
        ]
        
        logger.log_table(
            f"{model_tag}_results_table",
            ["Suite", "Metric", "Value"],
            results_data
        )
        
        print(f"Logged {len(wandb_metrics)} metrics for {model_tag} to W&B")
        
    except Exception as e:
        print(f"Error logging {model_tag} metrics to W&B: {e}")


# Standalone evaluation logging functions (for backward compatibility)
def log_evaluation_to_wandb(tag: str, merged_metrics: Dict[str, Dict[str, float]], project: str) -> None:
    """Log evaluation metrics to W&B (backward-compatible helper).

    This variant respects common env vars, uses a safe start method, and
    flattens only numeric metrics. If W&B is unavailable or disabled via env,
    it prints a diagnostic and returns.
    """
    if not WANDB_AVAILABLE:
        print("W&B not available, skipping wandb logging")
        return
    # Honor env-based disable/offline modes
    offline = os.getenv("WANDB_MODE", "online") == "offline" or os.getenv("WANDB_DISABLED", "").lower() in ("true", "1")
    if offline:
        print("W&B disabled/offline in environment, skipping wandb logging")
        return
    try:
        settings = wandb.Settings(start_method=os.getenv("WANDB_START_METHOD", "thread"))
        resume_mode = _coerce_resume(os.getenv("WANDB_RESUME"), "allow")
        run = wandb.init(
            project=os.getenv("WANDB_PROJECT", project),
            entity=os.getenv("WANDB_ENTITY") or None,
            name=f"eval-{tag}",
            group=os.getenv("WANDB_GROUP"),
            notes=os.getenv("WANDB_NOTES"),
            resume=resume_mode,
            id=os.getenv("WANDB_RUN_ID"),
            settings=settings,
            reinit=True,
        )
        flat: Dict[str, float] = {}
        for task, metrics in merged_metrics.items():
            if not isinstance(metrics, dict):
                continue
            for metric, val in metrics.items():
                if isinstance(val, (int, float)):
                    try:
                        flat[f"{task}/{metric}"] = float(val)
                    except Exception:
                        continue
        if flat:
            run.log(flat)
        run.finish()
        print(f"✓ Logged {len(flat)} metrics to W&B project '{run.project}'")
    except Exception as e:
        print(f"Failed to log to W&B: {e}")


def log_evaluation_to_tensorboard(
    tag: str, 
    merged_metrics: Dict[str, Dict[str, float]], 
    log_dir: str = "tb_logs"
) -> None:
    """Log evaluation metrics to TensorBoard."""
    if not TENSORBOARD_AVAILABLE:
        print("TensorBoard not available, skipping tensorboard logging")
        return
    try:
        tb_path = Path(log_dir) / tag
        tb_path.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_path))
        metric_count = 0
        for task, metrics in merged_metrics.items():
            for metric, val in metrics.items():
                writer.add_scalar(f"{task}/{metric}", val)
                metric_count += 1
        writer.close()
        print(f"✓ Logged {metric_count} metrics to TensorBoard at {tb_path}")
    except Exception as e:
        print(f"Failed to log to TensorBoard: {e}")


def create_training_combined_logger(
    config: "TrainingConfig",
    experiment_name: str,
    tensorboard_dir: Optional[str] = None,
    display_name: Optional[str] = None,
) -> CombinedLogger:
    """Create a combined logger for training with both W&B and TensorBoard.
    
    Args:
        config: Training configuration
        experiment_name: Name of the experiment
        tensorboard_dir: Directory for TensorBoard logs (optional)
        
    Returns:
        CombinedLogger instance
    """
    # Create W&B logger
    wandb_logger = create_training_logger(config, experiment_name, display_name=display_name)
    
    # Create TensorBoard logger
    tensorboard_logger = None
    if tensorboard_dir:
        tb_path = Path(tensorboard_dir) / experiment_name
        tensorboard_logger = TensorBoardLogger(str(tb_path))
    
    return CombinedLogger(wandb_logger, tensorboard_logger)
