from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import struct
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import device
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader, Subset

from ..config import TrainingConfig, TrainingMetrics
from ..training.offline_cache import (
    build_offline_cache_if_needed,
    init_offline_cache_for_trainer,
    decode_ids_probs_from_block,
    TeacherOfflineCache,
)
from ._mixins.token_entropy_logger import TokenEntropyLogger
from ._mixins.amp_oom import AmpOomMixin
from ._mixins.bandit import BanditMixin
from ._mixins.cache import CacheMixin
from ._mixins.checkpoint import CheckpointMixin
from ._mixins.entropy import EntropyMixin
from ._mixins.gls import GLSMixin
from ._mixins.kd_core import KDCoreMixin
from ._mixins.selection_scoring import SelectionScoringMixin
from .atkd import ATKDCacheBundle, compute_atkd_loss
from .ce_estimators import ce_is_estimator
from .kd_temperature_schedule import KDTemperatureSchedule
from .losses.udkd import UDKDLoss
from .oom_setup_helper import OOMSetupHelper
from .forward_batch_runner import ForwardBatchRunner, SkipBatch
from sekd.distill.sample_skipper import FrozenStudentSampleSkipper


class TrainingTimeLimitReached(RuntimeError):
    """Raised when wall-clock training time exceeds the configured limit."""

    def __init__(
        self,
        message: str,
        *,
        checkpoint_path: Optional[Path] = None,
        elapsed_s: float = 0.0,
        limit_s: Optional[float] = None,
    ) -> None:
        super().__init__(message)
        self.checkpoint_path = checkpoint_path
        self.elapsed_s = elapsed_s
        self.limit_s = limit_s


class Distiller(
    AmpOomMixin,
    CheckpointMixin,
    CacheMixin,
    GLSMixin,
    SelectionScoringMixin,
    EntropyMixin,
    KDCoreMixin,
    BanditMixin,
):
    """Main distillation trainer class.
    It takes a teacher and a student model, a tokenizer, and a dataloader, then performs distillation ."""

    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        dataloader,
        config: TrainingConfig,
        teacher_device: device | str = "cuda",
        student_device: device | str = "cuda",
        student_dtype: Optional[torch.dtype] = None,
        logger=None,  # Combined logger for W&B + TensorBoard
    ):
        self.config = config
        self.ddp_world_size = int(getattr(self.config, "ddp_world_size", 1))
        self.ddp_rank = int(getattr(self.config, "ddp_rank", 0))
        self.ddp_local_rank = int(getattr(self.config, "ddp_local_rank", 0))
        self.ddp_enabled = bool(
            self.ddp_world_size > 1 and dist.is_available() and dist.is_initialized()
        )

        self.teacher_rank0_only = bool(getattr(config, "_teacher_rank0_owner", False))
        self.teacher_required = bool(
            getattr(config, "_teacher_required", teacher_model is not None)
        )
        if teacher_model is not None:
            self.teacher = teacher_model.eval()
        else:
            self.teacher = None
            if self.teacher_required and not self.teacher_rank0_only:
                raise ValueError(
                    "Teacher model is required but unavailable for this rank."
                )
        self.student = student_model.to(student_device).train()
        teacher_device_str = getattr(config, "_teacher_device_str", None)
        if teacher_device_str is not None:
            self.teacher_device = torch.device(teacher_device_str)
        else:
            self.teacher_device = teacher_device
        self.teacher_available = self.teacher_required and (
            self.teacher is not None or self.teacher_rank0_only
        )
        self.tok = tokenizer
        self.dataloader = dataloader

        self.student_device = student_device
        self.student_dtype = student_dtype or self._detect_student_dtype()
        self.opt = cast(Optimizer, None)

        self._wrap_student_for_ddp()
        OOMSetupHelper.configure(self, config)
        if not isinstance(self.opt, Optimizer):
            raise RuntimeError("Optimizer failed to initialize during OOM setup.")

        # Logging setup
        self.logger = logger
        interval = getattr(self.config, "debug_log_interval", 100)
        try:
            interval = int(interval)
        except (TypeError, ValueError):
            interval = 20
        self._debug_log_interval = interval if interval > 0 else 0
        self._debug_forward_calls = 0
        self._kl_zero_counter = 0
        self.global_step = 0
        self._printed_cache_info = False
        self._total_train_tokens = 0
        self._cache_build_on_miss_done = False
        self._total_train_iterations = 0
        self._peak_memory_samples: List[
            float
        ] = []  # per-step peak memory (GB) when log_peak_memory is enabled

        # GLS ring buffer state (inactive unless enabled)
        self._gls_buf: Optional[torch.Tensor] = None
        self._gls_cap: int = int(getattr(self.config, "gls_queue_size", 30000))
        self._gls_count: int = 0
        self._gls_head: int = 0

        # Checkpointing
        self.output_dir = Path(config.output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._resume_epoch = 0
        self._resume_step = 0
        self._resume_checkpoint_path: Optional[Path] = None
        resume_path = getattr(self.config, "resume_from_checkpoint", None)
        if resume_path:
            ckpt_path = Path(str(resume_path))
            if not ckpt_path.is_absolute():
                ckpt_path = (self.output_dir / ckpt_path).resolve()
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
            epoch, step = self.load_checkpoint(str(ckpt_path))
            self._resume_epoch = int(epoch)
            self._resume_step = int(step)
            self._resume_checkpoint_path = ckpt_path

        # Optional entropy dump for token selection analysis (all distill types)
        dump_enabled = bool(getattr(self.config, "pos_rs_entropy_dump_enabled", True))
        dump_limit = max(0, int(getattr(self.config, "pos_rs_entropy_dump_limit", 5)))
        dump_path_cfg = getattr(self.config, "pos_rs_entropy_dump_path", None)
        distill_type = getattr(self.config, "distill_type", "unknown")
        dump_path: Optional[Path]
        if dump_path_cfg is not None:
            tmp_path = Path(dump_path_cfg)
            dump_path = (
                tmp_path if tmp_path.is_absolute() else self.output_dir / tmp_path
            )
        else:
            # Name file after distill type for clarity
            safe_name = distill_type.replace("-", "_")
            dump_path = self.output_dir / f"{safe_name}_entropy_dump.jsonl"
        if dump_enabled and dump_limit > 0:
            self._pos_rs_entropy_logger = TokenEntropyLogger(
                enabled=True,
                dump_path=dump_path,
                limit=dump_limit,
                tokenizer=self.tok,
                ddp_rank=self.ddp_rank,
            )
        else:
            self._pos_rs_entropy_logger = None

        # Optional dedicated debug dump for top-k token selection analysis
        topk_dbg_limit = max(0, int(getattr(self.config, "topk_debug_dump_limit", 0)))
        self._topk_debug_limit = topk_dbg_limit if self.ddp_rank == 0 else 0
        self._topk_debug_written = 0
        self._topk_debug_path: Optional[Path] = None
        if self._topk_debug_limit > 0:
            cfg_path = getattr(self.config, "topk_debug_dump_path", None)
            if cfg_path:
                tmp_path = Path(cfg_path)
                debug_path = (
                    tmp_path if tmp_path.is_absolute() else self.output_dir / tmp_path
                )
            else:
                debug_path = self.output_dir / "topk_selection_debug.jsonl"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            debug_path.write_text("", encoding="utf-8")
            self._topk_debug_path = debug_path

        # offline teacher logits cache: centralized initialization
        self.cache = None

        # Track once-per-run warnings
        self._warned_invalid_targets = False
        self._warned_invalid_logprob = False
        self._last_ce_selection_mask: Optional[torch.Tensor] = None

        # Entropy curriculum settings (optional)
        self._selection_curriculum_active = bool(
            getattr(self.config, "selection_curriculum", False)
        )
        steps = int(getattr(self.config, "selection_curriculum_steps", 2000))
        self._selection_curriculum_steps = max(1, steps)
        start_frac = float(getattr(self.config, "selection_curriculum_start", 0.0))
        end_frac = float(getattr(self.config, "selection_curriculum_end", 1.0))
        start_frac = min(max(start_frac, 0.0), 1.0)
        end_frac = min(max(end_frac, 0.0), 1.0)
        self._selection_curriculum_start = start_frac
        self._selection_curriculum_end = end_frac
        self._selection_curriculum_power = max(
            0.0, float(getattr(self.config, "selection_curriculum_power", 1.0))
        )

        # LinUCB contextual bandit setup
        self._init_bandit_manager()

        # Optional sample skipping (computed once from frozen student pre-pass)
        self.sample_skipper: Optional[FrozenStudentSampleSkipper] = None
        if bool(getattr(self.config, "skip_by_frozen_student", False)):
            self.sample_skipper = FrozenStudentSampleSkipper(
                config=self.config,
                student=self.student,
                student_device=torch.device(self.student_device),
                dataloader=self.dataloader,
                teacher=self.teacher,
                teacher_device=torch.device(self.teacher_device)
                if self.teacher_device is not None
                else None,
            )

    def _wrap_student_for_ddp(self) -> None:
        if self.ddp_enabled:
            device_ids = [self.ddp_local_rank]
            self.student = torch.nn.parallel.DistributedDataParallel(
                self.student,
                device_ids=device_ids,
                output_device=device_ids[0],
                find_unused_parameters=False,
            )

    @property
    def _student_base(self):
        if hasattr(self.student, "module"):
            return self.student.module
        return self.student

    def _count_batch_tokens(self, batch: Dict[str, Any]) -> int:
        labels = batch.get("labels")
        if not torch.is_tensor(labels) or labels.dim() < 2:
            return 0
        shift_labels = labels[:, 1:]
        valid = shift_labels.ne(-100)
        try:
            return int(valid.sum().item())
        except Exception:
            return 0

    def _param_count(self, model) -> int:
        if model is None:
            return 0
        counter = getattr(model, "num_parameters", None)
        if callable(counter):
            try:
                return int(counter())
            except Exception:
                pass
        try:
            return sum(int(p.numel()) for p in model.parameters())
        except Exception:
            return 0

    @staticmethod
    def _hash_selected_ids(selected_ids: List[int]) -> str:
        h = hashlib.sha1()
        for sid in selected_ids:
            try:
                h.update(struct.pack("<I", int(sid)))
            except Exception:
                h.update(str(sid).encode("utf-8"))
        return h.hexdigest()

    def _kl_directional(
        self, log_p_teacher: torch.Tensor, log_q_student: torch.Tensor
    ) -> torch.Tensor:
        objective = getattr(self.config, "kd_objective", "forward")
        if objective == "reverse":
            kl = self._kl_loss(log_q_student, log_p_teacher)
        else:
            kl = self._kl_loss(log_p_teacher, log_q_student)
        return kl.to(log_q_student.device)

    def _compute_dkd_rows_components(
        self,
        t_probs: torch.Tensor,
        s_probs: torch.Tensor,
        targets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DKD total + (TCKD, NCKD) per position.

        Returns:
            total_rows: [N]
            tckd_rows:  [N]
            nckd_rows:  [N]
        """
        t_pt = t_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        s_pt = s_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        t_rest = (t_probs.sum(dim=-1) - t_pt).clamp_min(1e-12)
        s_rest = (s_probs.sum(dim=-1) - s_pt).clamp_min(1e-12)

        t_bin = torch.stack([t_pt, t_rest], dim=-1)
        s_bin = torch.stack([s_pt, s_rest], dim=-1)
        t_bin = t_bin / t_bin.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        s_bin = s_bin / s_bin.sum(dim=-1, keepdim=True).clamp_min(1e-12)

        log_t_bin = torch.log(t_bin.clamp_min(1e-12))
        log_s_bin = torch.log(s_bin.clamp_min(1e-12))
        tckd = self._kl_directional(log_t_bin, log_s_bin)

        t_off = t_probs.clone()
        s_off = s_probs.clone()
        target_idx = targets.unsqueeze(-1)
        t_off.scatter_(1, target_idx, 0.0)
        s_off.scatter_(1, target_idx, 0.0)
        t_off_sum = t_off.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        s_off_sum = s_off.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        t_hat = t_off / t_off_sum
        s_hat = s_off / s_off_sum
        log_t_hat = torch.log(t_hat.clamp_min(1e-12))
        log_s_hat = torch.log(s_hat.clamp_min(1e-12))
        nckd = self._kl_directional(log_t_hat, log_s_hat)

        alpha = float(getattr(self.config, "dkd_alpha", 1.0))
        beta = float(getattr(self.config, "dkd_beta", 8.0))
        total = alpha * tckd + beta * nckd
        return total, tckd, nckd

    def _compute_dkd_total_rows(
        self, t_probs: torch.Tensor, s_probs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute DKD total loss per position given teacher/student probs and target ids."""
        total, _, _ = self._compute_dkd_rows_components(t_probs, s_probs, targets)
        return total

    def _compute_dkd_rows(
        self, t_probs: torch.Tensor, s_probs: torch.Tensor, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Backward-compatible DKD API.

        Historically some call sites expected (total, tckd, nckd). Returning a 3-tuple
        avoids "too many values to unpack" regressions when implementations evolve.
        """
        return self._compute_dkd_rows_components(t_probs, s_probs, targets)

    def _rs_bucket_bounds(self) -> Optional[Tuple[float, float]]:
        """Return (lower_q, upper_q) percentiles if RS bucketting is active."""
        if not bool(getattr(self.config, "rs_bucket_mode", False)):
            return None
        lower = getattr(self.config, "rs_bucket_lower_percent", None)
        upper = getattr(self.config, "rs_bucket_upper_percent", None)
        if lower is None or upper is None:
            return None
        try:
            lower_f = float(lower)
            upper_f = float(upper)
        except (TypeError, ValueError):
            return None
        if not (0.0 <= lower_f < upper_f <= 100.0):
            return None
        return lower_f / 100.0, upper_f / 100.0

    @staticmethod
    def _apply_rs_bucket_filter(
        vec: torch.Tensor,
        idx: torch.Tensor,
        bounds: Optional[Tuple[float, float]],
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Restrict candidate tensors to the requested percentile band."""
        if vec.numel() == 0 or bounds is None:
            return vec, idx, False
        lower_q, upper_q = bounds
        if lower_q is None or upper_q is None:
            return vec, idx, False
        if vec.numel() == 1:
            return vec, idx, False
        low = torch.quantile(vec, lower_q)
        high = torch.quantile(vec, upper_q)
        mask = (vec >= low) & (vec <= high)
        if not mask.any():
            return vec, idx, False
        return vec[mask], idx[mask], True

    def _detect_student_dtype(self) -> torch.dtype:
        """Infer the student's numeric precision so downstream casts stay consistent."""
        base = self._student_base
        try:
            param = next(base.parameters())
            return param.dtype
        except StopIteration:
            buffer = next(base.buffers(), None)
            if buffer is not None:
                return buffer.dtype
        # Fallback when the model has no parameters/buffers yet
        return torch.float32

    def _selection_curriculum_phase(self) -> float:
        """Return curriculum progress mapped into [0, 1], accounting for start/end targets."""
        if not self._selection_curriculum_active:
            return 1.0
        steps = self._selection_curriculum_steps
        if steps <= 0:
            progress = 1.0
        else:
            progress = min(1.0, max(0.0, self.global_step / float(steps)))
        power = self._selection_curriculum_power
        if power > 0.0:
            progress = progress**power
        start = self._selection_curriculum_start
        end = self._selection_curriculum_end
        if end < start:
            start, end = end, start
        phase = start + (end - start) * progress
        return max(0.0, min(1.0, phase))

    def _selection_curriculum_mask(
        self,
        ent_row: torch.Tensor,
        valid_mask: torch.Tensor,
        quota: int,
    ) -> torch.Tensor:
        """Return boolean mask selecting a contiguous entropy window of size `quota`."""
        if not self._selection_curriculum_active:
            return valid_mask.clone()
        mask_out = valid_mask.clone()
        n_valid = int(valid_mask.sum().item())
        if n_valid == 0 or quota <= 0:
            return mask_out
        quota = min(quota, n_valid)
        ent_valid = ent_row[valid_mask]
        if ent_valid.numel() == 0:
            return mask_out
        phase = self._selection_curriculum_phase()
        range_max = max(0, ent_valid.numel() - quota)
        start_idx = int(math.floor(range_max * phase + 1e-8))
        start_idx = min(max(0, start_idx), range_max)
        sorted_rel = torch.argsort(ent_valid, dim=0, descending=False)
        end_idx = start_idx + quota
        chosen_rel = sorted_rel[start_idx:end_idx]
        if chosen_rel.numel() == 0:
            return mask_out
        valid_indices = torch.where(valid_mask)[0]
        mask_out = torch.zeros_like(valid_mask, dtype=torch.bool)
        mask_out[valid_indices[chosen_rel]] = True
        return mask_out

    def _selection_curriculum_sampling_weights(
        self, ent_values: torch.Tensor
    ) -> torch.Tensor:
        """Return sampling weights that interpolate from low-entropy preference to high-entropy preference."""
        if ent_values.numel() == 0:
            return torch.ones_like(ent_values, dtype=torch.float32)
        ent_float = ent_values.float()
        finite_mask = torch.isfinite(ent_float)
        if not finite_mask.all():
            if finite_mask.any():
                fill_value = ent_float[finite_mask].mean()
            else:
                fill_value = ent_float.new_zeros(())
            ent_float = ent_float.masked_fill(~finite_mask, fill_value)
        min_val = ent_float.min()
        max_val = ent_float.max()
        span = (max_val - min_val).clamp_min(1e-6)
        norm = (ent_float - min_val) / span
        phase = self._selection_curriculum_phase()
        low = (1.0 - norm).clamp_min(1e-6)
        high = norm.clamp_min(1e-6)
        weights = low * (1.0 - phase) + high * phase
        return weights.clamp_min(1e-6)

    def _teacher_forward_logits(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.teacher_available:
            raise RuntimeError(
                "Teacher logits requested but no teacher is available on this rank."
            )
        if self.teacher is not None and not self.teacher_rank0_only:
            from torch.cuda.amp import autocast

            input_ids_t = input_ids.to(self.teacher_device)
            attn_t = attn_mask.to(self.teacher_device)
            with torch.no_grad():
                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    logits = self.teacher(
                        input_ids_t,
                        attention_mask=attn_t,
                        output_hidden_states=False,
                    ).logits
            return self._sanitize_logits(logits, "teacher")
        return self._teacher_forward_distributed(
            input_ids, attn_mask, amp_enabled, amp_dtype
        )

    def _teacher_forward_distributed(
        self,
        input_ids: torch.Tensor,
        attn_mask: torch.Tensor,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
    ) -> torch.Tensor:
        if not (self.teacher_rank0_only and self.ddp_enabled and dist.is_initialized()):
            raise RuntimeError(
                "Distributed teacher forwarding requested without active rank-0 ownership."
            )

        payload = (input_ids.cpu(), attn_mask.cpu())
        gather_list: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]]
        if self.ddp_rank == 0:
            gather_list = cast(
                List[Optional[Tuple[torch.Tensor, torch.Tensor]]],
                [None] * self.ddp_world_size,
            )
        else:
            gather_list = None
        dist.gather_object(payload, gather_list, dst=0)

        outputs: Optional[List[Optional[torch.Tensor]]] = None
        if self.ddp_rank == 0:
            if self.teacher is None:
                raise RuntimeError(
                    "Teacher model is unavailable for distributed forward."
                )
            assert gather_list is not None
            outputs = []
            from torch.cuda.amp import autocast

            with torch.no_grad():
                for item in gather_list:
                    if item is None:
                        outputs.append(None)
                        continue
                    ids_cpu, mask_cpu = item
                    ids = ids_cpu.to(self.teacher_device, non_blocking=True)
                    mask = mask_cpu.to(self.teacher_device, non_blocking=True)
                    with autocast(enabled=amp_enabled, dtype=amp_dtype):
                        logits = self.teacher(
                            ids,
                            attention_mask=mask,
                            output_hidden_states=False,
                        ).logits
                    outputs.append(
                        self._sanitize_logits(logits, "teacher").detach().cpu()
                    )

        recv = [None]
        dist.scatter_object_list(recv, outputs if self.ddp_rank == 0 else None, src=0)
        logits_cpu = recv[0]
        if logits_cpu is None:
            raise RuntimeError(
                "Distributed teacher failed to return logits for current rank."
            )
        return logits_cpu.to(self.student_device)

    def _log_token_selection(
        self,
        keep_mask: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        ent_raw: Optional[torch.Tensor] = None,
        selection_policy: Optional[str] = None,
    ) -> None:
        """Log selected tokens for analysis (all distill types)."""
        logger = self._pos_rs_entropy_logger
        if logger is None or not logger.enabled:
            return
        logger.record_selection(
            keep_mask=keep_mask,
            valid_next=valid_next,
            input_ids=input_ids,
            entropies=ent_raw,
            global_step=self.global_step,
            k_percent=float(getattr(self.config, "k_percent", 0.0)),
            distill_type=str(getattr(self.config, "distill_type", "unknown")),
            selection_policy=selection_policy,
        )

    def _log_selected_positions(
        self,
        selected_positions: List[Tuple[int, int]],
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        entropies: Optional[torch.Tensor],
        selection_policy: Optional[str],
    ) -> None:
        """Helper to log selection info when only (batch, pos) pairs are available."""
        logger = self._pos_rs_entropy_logger
        if logger is None or not logger.enabled:
            return
        if not selected_positions:
            return
        keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
        max_batch, max_pos = keep_mask.size(0), keep_mask.size(1)
        for b_idx, t_idx in selected_positions:
            if 0 <= b_idx < max_batch and 0 <= t_idx < max_pos:
                keep_mask[b_idx, t_idx] = True
        self._log_token_selection(
            keep_mask, valid_next, input_ids, entropies, selection_policy
        )

    def _log_topk_debug_info(
        self,
        *,
        keep_mask: torch.Tensor,
        valid_next: torch.Tensor,
        stat: Optional[torch.Tensor],
        ent_raw: Optional[torch.Tensor],
        selection_policy: Optional[str],
        selection_metric: str,
    ) -> None:
        """Record per-document ranking info to help debug top-k token selection."""
        if (
            self._topk_debug_path is None
            or self._topk_debug_written >= self._topk_debug_limit
        ):
            return
        if stat is None:
            return
        for i in range(keep_mask.size(0)):
            if self._topk_debug_written >= self._topk_debug_limit:
                break
            selected_idx = torch.where(keep_mask[i])[0]
            if selected_idx.numel() == 0:
                continue
            valid_mask = valid_next[i]
            valid_idx = torch.where(valid_mask)[0]
            if valid_idx.numel() == 0:
                continue
            metric_values = stat[i].detach().float()
            metric_valid = metric_values[valid_mask]
            if metric_valid.numel() == 0:
                continue
            top_k = min(int(selected_idx.numel()), int(metric_valid.numel()))
            if top_k <= 0:
                continue
            top_vals, top_rel_idx = torch.topk(
                metric_valid, k=top_k, largest=True, sorted=True
            )
            top_positions = valid_idx[top_rel_idx]
            sel_vals = metric_values[selected_idx].detach().float()
            record: Dict[str, Any] = {
                "doc_id": int(self._topk_debug_written),
                "selection_policy": selection_policy,
                "selection_metric": selection_metric,
                "k_percent": float(getattr(self.config, "k_percent", 0.0)),
                "selected_count": int(selected_idx.numel()),
                "valid_count": int(valid_idx.numel()),
                "selected_indices": [int(x) for x in selected_idx.cpu().tolist()],
                "selected_metric_values": sel_vals.cpu().tolist(),
                "top_indices_by_metric": [int(x) for x in top_positions.cpu().tolist()],
                "top_metric_values": top_vals.cpu().tolist(),
            }
            if ent_raw is not None:
                ent_sel = ent_raw[i][selected_idx].detach().float()
                record["teacher_entropy_for_selected"] = ent_sel.cpu().tolist()
            with self._topk_debug_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")
            self._topk_debug_written += 1

    def _forward_batch(self, batch):
        return ForwardBatchRunner(self, batch).run()

    def _compute_kd_loss(
        self,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        """Compute knowledge distillation loss for the configured strategy.

        Args:
            t_pred: Teacher predictions [B, L-1, V]
            t_log_probs: Teacher log probabilities [B, L-1, V]
            s_pred: Student logits aligned to next-token positions [B, L-1, V]
            s_log_probs: Student log probabilities at KD temperature [B, L-1, V]
            valid_next: Boolean mask of valid next-token positions [B, L-1]
            input_ids: Input ids [B, L]
            attention_mask: Attention mask [B, L]
            temperature: KD temperature used for the current batch
            debug_log: Whether to emit verbose debug logging for this batch.

        Returns:
            Tuple containing the KD loss tensor and optional auxiliary info (for LinUCB).
        """
        extra: Optional[Dict[str, Any]] = None
        kd_loss = s_pred.sum() * 0.0
        self._last_ce_selection_mask = None

        distill_type = self.config.distill_type

        if distill_type == "atkd":
            raise RuntimeError("AT-KD loss should be computed via compute_atkd_loss.")

        if distill_type == "vanilla":
            return self._kd_loss_vanilla(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type == "dkd":
            return self._kd_loss_dkd(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type in {"top-k-tok", "top-k-tok-dkd"}:
            return self._kd_loss_top_k_tok(
                distill_type=distill_type,
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type == "bucket":
            return self._kd_loss_bucket(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type in {"random", "random-dkd"}:
            return self._kd_loss_random(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type == "atkd":
            return self._kd_loss_atkd(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type == "pos-rs-kd":
            return self._kd_loss_pos_rs_kd(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type == "pos-rs-kd-dkd":
            return self._kd_loss_pos_rs_kd_dkd(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )
        if distill_type in {"linucb", "linucb-dkd"}:
            return self._kd_loss_linucb(
                t_pred=t_pred,
                t_log_probs=t_log_probs,
                s_pred=s_pred,
                s_log_probs=s_log_probs,
                valid_next=valid_next,
                input_ids=input_ids,
                attention_mask=attention_mask,
                temperature=temperature,
                debug_log=debug_log,
            )

        return kd_loss, extra

    def _kd_loss_vanilla(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        kl_pos = self._kl_directional(t_log_probs, s_log_probs)
        denom = valid_next.sum().clamp(min=1)
        kd_loss = (kl_pos * valid_next).sum() / denom
        return kd_loss, None

    def _kd_loss_dkd(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if t_log_probs is None or t_pred is None:
            raise RuntimeError(
                "DKD requires teacher logits; disable offline cache or ensure teacher is available."
            )
        mask = valid_next
        if mask.sum().item() == 0:
            kd_loss = s_pred.sum() * 0.0
        else:
            targets = input_ids[:, 1:].to(self.student_device)
            flat_targets = targets[mask].long()

            t_log_masked = t_log_probs.to(self.student_device)[mask]
            s_log_masked = s_log_probs[mask]
            t_probs = t_log_masked.exp()
            s_probs = s_log_masked.exp()

            kd_rows = self._compute_dkd_total_rows(t_probs, s_probs, flat_targets)
            if kd_rows.numel() == 0:
                kd_loss = s_pred.sum() * 0.0
            else:
                kd_loss = kd_rows.mean()
        return kd_loss, None

    def _kd_loss_top_k_tok(
        self,
        *,
        distill_type: str,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # top-k% tokens among valid positions; optionally global threshold via GLS
        # Select positions FIRST, then compute KL only on selected rows (efficient)
        if debug_log:
            print(
                f"[DEBUG] top-k-tok START: t_pred shape={t_pred.shape if t_pred is not None else 'None'}, valid_next.sum()={valid_next.sum().item()}",
                flush=True,
            )
            if t_pred is not None:
                print(
                    f"[DEBUG] t_pred stats: min={t_pred.min().item():.6f}, max={t_pred.max().item():.6f}, mean={t_pred.mean().item():.6f}",
                    flush=True,
                )
        pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
        normalize_topk = bool(getattr(self.config, "normalize_topk_by_length", False))
        score_enabled = bool(getattr(self.config, "score_token_selection", False))
        gls_effective = (
            bool(getattr(self.config, "gls_enabled", False)) and not normalize_topk
        )
        policy_bits = [f"k%={float(getattr(self.config, 'k_percent', 0.0)):.2f}"]
        policy_bits.append(f"normalize={'on' if normalize_topk else 'off'}")
        policy_bits.append(f"gls={'on' if gls_effective else 'off'}")
        policy_bits.append(f"score={'on' if score_enabled else 'off'}")
        selection_policy = f"{distill_type} | " + ", ".join(policy_bits)

        skip_entropy = (
            pct >= 0.9995
            and not normalize_topk
            and not gls_effective
            and not score_enabled
        )

        sel_gls_count = 0
        ent_raw: Optional[torch.Tensor] = None
        stat: Optional[torch.Tensor] = None
        selection_metric = str(
            getattr(self.config, "topk_tok_selection_metric", "teacher_entropy")
        )
        if skip_entropy:
            keep_mask = valid_next.clone()
            sel_topk_count = int(valid_next.sum().item())
        else:
            if selection_metric == "student_entropy":
                from sekd.training.entropy_utils import token_entropy

                # Student logits are already aligned to next-token positions: [B, L-1, V]
                ent_raw = token_entropy(s_pred.to(self.student_device))
            elif selection_metric == "student_ce":
                if s_log_probs is None:
                    raise ValueError(
                        "student_ce selection metric requires student log probabilities."
                    )
                targets = input_ids[:, 1:].to(self.student_device)
                vocab_size = s_log_probs.size(-1)
                invalid_mask = (targets < 0) | (targets >= vocab_size)
                invalid_count = int(invalid_mask.sum().item())
                if invalid_count > 0:
                    if not getattr(self, "_warned_selection_invalid_targets", False):
                        bad_vals = (
                            targets[invalid_mask].detach().to("cpu", non_blocking=True)
                        )
                        min_bad = int(bad_vals.min().item()) if bad_vals.numel() else 0
                        max_bad = int(bad_vals.max().item()) if bad_vals.numel() else 0
                        sample_vals = bad_vals.unique()
                        if sample_vals.numel() > 5:
                            sample_vals = sample_vals[:5]
                        sample_list = ", ".join(str(int(v.item())) for v in sample_vals)
                        print(
                            "[warn] Student-CE selection targets out of range "
                            f"(count={invalid_count}, min={min_bad}, max={max_bad}, sample=[{sample_list}]) "
                            "→ skipping those positions.",
                            flush=True,
                        )
                        self._warned_selection_invalid_targets = True
                    valid_next = valid_next & (~invalid_mask)
                    targets = targets.masked_fill(invalid_mask, 0)
                targets = targets.masked_fill(~valid_next, 0)
                targets = targets.clamp(min=0, max=vocab_size - 1)
                target_logp = s_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                ent_raw = (-target_logp).detach()
            elif selection_metric in (
                "ce_ratio",
                "ce_ratio_entropy",
                "ce_ratio_plus_entropy",
            ):
                # CE ratio: CE_s(t) / (CE_t(t) + epsilon)
                # CE ratio × entropy: CE_s(t) / (CE_t(t) + epsilon) * H_s(t)
                # CE ratio + entropy: CE_s(t) / (CE_t(t) + epsilon) + H_s(t)
                if s_log_probs is None:
                    raise ValueError(
                        f"{selection_metric} selection metric requires student log probabilities."
                    )
                if t_log_probs is None:
                    raise ValueError(
                        f"{selection_metric} selection metric requires teacher log probabilities."
                    )
                targets = input_ids[:, 1:].to(self.student_device)
                vocab_size = s_log_probs.size(-1)
                invalid_mask = (targets < 0) | (targets >= vocab_size)
                invalid_count = int(invalid_mask.sum().item())
                if invalid_count > 0:
                    if not getattr(self, "_warned_selection_invalid_targets", False):
                        bad_vals = (
                            targets[invalid_mask].detach().to("cpu", non_blocking=True)
                        )
                        min_bad = int(bad_vals.min().item()) if bad_vals.numel() else 0
                        max_bad = int(bad_vals.max().item()) if bad_vals.numel() else 0
                        sample_vals = bad_vals.unique()
                        if sample_vals.numel() > 5:
                            sample_vals = sample_vals[:5]
                        sample_list = ", ".join(str(int(v.item())) for v in sample_vals)
                        print(
                            f"[warn] {selection_metric} selection targets out of range "
                            f"(count={invalid_count}, min={min_bad}, max={max_bad}, sample=[{sample_list}]) "
                            "→ skipping those positions.",
                            flush=True,
                        )
                        self._warned_selection_invalid_targets = True
                    valid_next = valid_next & (~invalid_mask)
                    targets = targets.masked_fill(invalid_mask, 0)
                targets = targets.masked_fill(~valid_next, 0)
                targets = targets.clamp(min=0, max=vocab_size - 1)
                # Student CE: -log p_s(target)
                student_target_logp = s_log_probs.gather(
                    -1, targets.unsqueeze(-1)
                ).squeeze(-1)
                student_ce = (-student_target_logp).detach()
                # Teacher CE: -log p_t(target)
                targets_t = targets.to(t_log_probs.device, non_blocking=True)
                teacher_target_logp = t_log_probs.gather(
                    -1, targets_t.unsqueeze(-1)
                ).squeeze(-1)
                teacher_ce = (-teacher_target_logp).to(self.student_device).detach()
                # CE ratio with epsilon for numerical stability
                eps = 1e-6
                ce_ratio = student_ce / (teacher_ce + eps)
                if selection_metric in ("ce_ratio_entropy", "ce_ratio_plus_entropy"):
                    # Normalized student entropy: H_s(t) / log(|V|)
                    from sekd.training.entropy_utils import token_entropy

                    student_entropy = token_entropy(s_pred.to(self.student_device))
                    log_vocab = float(math.log(vocab_size))
                    student_entropy_norm = student_entropy / log_vocab
                    if selection_metric == "ce_ratio_entropy":
                        ent_raw = (ce_ratio * student_entropy_norm).detach()
                    else:
                        ent_raw = (ce_ratio + student_entropy_norm).detach()
                else:
                    ent_raw = ce_ratio
            else:
                # Default: teacher entropy (online logits or offline cache)
                ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            if debug_log:
                print(
                    f"[DEBUG] ent_raw shape={ent_raw.shape}, valid_next.shape={valid_next.shape}",
                    flush=True,
                )
                valid_ent = ent_raw[valid_next]
                print(
                    f"[DEBUG] ent_raw[valid_next] stats: min={valid_ent.min().item():.6f}, max={valid_ent.max().item():.6f}, mean={valid_ent.mean().item():.6f}, count={valid_ent.numel()}",
                    flush=True,
                )

            if selection_metric == "kl":
                # Rank by per-position KL between teacher and student distributions.
                # (KL direction follows kd_objective via _kl_directional)
                t_log_all = t_log_probs.to(self.student_device)
                kl_pos = self._kl_directional(t_log_all, s_log_probs)
                stat = kl_pos.masked_fill(~valid_next, float("-inf"))
            elif selection_metric == "reverse-kl":
                # Rank by reverse KL (student||teacher) regardless of global kd_objective.
                t_log_all = t_log_probs.to(self.student_device)
                kl_pos = self._kl_loss(s_log_probs, t_log_all)
                stat = kl_pos.masked_fill(~valid_next, float("-inf"))
            elif score_enabled:
                # === Two-stage selection for efficiency ===
                # Stage 1: Prefilter by entropy (cheap, no softmax needed)
                prefilter_mult = float(
                    getattr(self.config, "score_prefilter_multiplier", 3.0)
                )
                # Stage 2: Compute KL/CE only on prefiltered subset

                stat = torch.full_like(ent_raw, float("-inf"))
                for i in range(valid_next.size(0)):
                    mask_i = valid_next[i]
                    n_valid = int(mask_i.sum().item())
                    if n_valid < 3:
                        continue

                    # Stage 1: Select top (k * prefilter_mult) by entropy
                    k_final = max(1, min(n_valid, math.ceil(pct * n_valid)))
                    k_pre = min(n_valid, max(k_final, int(k_final * prefilter_mult)))

                    valid_idx = torch.where(mask_i)[0]
                    ent_valid = ent_raw[i][mask_i]
                    _, pre_rel_idx = torch.topk(
                        ent_valid, k=k_pre, largest=True, sorted=False
                    )
                    prefilter_idx = valid_idx[pre_rel_idx]  # [k_pre] absolute indices

                    # Stage 2: Compute KL/CE only on prefiltered positions (efficient!)
                    # Create mask for prefiltered positions
                    prefilter_mask = torch.zeros_like(mask_i, dtype=torch.bool)
                    prefilter_mask[prefilter_idx] = True

                    # Gather only prefiltered rows for KL computation
                    # Ensure indices are on the same device as teacher log-probs
                    pre_idx_t = prefilter_idx.to(t_log_probs.device)
                    t_rows_pre = t_log_probs[i, pre_idx_t, :].to(
                        self.student_device
                    )  # [k_pre, V]
                    s_rows_pre = s_log_probs[i, prefilter_idx, :]  # [k_pre, V]
                    kl_pre = self._kl_directional(t_rows_pre, s_rows_pre)  # [k_pre]

                    # Build full-size KL tensor with -inf for non-prefiltered
                    kl_pos_partial = torch.full(
                        (ent_raw.size(1),), float("-inf"), device=self.student_device
                    )
                    kl_pos_partial[prefilter_idx] = kl_pre

                    # Prepare score context with partial KL
                    score_ctx_i = self._prepare_score_context(
                        ent_raw[i : i + 1],
                        kl_pos_partial.unsqueeze(0),
                        s_log_probs[i : i + 1] if s_log_probs is not None else None,
                        prefilter_mask.unsqueeze(0),
                        input_ids[i : i + 1],
                    )
                    combined = self._build_score_vector(score_ctx_i, 0, prefilter_mask)
                    if combined is not None:
                        stat[i] = combined

                stat = stat.masked_fill(~valid_next, float("-inf"))
            else:
                # Entropy-only ranking (no softmax needed, fully efficient)
                stat = ent_raw.masked_fill(~valid_next, float("-inf"))

            use_gls = gls_effective
            sel_topk_count = 0

            # Create boolean mask for positions to include in KD loss
            keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)  # [B, L-1]

            shared_quota = None
            if normalize_topk:
                total_valid = int(valid_next.sum().item())
                avg_valid = total_valid / max(1, valid_next.size(0))
                shared_quota = max(1, math.ceil(pct * avg_valid))

            if not use_gls:
                # Per-example top-k
                for i in range(valid_next.size(0)):
                    mask = valid_next[i]
                    n_valid = int(mask.sum().item())
                    if n_valid < 3:
                        continue
                    if normalize_topk and shared_quota is not None:
                        k = min(n_valid, shared_quota)
                    else:
                        k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                    if self._selection_curriculum_active:
                        # Curriculum expects a per-position score tensor.
                        # Prefer entropy when available, otherwise fall back to the selection stat.
                        curr_src = ent_raw if ent_raw is not None else stat
                        curr_mask = self._selection_curriculum_mask(
                            curr_src[i], mask, int(k)
                        )
                        mask = mask & curr_mask
                        n_valid_curr = int(mask.sum().item())
                        if n_valid_curr == 0:
                            continue
                        k = min(k, n_valid_curr)
                        if k == 0:
                            continue
                    sel_topk_count += int(k)
                    valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    scores = stat[i][mask]
                    if scores.numel() == 0:
                        continue
                    _, rel_idx = torch.topk(scores, k=k, largest=True, sorted=False)
                    sel = valid_idx[rel_idx]
                    keep_mask[i, sel] = True
            else:
                # GLS: global threshold over history with warm-up fallback
                self._gls_init_if_needed()
                thr = self._gls_threshold(top_percent=self.config.k_percent)
                if thr is None:
                    # Warm-up: fallback to per-example top-k
                    for i in range(valid_next.size(0)):
                        mask = valid_next[i]
                        n_valid = int(mask.sum().item())
                        if n_valid < 3:
                            continue
                        if normalize_topk and shared_quota is not None:
                            k = min(n_valid, shared_quota)
                        else:
                            k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                        if self._selection_curriculum_active:
                            curr_src = ent_raw if ent_raw is not None else stat
                            curr_mask = self._selection_curriculum_mask(
                                curr_src[i], mask, int(k)
                            )
                            mask = mask & curr_mask
                            n_valid_curr = int(mask.sum().item())
                            if n_valid_curr == 0:
                                continue
                            k = min(k, n_valid_curr)
                            if k == 0:
                                continue
                        sel_topk_count += int(k)
                        valid_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                        scores = stat[i][mask]
                        if scores.numel() == 0:
                            continue
                        _, rel_idx = torch.topk(scores, k=k, largest=True, sorted=False)
                        sel = valid_idx[rel_idx]
                        keep_mask[i, sel] = True
                else:
                    keep_mask = (stat >= thr) & valid_next
                    if self._selection_curriculum_active:
                        for i in range(valid_next.size(0)):
                            mask = valid_next[i]
                            n_valid = int(mask.sum().item())
                            if n_valid == 0:
                                continue
                            if normalize_topk and shared_quota is not None:
                                k = min(n_valid, shared_quota)
                            else:
                                k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                            curr_src = ent_raw if ent_raw is not None else stat
                            curr_mask = self._selection_curriculum_mask(
                                curr_src[i], mask, int(k)
                            )
                            keep_mask[i] = keep_mask[i] & curr_mask
                    sel_gls_count = int(keep_mask.sum().item())
                # Push current batch stats and optionally log threshold
                flat_vals = stat[valid_next].detach().float().to("cpu")
                flat_vals = flat_vals[torch.isfinite(flat_vals)]
                self._gls_push(flat_vals)
                if (
                    getattr(self.config, "gls_log_threshold", False)
                    and ("thr" in locals())
                    and thr is not None
                    and self.logger
                ):
                    self.logger.log_scalar(
                        "train/gls_threshold", float(thr), self.global_step
                    )

        if not skip_entropy:
            self._log_topk_debug_info(
                keep_mask=keep_mask,
                valid_next=valid_next,
                stat=stat,
                ent_raw=ent_raw,
                selection_policy=selection_policy,
                selection_metric=selection_metric,
            )

        # Compute KL only on selected positions (efficient)
        self._last_ce_selection_mask = keep_mask
        rows = keep_mask.nonzero(as_tuple=False)  # [P, 2] -> (b, t)
        if rows.numel() == 0:
            print(
                f"[DEBUG] top-k-tok: keep_mask has NO selected positions! valid_next.sum()={valid_next.sum().item()}",
                flush=True,
            )
            kd_loss = s_pred.sum() * 0.0
            self._kl_zero_counter = 0
        else:
            b_idx, t_idx = rows[:, 0], rows[:, 1]
            # Teacher gather requires indices on the teacher tensor's device
            device_t = t_log_probs.device
            b_idx_t = b_idx.to(device_t)
            t_idx_t = t_idx.to(device_t)
            t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
            # Student tensors live on student device; indices are already there
            s_rows = s_log_probs[b_idx, t_idx, :]
            kl_per_pos = None
            udkd_enabled = bool(getattr(self.config, "udkd_loss", False))
            if distill_type == "top-k-tok-dkd":
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets_topk = student_targets[b_idx, t_idx]
                kd_rows, tckd_rows, nckd_rows = self._compute_dkd_rows_components(
                    t_rows.exp(), s_rows.exp(), targets_topk
                )
                kd_loss = kd_rows.mean() if kd_rows.numel() > 0 else s_pred.sum() * 0.0
                if self.logger and kd_rows.numel() > 0:
                    log_fn = getattr(self.logger, "log_dkd_components", None)
                    tckd_mean = float(tckd_rows.mean().item())
                    nckd_mean = float(nckd_rows.mean().item())
                    if callable(log_fn):
                        log_fn(tckd_mean, nckd_mean, self.global_step)
                    else:
                        self.logger.log(
                            {"train/dkd_tckd": tckd_mean, "train/dkd_nckd": nckd_mean},
                            self.global_step,
                        )
            elif udkd_enabled:
                # UDKD loss: uncertainty-driven decoupled KD
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets_udkd = student_targets[b_idx, t_idx]
                udkd_metric = str(
                    getattr(self.config, "udkd_uncertainty_metric", "unc")
                )
                udkd_loss_fn = UDKDLoss(uncertainty_metric=udkd_metric)
                teacher_probs = t_rows.exp()
                student_probs = s_rows.exp()
                udkd_total, udkd_tckd, udkd_nckd, udkd_gate = udkd_loss_fn(
                    teacher_probs, student_probs, targets_udkd
                )
                kd_loss = udkd_total.mean()
                # Log UDKD components
                if self.logger:
                    self.logger.log(
                        {
                            "train/udkd_tckd": float(udkd_tckd.mean().item()),
                            "train/udkd_nckd": float(udkd_nckd.mean().item()),
                            "train/udkd_gate_mean": float(udkd_gate.mean().item()),
                        },
                        self.global_step,
                    )
            else:
                kl_per_pos = self._kl_directional(t_rows, s_rows)
                # Weighted KD: optionally weight each token's KL by uncertainty
                weighted_kd = getattr(self.config, "weighted_kd", False)
                if not weighted_kd:
                    kd_loss = kl_per_pos.mean()
                else:
                    metric = getattr(self.config, "weighted_kd_metric", None)
                    if not metric:
                        ocm = str(getattr(self.config, "offline_cache_mode", "entropy"))
                        metric = "entropy" if ocm.startswith("entropy") else "unc"

                    if metric == "entropy":
                        if ent_raw is not None:
                            weights = ent_raw[b_idx, t_idx]
                        else:
                            t_probs = t_rows.exp()
                            weights = -(
                                t_probs * torch.log(t_probs.clamp_min(1e-8))
                            ).sum(dim=-1)
                    elif metric == "unc":
                        t_probs = t_rows.exp()
                        weights = 1.0 - t_probs.max(dim=-1).values
                    elif metric == "student_entropy":
                        s_probs = s_rows.exp()
                        weights = -(s_probs * torch.log(s_probs.clamp_min(1e-8))).sum(
                            dim=-1
                        )
                    else:
                        raise ValueError(
                            "weighted_kd_metric must be one of ['entropy', 'unc', 'student_entropy'], "
                            f"got {metric!r}"
                        )

                    weights = weights.float().clamp_min(1e-8)
                    kd_loss = (kl_per_pos * weights).sum() / weights.sum().clamp_min(
                        1e-12
                    )
            kd_scalar = float(kd_loss.detach().item())
            diff = (t_rows - s_rows).detach().abs()
            max_abs_diff = float(diff.max().item()) if diff.numel() > 0 else 0.0
            mean_abs_diff = float(diff.mean().item()) if diff.numel() > 0 else 0.0
            if (max_abs_diff < 1e-6) or (abs(kd_scalar) < 1e-7):
                self._kl_zero_counter += 1
            else:
                self._kl_zero_counter = 0
            if debug_log:
                if kl_per_pos is not None:
                    stats_msg = (
                        f"kl_per_pos stats: min={kl_per_pos.min().item():.6f}, "
                        f"max={kl_per_pos.max().item():.6f}, mean={kd_scalar:.6f}"
                    )
                else:
                    stats_msg = "DKD stats: mean={:.6f}".format(kd_scalar)
                print(
                    f"[DEBUG] top-k-tok: selected {rows.size(0)} positions, {stats_msg} | "
                    f"max|Δ|={max_abs_diff:.6e}, mean|Δ|={mean_abs_diff:.6e}",
                    flush=True,
                )
            if self._kl_zero_counter >= 3:
                raise RuntimeError(
                    "KL divergence between teacher and student logits is ~0 across multiple consecutive batches. "
                    f"Last batch stats: max|Δ|={max_abs_diff:.2e}, mean|Δ|={mean_abs_diff:.2e}, kd={kd_scalar:.2e}. "
                    "This usually means the teacher is not providing a signal (e.g., wrong model, identical weights, or disabled teacher)."
                )

        # Log selection counters (per batch)
        if self.logger:
            self.logger.log(
                {
                    "train/selected_tokens_topk": float(sel_topk_count),
                    "train/selected_tokens_gls": float(sel_gls_count),
                },
                self.global_step,
            )

        # Log token selection for analysis
        self._log_token_selection(
            keep_mask, valid_next, input_ids, ent_raw, selection_policy
        )

        return kd_loss, None

    def _kd_loss_bucket(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Bucket: distill on tokens with entropy in [lower_bound, upper_bound] percentiles
        # Select positions FIRST, then compute KL only on selected rows (efficient)
        ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]

        score_enabled = bool(getattr(self.config, "score_token_selection", False))

        # Create boolean mask for positions to include in KD loss
        keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)  # [B, L-1]

        Bsz = t_log_probs.size(0)
        for i in range(Bsz):
            valid_next_i = valid_next[i]
            if valid_next_i.sum() < 3:  # Need at least 3 tokens for bucket selection
                continue

            if score_enabled:
                # === Two-stage selection for efficiency ===
                # Stage 1: Prefilter by entropy to bucket range
                ent_valid = ent_raw[i][valid_next_i].float()
                lower_thr = torch.quantile(
                    ent_valid, self.config.bucket_lower_percent / 100.0
                )
                upper_thr = torch.quantile(
                    ent_valid, self.config.bucket_upper_percent / 100.0
                )

                valid_idx = torch.where(valid_next_i)[0]
                prefilter_mask_rel = (ent_valid >= lower_thr) & (ent_valid <= upper_thr)

                if not prefilter_mask_rel.any():
                    continue

                prefilter_idx = valid_idx[
                    prefilter_mask_rel
                ]  # Absolute indices in bucket

                # Stage 2: Compute KL/CE only on bucket positions (efficient!)
                pre_idx_t = prefilter_idx.to(t_log_probs.device)
                t_rows_pre = t_log_probs[i, pre_idx_t, :].to(
                    self.student_device
                )  # [k_bucket, V]
                s_rows_pre = s_log_probs[i, prefilter_idx, :]  # [k_bucket, V]
                kl_pre = self._kl_directional(t_rows_pre, s_rows_pre)  # [k_bucket]

                # Build full-size KL tensor
                kl_pos_partial = torch.full(
                    (ent_raw.size(1),), float("-inf"), device=self.student_device
                )
                kl_pos_partial[prefilter_idx] = kl_pre

                # Create prefilter mask for score context
                prefilter_mask_full = torch.zeros_like(valid_next_i, dtype=torch.bool)
                prefilter_mask_full[prefilter_idx] = True

                # Prepare score context with partial KL
                score_ctx_i = self._prepare_score_context(
                    ent_raw[i : i + 1],
                    kl_pos_partial.unsqueeze(0),
                    s_log_probs[i : i + 1] if s_log_probs is not None else None,
                    prefilter_mask_full.unsqueeze(0),
                    input_ids[i : i + 1],
                )
                combined = self._build_score_vector(score_ctx_i, 0, prefilter_mask_full)
                if combined is None:
                    continue

                # Apply bucket thresholds to combined score
                vec = combined[prefilter_mask_full].float()
                if vec.numel() < 1:
                    continue
                score_lower = torch.quantile(
                    vec,
                    max(
                        0.0,
                        (
                            self.config.bucket_lower_percent
                            - self.config.bucket_lower_percent
                        )
                        / 100.0,
                    ),
                )
                score_upper = torch.quantile(
                    vec,
                    min(
                        1.0,
                        (
                            self.config.bucket_upper_percent
                            - self.config.bucket_lower_percent
                        )
                        / (100.0 - self.config.bucket_lower_percent),
                    ),
                )

                # Final selection within prefiltered set
                final_sel = (vec >= score_lower) & (vec <= score_upper)
                if final_sel.any():
                    keep_mask[i, prefilter_idx[final_sel]] = True
            else:
                # Entropy-only bucket (no softmax needed)
                vec = ent_raw[i][valid_next_i].float()
                lower_thr = torch.quantile(
                    vec, self.config.bucket_lower_percent / 100.0
                )
                upper_thr = torch.quantile(
                    vec, self.config.bucket_upper_percent / 100.0
                )

                keep_idx = torch.where(valid_next_i)[0]
                sel_mask = (vec >= lower_thr) & (vec <= upper_thr)
                if sel_mask.any():
                    keep_mask[i, keep_idx[sel_mask]] = True

        # Compute KL only on selected positions (efficient)
        rows = keep_mask.nonzero(as_tuple=False)  # [P, 2] -> (b, t)
        if rows.numel() == 0:
            kd_loss = s_pred.sum() * 0.0
        else:
            b_idx, t_idx = rows[:, 0], rows[:, 1]
            device_t = t_log_probs.device
            b_idx_t = b_idx.to(device_t)
            t_idx_t = t_idx.to(device_t)
            t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
            s_rows = s_log_probs[b_idx, t_idx, :]
            kd_loss = self._kl_directional(t_rows, s_rows).mean()
        return kd_loss, None

    def _kd_loss_random(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Random selection with optional score-weighted sampling
        # Select positions FIRST, then compute KL only on selected rows (efficient)
        use_dkd = self.config.distill_type == "random-dkd"
        score_enabled = bool(getattr(self.config, "score_token_selection", False))
        score_ctx = None
        ent_raw = None
        if score_enabled or self._selection_curriculum_active:
            ent_raw = self._entropy_for_selection(
                input_ids, t_pred
            )  # [B, L-1] for context only
        if score_enabled:
            assert ent_raw is not None
            kl_pos_for_score = self._kl_directional(t_log_probs, s_log_probs)
            score_ctx = self._prepare_score_context(
                ent_raw, kl_pos_for_score, s_log_probs, valid_next, input_ids
            )

        # Create boolean mask for positions to include in KD loss
        keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)  # [B, L-1]

        Bsz = t_log_probs.size(0)
        for i in range(Bsz):
            valid_next_i = valid_next[i]
            valid_count = int(valid_next_i.sum().item())
            if valid_count < 2:
                continue
            k_count = max(1, int(valid_count * self.config.k_percent / 100.0))
            valid_indices = torch.where(valid_next_i)[0]
            if len(valid_indices) < k_count:
                continue

            if score_enabled:
                if score_ctx is None:
                    continue
                combined = self._build_score_vector(score_ctx, i, valid_next_i)
                if combined is None:
                    continue
                score_valid = combined[valid_next_i].float()
                if self._selection_curriculum_active and ent_raw is not None:
                    ent_valid = ent_raw[i][valid_next_i]
                    weights = self._selection_curriculum_sampling_weights(ent_valid).to(
                        score_valid.device
                    )
                    score_valid = score_valid * weights
                # turn scores into sampling probs
                s = score_valid - score_valid.min()
                s = torch.clamp(s, min=1e-8)
                probs = s / s.sum()
                rel = torch.multinomial(probs, num_samples=k_count, replacement=False)
                selected_indices = valid_indices[rel]
            else:
                if self._selection_curriculum_active and ent_raw is not None:
                    ent_valid = ent_raw[i][valid_next_i]
                    weights = self._selection_curriculum_sampling_weights(ent_valid).to(
                        self.student_device
                    )
                    probs = weights / weights.sum()
                    rel = torch.multinomial(
                        probs, num_samples=k_count, replacement=False
                    )
                    selected_indices = valid_indices[rel]
                else:
                    perm = torch.randperm(
                        len(valid_indices), device=self.student_device
                    )
                    selected_indices = valid_indices[perm[:k_count]]

            keep_mask[i, selected_indices] = True

        # Compute KL only on selected positions (efficient)
        rows = keep_mask.nonzero(as_tuple=False)  # [P, 2] -> (b, t)
        if rows.numel() == 0:
            kd_loss = s_pred.sum() * 0.0
        else:
            b_idx, t_idx = rows[:, 0], rows[:, 1]
            device_t = t_log_probs.device
            b_idx_t = b_idx.to(device_t)
            t_idx_t = t_idx.to(device_t)
            t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
            s_rows = s_log_probs[b_idx, t_idx, :]
            if use_dkd:
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets = student_targets[b_idx, t_idx]
                kd_loss = self._compute_dkd_total_rows(
                    t_rows.exp(), s_rows.exp(), targets
                ).mean()
            elif bool(getattr(self.config, "udkd_loss", False)):
                # UDKD loss: uncertainty-driven decoupled KD
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets_udkd = student_targets[b_idx, t_idx]
                udkd_metric = str(
                    getattr(self.config, "udkd_uncertainty_metric", "unc")
                )
                udkd_loss_fn = UDKDLoss(uncertainty_metric=udkd_metric)
                teacher_probs = t_rows.exp()
                student_probs = s_rows.exp()
                udkd_total, udkd_tckd, udkd_nckd, udkd_gate = udkd_loss_fn(
                    teacher_probs, student_probs, targets_udkd
                )
                kd_loss = udkd_total.mean()
                if self.logger:
                    self.logger.log(
                        {
                            "train/udkd_tckd": float(udkd_tckd.mean().item()),
                            "train/udkd_nckd": float(udkd_nckd.mean().item()),
                            "train/udkd_gate_mean": float(udkd_gate.mean().item()),
                        },
                        self.global_step,
                    )
            else:
                kd_loss = self._kl_directional(t_rows, s_rows).mean()
        return kd_loss, None

    def _kd_loss_atkd(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if t_log_probs is None:
            raise RuntimeError(
                "AT-KD requires teacher log-probs; disable offline cache or ensure teacher is available."
            )
        rows = valid_next.nonzero(as_tuple=False)
        if rows.numel() == 0:
            kd_loss = s_pred.sum() * 0.0
        else:
            batch_idx, pos_idx = (
                rows[:, 0],
                rows[:, 1],
            )  # batch and position indices of valid tokens
            device_t = t_log_probs.device
            batch_idx_t = batch_idx.to(device_t)
            pos_idx_t = pos_idx.to(device_t)
            if t_pred is None:
                raise RuntimeError(
                    "AT-KD requires teacher logits for T=1 probability computation."
                )
            t_rows_logits = t_pred[batch_idx_t, pos_idx_t, :].to(self.student_device)
            s_rows_logits = s_pred[batch_idx, pos_idx, :].to(self.student_device)

            P_total = t_rows_logits.size(0)
            t_rows_T1 = torch.log_softmax(t_rows_logits.float(), dim=-1)
            s_rows_T1 = torch.log_softmax(s_rows_logits.float(), dim=-1)

            teacher_probs = torch.exp(t_rows_T1)
            student_probs = torch.exp(s_rows_T1)

            row_indices = torch.arange(P_total, device=self.student_device)
            # Ground-truth target ids
            # input_ids[:, 1:] gives the next-token targets; index by (batch_idx, pos_idx) to get gold labels
            gold_target_ids = input_ids[:, 1:].to(self.student_device)[
                batch_idx, pos_idx
            ]
            teacher_tgt = teacher_probs[row_indices, gold_target_ids]
            student_tgt = student_probs[row_indices, gold_target_ids]
            # UnC metric from "Revisiting KD in LLMs": 1 - p(target)
            uncertainty = 1.0 - teacher_tgt

            hard_pct = float(
                getattr(
                    self.config,
                    "atkd_hard_percent",
                    getattr(self.config, "k_percent", 50),
                )
            )
            hard_pct = max(0.0, min(100.0, hard_pct))
            if P_total == 0 or hard_pct <= 0.0:
                hard_count_req = 0
            elif hard_pct >= 100.0:
                hard_count_req = P_total
            else:
                hard_count_req = int(math.ceil((hard_pct / 100.0) * P_total))
                hard_count_req = max(1, min(P_total, hard_count_req))

            hard_mask = torch.zeros(
                P_total, dtype=torch.bool, device=self.student_device
            )
            if hard_count_req >= P_total:
                hard_mask[:] = True
            elif hard_count_req > 0:
                _, hard_idx = torch.topk(
                    uncertainty, hard_count_req, largest=True, sorted=False
                )
                hard_mask[hard_idx] = True
            easy_mask = ~hard_mask
            hard_count = int(hard_mask.sum().item())
            easy_count = int(easy_mask.sum().item())

            # TCKD: binary distribution [p(target), 1-p(target)] - uses ground-truth target per DKD paper
            teacher_rest = (1.0 - teacher_tgt).clamp_min(0.0)
            student_rest = (1.0 - student_tgt).clamp_min(0.0)
            teacher_bin = torch.stack([teacher_tgt, teacher_rest], dim=-1)
            student_bin = torch.stack([student_tgt, student_rest], dim=-1)
            teacher_bin = teacher_bin / teacher_bin.sum(dim=-1, keepdim=True).clamp_min(
                1e-9
            )
            student_bin = student_bin / student_bin.sum(dim=-1, keepdim=True).clamp_min(
                1e-9
            )
            log_teacher_bin = torch.log(teacher_bin.clamp_min(1e-9))
            log_student_bin = torch.log(student_bin.clamp_min(1e-9))
            kl_bin = (
                F.kl_div(
                    log_student_bin, log_teacher_bin, log_target=True, reduction="none"
                )
                .sum(-1)
                .to(s_pred.dtype)
            )

            # NCKD: distribution over non-target classes (vocab minus target) - uses ground-truth target per DKD paper
            teacher_off = teacher_probs.clone()
            teacher_off[row_indices, gold_target_ids] = 0.0
            student_off = student_probs.clone()
            student_off[row_indices, gold_target_ids] = 0.0
            teacher_off_sum = teacher_off.sum(dim=-1)
            student_off_sum = student_off.sum(dim=-1)

            kl_off = torch.zeros(
                P_total, device=self.student_device, dtype=s_pred.dtype
            )
            valid_off = teacher_off_sum > 1e-8
            if valid_off.any():
                idx_valid = valid_off.nonzero(as_tuple=False).squeeze(-1)
                t_hat = teacher_off[idx_valid] / teacher_off_sum[idx_valid].unsqueeze(
                    -1
                )
                s_hat = student_off[idx_valid] / student_off_sum[idx_valid].unsqueeze(
                    -1
                ).clamp_min(1e-8)
                log_t_hat = torch.log(t_hat.clamp_min(1e-9))
                log_s_hat = torch.log(s_hat.clamp_min(1e-9))
                kl_vals = (
                    F.kl_div(log_s_hat, log_t_hat, log_target=True, reduction="none")
                    .sum(-1)
                    .to(s_pred.dtype)
                )
                kl_off[idx_valid] = kl_vals

            lambda_easy = float(getattr(self.config, "atkd_loss_lambda", 0.2))
            lambda_easy = float(min(max(lambda_easy, 0.0), 1.0))

            zero_like = s_pred.sum() * 0.0
            easy_avg = None
            hard_avg = None
            if easy_count > 0:
                easy_values = kl_off[easy_mask]
                easy_avg = easy_values.mean()
            if hard_count > 0:
                hard_values = kl_bin[hard_mask] + kl_off[hard_mask]
                hard_avg = hard_values.mean()

            # Eq. (5) from AT-KD paper: λ * avg(easy) + (1-λ) * avg(hard)
            # Each group is averaged within itself, then combined with λ weighting
            if easy_count == 0 and hard_count == 0:
                kd_loss = zero_like
            elif easy_count == 0:
                assert hard_avg is not None
                kd_loss = hard_avg
            elif hard_count == 0:
                assert easy_avg is not None
                kd_loss = easy_avg
            else:
                assert easy_avg is not None and hard_avg is not None
                kd_loss = lambda_easy * easy_avg + (1.0 - lambda_easy) * hard_avg

            if self.logger:
                metrics = {
                    "train/atkd_easy_tokens": float(easy_count),
                    "train/atkd_hard_tokens": float(hard_count),
                    "train/atkd_lambda_easy": lambda_easy,
                    "train/atkd_hard_percent": hard_pct,
                }
                if hard_count > 0:
                    metrics["train/atkd_unc_threshold"] = float(
                        uncertainty[hard_mask].min().item()
                    )
                if easy_avg is not None:
                    metrics["train/atkd_easy_loss"] = float(easy_avg.item())
                if hard_avg is not None:
                    metrics["train/atkd_hard_loss"] = float(hard_avg.item())
                self.logger.log(metrics, self.global_step)
        return kd_loss, None

    def _kd_loss_pos_rs_kd(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # RS-KD over POSITIONS: sample K% positions by distribution q(i)
        # Select positions FIRST with importance weights, then compute KL only on selected rows

        # Determine selection metric based on offline_cache_mode
        ocm = getattr(self.config, "offline_cache_mode", "entropy")
        if ocm == "unc":
            # Use uncertainty = 1 - max(p) as importance measure
            t_probs = torch.softmax(t_pred, dim=-1)  # [B, L, V]
            unc_raw = 1.0 - t_probs.max(dim=-1).values[:, :-1]  # [B, L-1]
            selection_scores = unc_raw
            ent_raw = None  # Not needed for unc mode
        else:
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            selection_scores = ent_raw

        score_enabled = bool(getattr(self.config, "score_token_selection", False))
        score_ctx = None
        if score_enabled:
            kl_pos_for_score = self._kl_directional(t_log_probs, s_log_probs)
            score_ctx = self._prepare_score_context(
                selection_scores, kl_pos_for_score, s_log_probs, valid_next, input_ids
            )

        # Build per-position importance weights
        Bsz = t_pred.size(0)
        alpha = float(getattr(self.config, "rs_alpha", 1.0))
        q_floor = float(getattr(self.config, "rs_floor", 1e-6))
        pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
        bucket_bounds = self._rs_bucket_bounds()
        normalize_topk = bool(getattr(self.config, "normalize_topk_by_length", False))
        shared_quota = None
        if normalize_topk:
            total_valid = int(valid_next.sum().item())
            avg_valid = total_valid / max(1, valid_next.size(0))
            shared_quota = max(1, math.ceil(pct * avg_valid))
        policy_bits = [
            f"k%={float(getattr(self.config, 'k_percent', 0.0)):.2f}",
            f"mode={ocm}",
            f"alpha={alpha:.3f}",
            f"normalize={'on' if normalize_topk else 'off'}",
            f"bucket={'on' if bucket_bounds is not None else 'off'}",
            f"score={'on' if score_enabled else 'off'}",
        ]
        selection_policy = f"{self.config.distill_type} | " + ", ".join(policy_bits)

        # Collect selected positions (duplicates allowed for q-weighted KD)
        selected_positions: List[Tuple[int, int]] = []  # (b, t)
        match_full_kd = bool(getattr(self.config, "pos_rs_match_full_kd", False))
        selected_q_values: List[float] = []

        for i in range(Bsz):
            valid_next_i = valid_next[i]  # [L-1] bool of valid positions
            if valid_next_i.sum().item() < 3:
                continue

            if score_enabled:
                if score_ctx is None:
                    continue
                combined = self._build_score_vector(score_ctx, i, valid_next_i)
                if combined is None:
                    continue
                vec = combined[valid_next_i].float()
            else:
                vec = selection_scores[i][valid_next_i].float()

            valid_idx = torch.where(valid_next_i)[0]
            if bucket_bounds is not None:
                filtered_vec, filtered_idx, filtered = self._apply_rs_bucket_filter(
                    vec, valid_idx, bucket_bounds
                )
                if filtered:
                    vec = filtered_vec
                    valid_idx = filtered_idx

            valid_count = int(valid_idx.numel())
            if valid_count < 1:
                continue

            vec = torch.clamp(vec, min=1e-8)  # H_t >= 0 already
            logits = vec if alpha == 1.0 else vec * alpha
            q = torch.softmax(logits, dim=0)
            if not torch.isfinite(q).all():
                q = torch.full_like(vec, 1.0 / max(1, vec.numel()))
            else:
                q = torch.clamp(q, min=q_floor)
                q_sum = q.sum()
                if q_sum <= 0:
                    q = torch.full_like(vec, 1.0 / max(1, vec.numel()))
                else:
                    q = q / q_sum

            if normalize_topk and shared_quota is not None:
                k_count = min(valid_count, shared_quota)
            else:
                k_count = max(1, min(valid_count, math.ceil(pct * valid_count)))

            # Sampling WITH replacement: duplicates => multiple contributions (q-weighted KD)
            sel_rel = torch.multinomial(q, num_samples=k_count, replacement=True)  # [k]
            selected_abs = valid_idx[sel_rel]  # [k]

            # Store (batch_idx, time_idx) - optionally keep q(t) for importance weighting
            sel_q_vals = q[sel_rel] if match_full_kd else None
            for idx_sel, pos in enumerate(selected_abs):
                selected_positions.append((i, int(pos.item())))
                if match_full_kd and sel_q_vals is not None:
                    selected_q_values.append(float(sel_q_vals[idx_sel].item()))

        # Compute KD loss as the mean over all sampled positions (q-weighted KD)
        if len(selected_positions) == 0:
            kd_loss = s_pred.sum() * 0.0
        else:
            b_indices = torch.tensor(
                [p[0] for p in selected_positions],
                dtype=torch.long,
                device=self.student_device,
            )
            t_indices = torch.tensor(
                [p[1] for p in selected_positions],
                dtype=torch.long,
                device=self.student_device,
            )
            # Move indices to teacher device for gather
            b_idx_t = b_indices.to(t_log_probs.device)
            t_idx_t = t_indices.to(t_log_probs.device)
            t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
            s_rows = s_log_probs[b_indices, t_indices, :]
            udkd_enabled = bool(getattr(self.config, "udkd_loss", False))
            if udkd_enabled:
                # UDKD loss: uncertainty-driven decoupled KD
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets_udkd = student_targets[b_indices, t_indices]
                udkd_metric = str(
                    getattr(self.config, "udkd_uncertainty_metric", "unc")
                )
                udkd_loss_fn = UDKDLoss(uncertainty_metric=udkd_metric)
                teacher_probs = t_rows.exp()
                student_probs = s_rows.exp()
                udkd_total, udkd_tckd, udkd_nckd, udkd_gate = udkd_loss_fn(
                    teacher_probs, student_probs, targets_udkd
                )
                kd_loss = udkd_total.mean()
                if self.logger:
                    self.logger.log(
                        {
                            "train/udkd_tckd": float(udkd_tckd.mean().item()),
                            "train/udkd_nckd": float(udkd_nckd.mean().item()),
                            "train/udkd_gate_mean": float(udkd_gate.mean().item()),
                        },
                        self.global_step,
                    )
            else:
                kl_per_pos = self._kl_directional(t_rows, s_rows)  # [P]
                if match_full_kd and selected_q_values:
                    q_tensor = torch.tensor(
                        selected_q_values,
                        dtype=kl_per_pos.dtype,
                        device=self.student_device,
                    )
                    denom = torch.clamp(q_tensor, min=1e-8)
                    kd_loss = (kl_per_pos / denom).mean()
                else:
                    # Simple mean over all sampled positions (duplicates count multiple times)
                    kd_loss = kl_per_pos.mean()
        log_ent = ent_raw if ocm != "unc" else None
        self._log_selected_positions(
            selected_positions, valid_next, input_ids, log_ent, selection_policy
        )
        return kd_loss, None

    def _kd_loss_pos_rs_kd_dkd(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        # Determine selection metric based on offline_cache_mode
        ocm = getattr(self.config, "offline_cache_mode", "entropy")
        ent_raw: Optional[torch.Tensor] = None
        if ocm == "unc":
            # Use uncertainty = 1 - max(p) as importance measure
            t_probs = torch.softmax(t_pred, dim=-1)  # [B, L, V]
            unc_raw = 1.0 - t_probs.max(dim=-1).values[:, :-1]  # [B, L-1]
            selection_scores = unc_raw.to(valid_next.device)
        else:
            ent_raw = self._entropy_for_selection(input_ids, t_pred)  # [B, L-1]
            ent_raw = ent_raw.to(valid_next.device)
            selection_scores = ent_raw

        score_enabled = bool(getattr(self.config, "score_token_selection", False))
        score_ctx = None
        if score_enabled:
            kl_pos_for_score = self._kl_directional(t_log_probs, s_log_probs)
            score_ctx = self._prepare_score_context(
                selection_scores, kl_pos_for_score, s_log_probs, valid_next, input_ids
            )

        selected_positions: List[
            Tuple[int, int]
        ] = []  # (b, t) - no weights for q-weighted KD
        match_full_kd = bool(getattr(self.config, "pos_rs_match_full_kd", False))
        selected_q_values: List[float] = []
        Bsz = t_pred.size(0)
        alpha_rs = float(getattr(self.config, "rs_alpha", 1.0))
        q_floor = float(getattr(self.config, "rs_floor", 1e-6))
        pct = max(0.0, min(1.0, self.config.k_percent / 100.0))
        bucket_bounds = self._rs_bucket_bounds()
        normalize_topk = bool(getattr(self.config, "normalize_topk_by_length", False))
        shared_quota = None
        if normalize_topk:
            total_valid = int(valid_next.sum().item())
            avg_valid = total_valid / max(1, valid_next.size(0))
            shared_quota = max(1, math.ceil(pct * avg_valid))
        policy_bits = [
            f"k%={float(getattr(self.config, 'k_percent', 0.0)):.2f}",
            f"mode={ocm}",
            f"alpha={alpha_rs:.3f}",
            f"normalize={'on' if normalize_topk else 'off'}",
            f"bucket={'on' if bucket_bounds is not None else 'off'}",
            f"score={'on' if score_enabled else 'off'}",
        ]
        selection_policy = f"{self.config.distill_type} | " + ", ".join(policy_bits)

        for i in range(Bsz):
            valid_next_i = valid_next[i]
            if valid_next_i.sum().item() < 1:
                continue
            valid_idx = torch.where(valid_next_i)[0]

            if score_enabled:
                if score_ctx is None:
                    continue
                combined = self._build_score_vector(score_ctx, i, valid_next_i)
                if combined is None:
                    continue
                vec = combined[valid_next_i].float()
            else:
                vec = selection_scores[i][valid_next_i].float()

            if bucket_bounds is not None:
                vec, valid_idx, _ = self._apply_rs_bucket_filter(
                    vec, valid_idx, bucket_bounds
                )

            valid_count = int(valid_idx.numel())
            if valid_count < 1:
                continue

            vec = torch.clamp(vec, min=1e-8)
            logits = vec if alpha_rs == 1.0 else vec * alpha_rs
            q = torch.softmax(logits, dim=0)
            q = torch.clamp(q, min=q_floor)
            q_sum = q.sum()
            if not torch.isfinite(q_sum) or q_sum <= 0:
                q = torch.full_like(vec, 1.0 / max(1, vec.numel()))
            else:
                q = q / q_sum

            if normalize_topk and shared_quota is not None:
                k_count = min(valid_count, shared_quota)
            else:
                k_count = max(1, min(valid_count, math.ceil(pct * valid_count)))

            # Sampling WITH replacement: duplicates => multiple contributions (q-weighted KD)
            sel_rel = torch.multinomial(q, num_samples=k_count, replacement=True)
            sel_abs = valid_idx[sel_rel]
            sel_q_vals = q[sel_rel] if match_full_kd else None
            for idx_sel, pos in enumerate(sel_abs):
                selected_positions.append((i, int(pos.item())))
                if match_full_kd and sel_q_vals is not None:
                    selected_q_values.append(float(sel_q_vals[idx_sel].item()))

        if len(selected_positions) == 0:
            kd_loss = s_pred.sum() * 0.0
        else:
            b_indices = torch.tensor(
                [p[0] for p in selected_positions],
                dtype=torch.long,
                device=self.student_device,
            )
            t_indices = torch.tensor(
                [p[1] for p in selected_positions],
                dtype=torch.long,
                device=self.student_device,
            )
            device_t = t_log_probs.device
            b_idx_t = b_indices.to(device_t)
            t_idx_t = t_indices.to(device_t)
            t_rows_probs = torch.exp(t_log_probs[b_idx_t, t_idx_t, :]).to(
                self.student_device
            )
            s_rows_probs = torch.exp(s_log_probs[b_indices, t_indices, :])
            student_targets = input_ids[:, 1:].to(self.student_device)
            targets = student_targets[b_indices, t_indices]
            kd_rows = self._compute_dkd_total_rows(t_rows_probs, s_rows_probs, targets)
            if match_full_kd and selected_q_values:
                q_tensor = torch.tensor(
                    selected_q_values, dtype=kd_rows.dtype, device=self.student_device
                )
                denom = torch.clamp(q_tensor, min=1e-8)
                kd_loss = (kd_rows / denom).mean()
            else:
                # Simple mean over all sampled positions (duplicates count multiple times)
                kd_loss = kd_rows.mean()
        log_ent = ent_raw if (not (ocm == "unc")) else None
        self._log_selected_positions(
            selected_positions, valid_next, input_ids, log_ent, selection_policy
        )
        return kd_loss, None

    def _kd_loss_linucb(
        self,
        *,
        t_pred: torch.Tensor,
        t_log_probs: torch.Tensor,
        s_pred: torch.Tensor,
        s_log_probs: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        temperature: float,
        debug_log: bool,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        if self.bandit_manager is None:
            raise RuntimeError("LinUCB bandit is not initialized.")
        ent_raw = self._entropy_for_selection(input_ids, t_pred)
        # Use the same temperature as KD for CE computation (consistency fix)
        kl_pos = self._kl_directional(t_log_probs, s_log_probs)
        student_entropy = (-(s_log_probs.exp() * s_log_probs).sum(-1)).detach()

        targets = input_ids[:, 1:].to(self.student_device)
        targets = targets.masked_fill(~valid_next, 0)
        # Teacher/student CE and KL are the contextual features consumed by the bandit.
        # Use log_probs at temperature T (already computed) for consistency
        targets_t = targets.to(t_log_probs.device, non_blocking=True)
        teacher_ce = (
            (-t_log_probs.gather(-1, targets_t.unsqueeze(-1)).squeeze(-1))
            .to(self.student_device)
            .detach()
        )
        student_ce = (
            -s_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        ).detach()

        kd_terms, metrics, selection = self.bandit_manager.select_tokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            ent_raw=ent_raw.detach(),
            student_entropy=student_entropy,
            teacher_ce=teacher_ce,
            student_ce=student_ce,
            kl_pos=kl_pos,
            valid_next=valid_next,
            temperature=temperature,
        )
        use_dkd = getattr(self.config, "distill_type", "linucb") == "linucb-dkd"
        udkd_enabled = bool(getattr(self.config, "udkd_loss", False))
        if use_dkd:
            if selection is None or selection[0].numel() == 0:
                kd_loss = s_pred.sum() * 0.0
            else:
                b_idx_cpu, t_idx_cpu = selection
                b_idx_student = b_idx_cpu.to(self.student_device)
                t_idx_student = t_idx_cpu.to(self.student_device)
                device_t = t_log_probs.device
                b_idx_t = b_idx_cpu.to(device_t)
                t_idx_t = t_idx_cpu.to(device_t)
                t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
                s_rows = s_log_probs[b_idx_student, t_idx_student, :]
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets_sel = student_targets[b_idx_student, t_idx_student]
                kd_rows = self._compute_dkd_total_rows(
                    t_rows.exp(), s_rows.exp(), targets_sel
                )
                kd_loss = kd_rows.mean() if kd_rows.numel() > 0 else s_pred.sum() * 0.0
        elif udkd_enabled:
            # UDKD loss for linucb
            if selection is None or selection[0].numel() == 0:
                kd_loss = s_pred.sum() * 0.0
            else:
                b_idx_cpu, t_idx_cpu = selection
                b_idx_student = b_idx_cpu.to(self.student_device)
                t_idx_student = t_idx_cpu.to(self.student_device)
                device_t = t_log_probs.device
                b_idx_t = b_idx_cpu.to(device_t)
                t_idx_t = t_idx_cpu.to(device_t)
                t_rows = t_log_probs[b_idx_t, t_idx_t, :].to(self.student_device)
                s_rows = s_log_probs[b_idx_student, t_idx_student, :]
                student_targets = input_ids[:, 1:].to(self.student_device)
                targets_udkd = student_targets[b_idx_student, t_idx_student]
                udkd_metric = str(
                    getattr(self.config, "udkd_uncertainty_metric", "unc")
                )
                udkd_loss_fn = UDKDLoss(uncertainty_metric=udkd_metric)
                teacher_probs = t_rows.exp()
                student_probs = s_rows.exp()
                udkd_total, udkd_tckd, udkd_nckd, udkd_gate = udkd_loss_fn(
                    teacher_probs, student_probs, targets_udkd
                )
                kd_loss = udkd_total.mean()
                if self.logger:
                    self.logger.log(
                        {
                            "train/udkd_tckd": float(udkd_tckd.mean().item()),
                            "train/udkd_nckd": float(udkd_nckd.mean().item()),
                            "train/udkd_gate_mean": float(udkd_gate.mean().item()),
                        },
                        self.global_step,
                    )
        else:
            kd_loss = torch.cat(kd_terms).mean() if kd_terms else s_pred.sum() * 0.0
        extra = metrics or None
        return kd_loss, extra

    def train(self, epochs: int = 1, log_every: int = 100):
        """Run distillation training for specified number of epochs."""
        overall_train_start = time.time()
        self._wall_start = overall_train_start
        self._time_limit_triggered = False
        self._max_train_seconds: Optional[float] = None
        max_train_hours = getattr(self.config, "max_train_hours", None)
        try:
            if max_train_hours is not None:
                hours_val = float(max_train_hours)
                if hours_val > 0:
                    self._max_train_seconds = hours_val * 3600.0
        except (TypeError, ValueError):
            self._max_train_seconds = None
        rank_is_zero = (not self.ddp_enabled) or (self.ddp_rank == 0)
        build_cache_on_rank = (not self.ddp_enabled) or (self.ddp_rank == 0)
        if rank_is_zero and self._max_train_seconds is not None:
            print(
                f"[timelimit] Max wall-clock training time: {self._max_train_seconds / 3600.0:.2f}h"
            )
        # Optional frozen-student pre-pass for skipping a fraction of samples.
        if self.sample_skipper is not None:
            self.sample_skipper.prepare(rank_is_zero=rank_is_zero)

        if getattr(self.config, "offline_cache", False):
            self._prepare_offline_cache(build_cache_on_rank)

        self._reset_bandit_state()
        kd_schedule = self._create_kd_schedule(epochs)
        self._run_training_epochs(epochs, log_every, rank_is_zero, kd_schedule)
        # Final checkpoint and cleanup at the end
        if self.config.checkpoint_steps > 0 and rank_is_zero:
            print("Training completed. Performing final cleanup of old checkpoints...")
            self._cleanup_old_checkpoints()

        overall_train_elapsed = time.time() - overall_train_start
        if rank_is_zero:
            print(
                f"[distill] Total training duration: {overall_train_elapsed:.2f}s for {epochs} epoch(s)"
            )
            self._log_peak_memory_summary(overall_train_elapsed)
            self._write_efficiency_row(overall_train_elapsed)

    def _write_efficiency_row(self, overall_train_elapsed_s: float) -> None:
        if not bool(getattr(self.config, "log_efficiency_csv", False)):
            return

        csv_path = Path(
            getattr(
                self.config, "efficiency_csv_path", "results/table_efficiency_test.csv"
            )
        )
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        display_name = os.getenv("RUN_DISPLAY_NAME", "").strip()
        output_dir = str(getattr(self.config, "output_dir", ""))

        forward_s = 0.0
        prepass_tokens = 0
        prepass_strategy = None
        if self.sample_skipper is not None:
            forward_s = float(
                getattr(self.sample_skipper, "prepass_forward_s", 0.0) or 0.0
            )
            prepass_tokens = int(getattr(self.sample_skipper, "prepass_tokens", 0) or 0)
            prepass_strategy = getattr(self.sample_skipper, "prepass_strategy", None)

        iterations = int(self._total_train_iterations)
        total_wall_min = float(overall_train_elapsed_s) / 60.0
        forward_min = (
            float(forward_s) / 60.0 if self.sample_skipper is not None else 0.0
        )

        student_params = self._param_count(self._student_base)
        teacher_params = (
            self._param_count(self.teacher) if self.teacher is not None else 0
        )

        train_tokens = int(self._total_train_tokens)
        train_flops = train_tokens * (
            6 * student_params + (2 * teacher_params if teacher_params else 0)
        )

        prepass_flops = 0
        if prepass_tokens > 0:
            prepass_flops = prepass_tokens * (2 * student_params)
            if prepass_strategy in {"kl", "ce_ratio"} and teacher_params:
                prepass_flops += prepass_tokens * (2 * teacher_params)

        total_flops = int(train_flops + prepass_flops)

        # Compute peak memory statistics
        peak_mem_avg = 0.0
        if self._peak_memory_samples:
            peak_mem_avg = sum(self._peak_memory_samples) / len(
                self._peak_memory_samples
            )

        header = [
            "display_name",
            "output_dir",
            "total_wall_minutes",
            "forward_wall_minutes",
            "iterations",
            "total_flops",
            "peak_memory_gb",
        ]
        row = {
            "display_name": display_name,
            "output_dir": output_dir,
            "total_wall_minutes": f"{total_wall_min:.2f}",
            "forward_wall_minutes": f"{forward_min:.2f}",
            "iterations": str(iterations),
            "total_flops": str(total_flops),
            "peak_memory_gb": f"{peak_mem_avg:.3f}",
        }

        write_header = not csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _maybe_handle_time_limit(
        self, epoch: int, step: int, rank_is_zero: bool
    ) -> None:
        if self._max_train_seconds is None or self._time_limit_triggered:
            return
        elapsed = time.time() - self._wall_start
        stop = elapsed >= self._max_train_seconds
        if self.ddp_enabled and dist.is_initialized():
            flag = torch.tensor(1 if stop else 0, device=self.student_device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            stop = bool(flag.item())
        if not stop:
            return
        ckpt_path = None
        if rank_is_zero:
            ckpt_path = self.save_checkpoint(epoch, step, force=True)
        if self.ddp_enabled and dist.is_initialized():
            dist.barrier()
        self._time_limit_triggered = True
        limit_hours = (
            self._max_train_seconds / 3600.0 if self._max_train_seconds else 0.0
        )
        msg = f"Reached max_train_hours after {elapsed / 3600.0:.2f}h (limit {limit_hours:.2f}h)."
        raise TrainingTimeLimitReached(
            msg,
            checkpoint_path=ckpt_path if rank_is_zero else None,
            elapsed_s=elapsed,
            limit_s=self._max_train_seconds,
        )

    def _log_peak_memory_summary(self, elapsed_s: float) -> None:
        """Log peak GPU memory to stdout and append to results/peak_memory_{node}_{gpu}.json."""
        if self.student_device.type != "cuda":
            return

        hours = int(elapsed_s // 3600)
        minutes = int((elapsed_s % 3600) // 60)
        wall_str = f"{hours}h{minutes:02d}m"

        student_peak_gb = torch.cuda.max_memory_allocated(self.student_device) / (
            1024**3
        )
        student_reserved_gb = torch.cuda.max_memory_reserved(self.student_device) / (
            1024**3
        )
        avg_peak_gb: Optional[float] = None
        max_peak_gb: Optional[float] = None
        run_max_gb: Optional[float] = None
        peak_steps = 0
        if self._peak_memory_samples:
            peak_steps = len(self._peak_memory_samples)
            avg_peak_gb = sum(self._peak_memory_samples) / peak_steps
            max_peak_gb = max(self._peak_memory_samples)
            run_max_gb = max(max_peak_gb, student_peak_gb)
        else:
            run_max_gb = student_peak_gb
        teacher_dev = getattr(self, "teacher_device", None)
        teacher_loaded = self.teacher is not None
        devices = {}
        devices[str(self.student_device)] = {
            "peak_gb": round(student_peak_gb, 2),
            "reserved_gb": round(student_reserved_gb, 2),
            "max_allocation_gb": round(run_max_gb, 2)
            if run_max_gb is not None
            else round(student_peak_gb, 2),
            "role": "student",
            "gpu_type": torch.cuda.get_device_name(self.student_device),
        }

        total_peak = student_peak_gb
        if (
            teacher_loaded
            and teacher_dev
            and teacher_dev.type == "cuda"
            and teacher_dev != self.student_device
        ):
            teacher_peak_gb = torch.cuda.max_memory_allocated(teacher_dev) / (1024**3)
            teacher_reserved_gb = torch.cuda.max_memory_reserved(teacher_dev) / (
                1024**3
            )
            teacher_run_max_gb = teacher_peak_gb
            devices[str(teacher_dev)] = {
                "peak_gb": round(teacher_peak_gb, 2),
                "reserved_gb": round(teacher_reserved_gb, 2),
                "max_allocation_gb": round(teacher_run_max_gb, 2),
                "role": "teacher",
                "gpu_type": torch.cuda.get_device_name(teacher_dev),
            }
            total_peak += teacher_peak_gb
        elif teacher_loaded and teacher_dev and teacher_dev == self.student_device:
            devices[str(self.student_device)]["role"] = "student+teacher"

        dev_parts = [
            f"{dev} ({info['role']}): {info['max_allocation_gb']:.2f} GB"
            for dev, info in devices.items()
        ]
        print(
            f"[memory] Peak GPU memory - {', '.join(dev_parts)}, total: {total_peak:.2f} GB"
        )
        if avg_peak_gb is not None and max_peak_gb is not None:
            print(
                "[memory] Student peak (per-step, reset each step) - "
                f"avg: {avg_peak_gb:.2f} GB, max_step: {max_peak_gb:.2f} GB, "
                f"max_run: {run_max_gb:.2f} GB, last: {student_peak_gb:.2f} GB, "
                f"reserved(last): {student_reserved_gb:.2f} GB "
                f"over {peak_steps} steps"
            )
        else:
            print(
                "[memory] Student peak reflects only the last step "
                f"(per-step tracking disabled); max_run: {run_max_gb:.2f} GB, "
                f"last: {student_peak_gb:.2f} GB, reserved(last): {student_reserved_gb:.2f} GB"
            )

        display_name = os.getenv("RUN_DISPLAY_NAME", "").strip()
        output_dir = str(getattr(self.config, "output_dir", ""))
        autopilot_run_label = os.getenv("AUTOPILOT_RUN_LABEL", "").strip()
        autopilot_run_serial = os.getenv("AUTOPILOT_RUN_SERIAL", "").strip()
        slurm_job_id = os.getenv("SLURM_JOB_ID", "local").strip() or "local"
        entry = {
            "display_name": display_name,
            "output_dir": output_dir,
            "autopilot_run_label": autopilot_run_label or None,
            "autopilot_run_serial": autopilot_run_serial or None,
            "slurm_job_id": slurm_job_id,
            "wall_time": wall_str,
            "wall_time_sec": round(float(elapsed_s), 2),
            "wall_time_minutes": round(float(elapsed_s) / 60.0, 2),
            "peak_memory_total_gb": round(total_peak, 2),
            "peak_memory_last_gb": round(student_peak_gb, 2),
            "peak_memory_reserved_last_gb": round(student_reserved_gb, 2),
            "peak_memory_max_allocation_gb": round(run_max_gb, 2)
            if run_max_gb is not None
            else round(student_peak_gb, 2),
            "devices": devices,
        }
        if avg_peak_gb is not None and max_peak_gb is not None:
            entry["avg_peak_memory_gb"] = round(avg_peak_gb, 3)
            entry["max_peak_memory_gb"] = round(max_peak_gb, 3)
            entry["avg_peak_memory_steps"] = int(peak_steps)
            if self.logger is not None:
                self.logger.log_scalar(
                    "train/avg_peak_memory_gb", float(avg_peak_gb), self.global_step
                )

        # Build filename with node name and GPU type for easier filtering
        node_name = os.getenv(
            "SLURMD_NODENAME", os.getenv("HOSTNAME", "unknown")
        ).strip()
        # Sanitize node_name to be filesystem-safe
        node_name = node_name.replace("/", "_").replace("\\", "_").replace(":", "_")
        # Get GPU type from the first device (student)
        gpu_type = torch.cuda.get_device_name(self.student_device)
        # Sanitize GPU type for filename (e.g., "NVIDIA GeForce RTX 3090" -> "NVIDIA_GeForce_RTX_3090")
        gpu_type_safe = (
            gpu_type.replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
        )
        peak_filename = f"peak_memory_{node_name}_{gpu_type_safe}.json"
        peak_path = Path("results") / peak_filename
        peak_path.parent.mkdir(parents=True, exist_ok=True)
        existing: list = []
        if peak_path.exists():
            try:
                existing = json.loads(peak_path.read_text())
            except (json.JSONDecodeError, ValueError):
                existing = []
        existing.append(entry)
        peak_path.write_text(json.dumps(existing, indent=2) + "\n")

    def _prepare_offline_cache(self, build_cache_on_rank: bool) -> None:
        cache_ready_flag = bool(getattr(self.config, "_cache_is_ready", False))
        force_hash = getattr(self.config, "offline_cache_force_hash", None)
        selected_ids: Optional[List[int]] = None
        if (
            bool(getattr(self.config, "offline_cache_selected_only", False))
            and self.sample_skipper is not None
        ):
            selected_raw = self.sample_skipper.selected_sample_ids
            if selected_raw:
                selected_ids = sorted(int(sid) for sid in selected_raw)
                setattr(
                    self.config,
                    "_offline_cache_selection_hash",
                    self._hash_selected_ids(selected_ids),
                )
                setattr(
                    self.config, "_offline_cache_selection_count", len(selected_ids)
                )
        if build_cache_on_rank:
            if self.cache is None:
                self.cache = init_offline_cache_for_trainer(
                    getattr(self.config, "offline_cache_dir", None),
                    self.compute_cache_signature(),
                    override_hash=force_hash,
                )
            if force_hash:
                missing_tolerance = int(
                    getattr(self.config, "offline_cache_missing_tolerance", 100) or 100
                )
                expected_items = getattr(self.config, "_expected_dataset_items", None)
                selection_count = getattr(
                    self.config, "_offline_cache_selection_count", None
                )
                if selection_count is not None:
                    expected_items = int(selection_count)
                elif expected_items is None or expected_items < 0:
                    try:
                        expected_items = int(len(self.dataloader.dataset))
                    except Exception:
                        expected_items = None
                try:
                    manifest_items = int(
                        len(getattr(self.cache, "manifest", {}).get("items", {}))
                    )
                except Exception:
                    manifest_items = 0
                if expected_items is not None and expected_items >= 0:
                    missing_items = max(int(expected_items) - manifest_items, 0)
                    if missing_items > 0:
                        if missing_items > missing_tolerance:
                            raise RuntimeError(
                                f"Force-hash cache is missing {missing_items}/{expected_items} items "
                                f"(tolerance={missing_tolerance}). Refuse to train to avoid skipping most data. "
                                "Rebuild the cache or raise --offline_cache_missing_tolerance/ OFFLINE_CACHE_MISSING_TOLERANCE."
                            )
                        setattr(self.config, "_allow_partial_offline_cache", True)
                        setattr(
                            self.config,
                            "_offline_cache_missing_items",
                            int(missing_items),
                        )
                        setattr(
                            self.config,
                            "_offline_cache_missing_tolerance",
                            int(missing_tolerance),
                        )
                        if (not self.ddp_enabled) or (self.ddp_rank == 0):
                            print(
                                f"[logits-cache] Force-hash cache missing {missing_items}/{expected_items} items; "
                                f"skipping those samples during training (tolerance={missing_tolerance})."
                            )
                hit_rate = self._probe_cache_hit_rate()
                if hit_rate is not None:
                    min_hit = float(
                        getattr(self.config, "offline_cache_min_hit_rate", 0.9) or 0.9
                    )
                    if hit_rate < min_hit:
                        raise RuntimeError(
                            f"Force-hash cache appears mismatched: hit_rate={hit_rate:.3f} < {min_hit:.2f}. "
                            "This would skip most batches (no training steps logged). "
                            "Rebuild the cache for the current dataset/tokenizer or lower "
                            "--offline_cache_min_hit_rate/ OFFLINE_CACHE_MIN_HIT_RATE."
                        )
            if self.teacher_available:
                try:
                    cache_dataset = self.dataloader.dataset
                    if selected_ids:
                        cache_dataset = Subset(cache_dataset, selected_ids)
                    cache_batch_size = getattr(
                        self.config, "offline_cache_batch_size", None
                    )
                    if cache_batch_size is None or cache_batch_size <= 0:
                        cache_batch_size = self.config.batch_size
                    cache_batch_size = int(cache_batch_size)
                    dl_for_cache = DataLoader(
                        cache_dataset,
                        batch_size=cache_batch_size,
                        shuffle=False,
                        collate_fn=self.dataloader.collate_fn,
                        num_workers=0,
                        pin_memory=False,
                        persistent_workers=False,
                    )
                except Exception:
                    dl_for_cache = self.dataloader

                self.cache = build_offline_cache_if_needed(
                    cache=self.cache,
                    teacher=self.teacher,
                    tok=self.tok,
                    dataloader=dl_for_cache,
                    config=self.config,
                    teacher_device=self.teacher_device,
                    sanitize_logits_fn=self._sanitize_logits,
                )
            elif not cache_ready_flag:
                raise RuntimeError(
                    "Teacherless run requires an existing offline cache. Provide a cache built with the same signature."
                )

        if self.ddp_enabled and dist.is_available() and dist.is_initialized():
            dist.barrier()
            if not build_cache_on_rank:
                self.cache = init_offline_cache_for_trainer(
                    getattr(self.config, "offline_cache_dir", None),
                    self.compute_cache_signature(),
                    override_hash=force_hash,
                )
                if not self.teacher_available and not cache_ready_flag:
                    raise RuntimeError(
                        "Offline cache not ready on non-building rank and teacher unavailable."
                    )

    def _reset_bandit_state(self) -> None:
        if self.bandit_manager is not None:
            # Guard against stale pending batches when resuming training or restarting loops.
            self.bandit_manager.reset()

    def _maybe_create_profiler(self, rank_is_zero: bool):
        if not rank_is_zero:
            return None
        if not bool(getattr(self.config, "profiler_enabled", False)):
            return None
        try:
            from torch.profiler import (
                profile,
                ProfilerActivity,
                schedule,
                tensorboard_trace_handler,
            )
        except Exception as exc:
            print(
                f"[profiler] Unable to import torch.profiler ({exc}); disabling.",
                flush=True,
            )
            return None

        run_label = os.getenv("RUN_DISPLAY_NAME", "").strip()
        job_name = os.getenv("JOB_NAME", "").strip()
        slurm_job_id = os.getenv("SLURM_JOB_ID", "").strip()
        tag = (
            job_name or run_label or (f"job_{slurm_job_id}" if slurm_job_id else "run")
        )
        tag = tag.replace(" ", "_")
        base_dir = getattr(self.config, "profiler_dir", None) or os.path.join(
            "results", "gpu_util", tag
        )
        Path(base_dir).mkdir(parents=True, exist_ok=True)

        wait = int(getattr(self.config, "profiler_wait", 1))
        warmup = int(getattr(self.config, "profiler_warmup", 1))
        active = int(getattr(self.config, "profiler_active", 3))
        repeat = int(getattr(self.config, "profiler_repeat", 1))
        record_shapes = bool(getattr(self.config, "profiler_record_shapes", True))
        with_stack = bool(getattr(self.config, "profiler_with_stack", True))
        profile_memory = bool(getattr(self.config, "profiler_profile_memory", True))

        print(
            f"[profiler] Enabled. Traces -> {base_dir} "
            f"(wait={wait}, warmup={warmup}, active={active}, repeat={repeat})",
            flush=True,
        )
        self._profiler_dir = base_dir  # Store for later PNG generation
        activities = [ProfilerActivity.CPU]
        if self.student_device.type == "cuda" and torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        return profile(
            activities=activities,
            schedule=schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=tensorboard_trace_handler(base_dir),
            record_shapes=record_shapes,
            with_stack=with_stack,
            profile_memory=profile_memory,
        )

    def _generate_memory_profile_plots(self) -> None:
        """Generate memory profile PNG plots from profiler traces."""
        profiler_dir = getattr(self, "_profiler_dir", None)
        if not profiler_dir:
            return
        profiler_path = Path(profiler_dir)
        if not profiler_path.exists():
            return

        trace_files = list(profiler_path.glob("*.trace.json"))
        if not trace_files:
            return

        try:
            # Import the plotting module
            import sys

            tools_dir = Path(__file__).parent.parent.parent / "tools"
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))

            from plot_memory_profile import process_trace_file

            print(
                f"[profiler] Generating memory profile PNGs from {len(trace_files)} trace(s)...",
                flush=True,
            )
            for trace_file in trace_files:
                process_trace_file(trace_file, profiler_path)
        except Exception as e:
            print(
                f"[profiler] Warning: Could not generate memory plots: {e}", flush=True
            )

    def _probe_cache_hit_rate(self, sample_size: int = 128) -> Optional[float]:
        if self.cache is None:
            return None
        dataset = getattr(self.dataloader, "dataset", None)
        if dataset is None:
            return None
        try:
            total = min(int(sample_size), int(len(dataset)))
        except Exception:
            return None
        if total <= 0:
            return None
        hits = 0
        for idx in range(total):
            try:
                item = dataset[idx]
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            input_ids = item.get("input_ids")
            if input_ids is None:
                continue
            if not torch.is_tensor(input_ids):
                try:
                    input_ids = torch.tensor(input_ids)
                except Exception:
                    continue
            key = TeacherOfflineCache.key_from_ids(input_ids)
            if self.cache.has(key):
                hits += 1
        if total <= 0:
            return None
        return hits / float(total)

    def _create_kd_schedule(self, epochs: int) -> KDTemperatureSchedule:
        # Prepare KD temperature annealing schedule (in units of optimizer updates)
        updates_per_epoch = math.ceil(
            len(self.dataloader) / max(1, self.config.gradient_accumulation_steps)
        )
        total_updates = updates_per_epoch * max(1, epochs)
        kd_schedule = KDTemperatureSchedule(self.config, total_updates)
        return kd_schedule

    def _run_training_epochs(
        self,
        epochs: int,
        log_every: int,
        rank_is_zero: bool,
        kd_schedule: KDTemperatureSchedule,
    ) -> None:
        # Track OOM failures: consecutive and total
        consecutive_oom_failures = 0
        total_oom_failures = 0
        max_consecutive_ooms = 10
        max_total_oom_percent = 50  # Abort if >50% of attempted forwards OOM
        oom_failure_sleep_s = 100
        total_batch_attempts = 0
        self._kl_zero_counter = 0
        self._debug_forward_calls = 0

        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= epochs:
            if rank_is_zero:
                print(
                    f"[resume] start_epoch={start_epoch} >= epochs={epochs}; skipping training."
                )
            return

        profiler = self._maybe_create_profiler(rank_is_zero)
        profiler_steps = 0
        profiler_max_steps = getattr(self.config, "profiler_max_steps", 200)

        def _profiler_step() -> None:
            nonlocal profiler_steps
            if profiler is None:
                return
            profiler_steps += 1
            if profiler_max_steps is None or profiler_steps <= int(profiler_max_steps):
                try:
                    profiler.step()
                except Exception:
                    pass

        if profiler is not None:
            profiler.__enter__()
        try:
            for epoch in range(start_epoch, epochs):
                step_start = time.time()
                running = {"loss": 0.0, "kl": 0.0, "ce": 0.0}
                bandit_running: Dict[str, float] = {}
                bandit_steps = 0
                last_reward_metrics: Optional[Dict[str, float]] = None
                self.opt.zero_grad(set_to_none=True)  # Initialize gradients

                if self.ddp_enabled:
                    sampler = getattr(self.dataloader, "sampler", None)
                    if sampler is not None and hasattr(sampler, "set_epoch"):
                        sampler.set_epoch(epoch)
                    else:
                        batch_sampler = getattr(self.dataloader, "batch_sampler", None)
                        if batch_sampler is not None and hasattr(
                            batch_sampler, "set_epoch"
                        ):
                            batch_sampler.set_epoch(epoch)

                step = 0  # Manual step counter that only increments on success
                resume_step = (
                    int(getattr(self, "_resume_step", 0)) if epoch == start_epoch else 0
                )
                if resume_step > 0 and rank_is_zero:
                    print(
                        f"[resume] Skipping {resume_step} batches in epoch {epoch + 1} to resume."
                    )
                for batch in self.dataloader:
                    if resume_step > 0 and step < resume_step:
                        step += 1
                        _profiler_step()
                        continue
                    attempt_retries = 0
                    skipped_this_batch = False
                    loss = torch.tensor(0.0, device=self.student_device)
                    kl_val = 0.0
                    ce_val = 0.0
                    bandit_metrics = None
                    _log_peak_mem = bool(getattr(self.config, "log_peak_memory", False))
                    while True:
                        total_batch_attempts += 1
                        try:
                            # Reset peak memory stats before forward+backward
                            if _log_peak_mem and self.student_device.type == "cuda":
                                torch.cuda.reset_peak_memory_stats(self.student_device)

                            loss, kl_val, ce_val, bandit_metrics = self._forward_batch(
                                batch
                            )
                            # Scale loss by gradient accumulation steps
                            loss = loss / self.config.gradient_accumulation_steps

                            # ===== OOM Reduction: AMP backward pass =====
                            scaler = getattr(self, "_scaler", None)
                            if scaler is not None:
                                # fp16: use scaler
                                scaler.scale(loss).backward()
                            else:
                                # bfloat16 or no AMP: standard backward
                                loss.backward()

                            # Capture peak memory after backward
                            if _log_peak_mem and self.student_device.type == "cuda":
                                peak_gb = torch.cuda.max_memory_allocated(
                                    self.student_device
                                ) / (1024**3)
                                self._peak_memory_samples.append(peak_gb)
                                if self.logger and rank_is_zero:
                                    self.logger.log_scalar(
                                        "train/peak_memory_gb",
                                        peak_gb,
                                        self.global_step,
                                    )

                            # Reset consecutive OOM counter on success
                            consecutive_oom_failures = 0
                            step += 1  # Only increment step on successful batch
                            break
                        except SkipBatch:
                            skipped_this_batch = True
                            break
                        except torch.cuda.OutOfMemoryError:
                            attempt_retries += 1
                            consecutive_oom_failures += 1
                            total_oom_failures += 1
                            # Clear partial grads and GPU caches
                            try:
                                self.opt.zero_grad(set_to_none=True)
                            except Exception:
                                pass
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                            try:
                                import gc

                                gc.collect()
                            except Exception:
                                pass

                            oom_rate = (
                                (total_oom_failures / total_batch_attempts) * 100
                                if total_batch_attempts > 0
                                else 0.0
                            )
                            if rank_is_zero:
                                print(
                                    f"[OOM][rank {self.ddp_rank}] CUDA out of memory at epoch {epoch + 1}, step {step + 1}. "
                                    f"Consecutive: {consecutive_oom_failures}/{max_consecutive_ooms}, "
                                    f"Total: {total_oom_failures}/{total_batch_attempts} ({oom_rate:.1f}%). "
                                    f"Retrying batch (attempt {attempt_retries + 1})..."
                                )

                            # Exit if too many retries for this batch or globally
                            if attempt_retries >= max_consecutive_ooms and rank_is_zero:
                                print(
                                    f"[OOM] Reached {max_consecutive_ooms} consecutive OOM failures on the same batch. "
                                    f"Sleeping {oom_failure_sleep_s}s before exiting..."
                                )
                                time.sleep(oom_failure_sleep_s)
                                raise RuntimeError(
                                    f"Training aborted after {max_consecutive_ooms} consecutive CUDA OOM failures on one batch."
                                )

                            if (
                                total_batch_attempts >= 20
                                and oom_rate > max_total_oom_percent
                                and rank_is_zero
                            ):
                                print(
                                    f"[OOM] OOM rate ({oom_rate:.1f}%) exceeds threshold ({max_total_oom_percent}%). "
                                    f"Sleeping {oom_failure_sleep_s}s before exiting..."
                                )
                                time.sleep(oom_failure_sleep_s)
                                raise RuntimeError(
                                    f"Training aborted: {total_oom_failures}/{total_batch_attempts} forward attempts failed with OOM ({oom_rate:.1f}%)."
                                )

                            # Retry the same batch with cleared caches
                            continue

                    if skipped_this_batch:
                        _profiler_step()
                        continue

                    self._total_train_iterations += 1
                    self._total_train_tokens += self._count_batch_tokens(batch)

                    # Only update weights after accumulation steps
                    if step % self.config.gradient_accumulation_steps == 0:
                        # ===== OOM Reduction: AMP optimizer step =====
                        scaler = getattr(self, "_scaler", None)
                        if scaler is not None:
                            # fp16: unscale, clip, step, update scaler
                            scaler.unscale_(self.opt)
                            torch.nn.utils.clip_grad_norm_(
                                self.student.parameters(), 1.0
                            )
                            scaler.step(self.opt)
                            scaler.update()
                        else:
                            # bfloat16 or no AMP: standard step
                            torch.nn.utils.clip_grad_norm_(
                                self.student.parameters(), 1.0
                            )
                            self.opt.step()

                        self.opt.zero_grad(set_to_none=True)
                        self.global_step += 1
                        # Update KD temperature per schedule if enabled
                        if bool(getattr(self.config, "anneal_kd_temperature", False)):
                            self.config.kd_temperature = kd_schedule.kd_T_at(
                                self.global_step
                            )
                            # Log the current KD temperature to visualize the annealing schedule
                            if self.logger and rank_is_zero:
                                self.logger.log_scalar(
                                    "train/kd_temperature",
                                    float(self.config.kd_temperature),
                                    self.global_step,
                                )
                        reward_metrics = None
                        if self.bandit_manager is not None and self.teacher_available:
                            # Consume queued actions collected during forward passes and update the bandit.
                            reward_metrics = self.bandit_manager.process_rewards(
                                self.student, self.teacher
                            )
                        if reward_metrics:
                            last_reward_metrics = reward_metrics
                            if self.logger and rank_is_zero:
                                self.logger.log(reward_metrics, self.global_step)

                        # Save checkpoint if needed
                        if (
                            self.config.checkpoint_steps > 0
                            and self.global_step % self.config.checkpoint_steps == 0
                            and rank_is_zero
                        ):
                            self.save_checkpoint(epoch, step)
                        # Stop if wall-clock limit reached
                        self._maybe_handle_time_limit(epoch, step, rank_is_zero)

                    # logging
                    running["loss"] += (
                        loss.item() * self.config.gradient_accumulation_steps
                    )  # Unscale for logging
                    running["kl"] += kl_val
                    running["ce"] += ce_val
                    if bandit_metrics:
                        # These metrics describe the current batch's bandit decisions (e.g., token counts).
                        for key, value in bandit_metrics.items():
                            bandit_running[key] = bandit_running.get(key, 0.0) + value
                        bandit_steps += 1

                    # Logging every step using TrainingMetrics
                    metrics = TrainingMetrics(
                        loss=loss.item() * self.config.gradient_accumulation_steps,
                        kl_loss=kl_val,
                        ce_loss=ce_val,
                        epoch=epoch + 1,
                        step=step + 1,
                        global_step=self.global_step,
                    )

                    # Log metrics
                    if self.logger and rank_is_zero:
                        log_metrics = {
                            **metrics.to_dict(),
                            "train/step": step + 1,
                            "train/global_step": self.global_step,
                        }
                        if bandit_metrics:
                            log_metrics.update(bandit_metrics)
                        self.logger.log(log_metrics, self.global_step)

                    if (step + 1) % log_every == 0:
                        n = log_every
                        avg_loss = running["loss"] / n
                        avg_kl = running["kl"] / n
                        avg_ce = running["ce"] / n
                        avg_bandit: Dict[str, float] = {}
                        if bandit_steps > 0:
                            avg_bandit = {
                                k: v / bandit_steps for k, v in bandit_running.items()
                            }

                        elapsed = time.time() - step_start
                        step_start = time.time()
                        bandit_str = ""
                        if avg_bandit:
                            sel = avg_bandit.get("bandit/selected_tokens", 0.0)
                            overlap = avg_bandit.get("bandit/overlap_selected", 0.0)
                            bandit_str = (
                                f" | bandit_sel={sel:.2f} overlap={overlap:.2f}"
                            )
                        reward_str = ""
                        if last_reward_metrics:
                            avg_reward = last_reward_metrics.get(
                                "bandit/avg_reward", 0.0
                            )
                            reward_str = f" | bandit_reward={avg_reward:.4f}"
                        if rank_is_zero:
                            print(
                                f"ep{epoch + 1} step{step + 1} | "
                                f"loss={avg_loss:.4f} kl={avg_kl:.4f} ce={avg_ce:.4f} "
                                f"| global_step={self.global_step}{bandit_str}{reward_str} | {elapsed:.2f}s total, {elapsed / log_every:.2f}s/step"
                            )

                        # Log averages using new combined logger or legacy loggers
                        avg_metrics = {
                            "train/avg_loss": avg_loss,
                            "train/avg_kl_loss": avg_kl,
                            "train/avg_ce_loss": avg_ce,
                            "train/elapsed_time": elapsed,
                            "train/steps_per_second": log_every / elapsed,
                        }
                        if avg_bandit:
                            avg_metrics.update(avg_bandit)
                        if last_reward_metrics:
                            avg_metrics.update(last_reward_metrics)

                        # Log averages
                        if self.logger and rank_is_zero:
                            self.logger.log(avg_metrics, self.global_step)
                            self.logger.flush()

                        running = {k: 0.0 for k in running}
                        bandit_running = {}
                        bandit_steps = 0

                    _profiler_step()
        finally:
            if profiler is not None:
                try:
                    profiler.__exit__(None, None, None)
                except Exception:
                    pass
                # Generate memory profile PNGs from traces
                self._generate_memory_profile_plots()


__all__ = ["Distiller"]
