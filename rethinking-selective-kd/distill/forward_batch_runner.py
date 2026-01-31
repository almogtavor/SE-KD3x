from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import device
from torch.optim import AdamW, Optimizer

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
from .oom_setup_helper import OOMSetupHelper
from .selective_lm_head import (
    compute_student_entropy_and_select,
    selective_student_logits,
    selective_teacher_logits,
)


class SkipBatch(RuntimeError):
    """Raised to skip processing a batch entirely (no forward/backward)."""


class ForwardBatchRunner:
    def __init__(self, distiller, batch):
        self.distiller = distiller
        self.batch = batch
        self.extra_metrics: Optional[Dict[str, float]] = None
        self.ce_loss_override: Optional[torch.Tensor] = None
        self.ce_selection_mask: Optional[torch.Tensor] = None
        self.ce_all_tokens = bool(getattr(distiller.config, "enable_ce_on_all_tokens", False))

    def run(self) -> Tuple[torch.Tensor, float, float, Optional[Dict[str, float]]]:
        skipper = getattr(self.distiller, "sample_skipper", None)
        filtered = skipper.maybe_filter_batch(self.batch) if skipper is not None else self.batch
        if filtered is None:
            raise SkipBatch("all samples in batch were filtered")
        self.batch = filtered
        self._maybe_filter_cache_missing_samples()
        self._init_debug()
        self._move_inputs()
        self._setup_temperature()
        self._setup_amp()

        # Branch: selective lm_head flow (avoids full [B,T,V] logits tensors)
        if self._should_use_selective_lm_head():
            return self._run_selective_lm_head_flow()

        self._student_forward()
        self._setup_cache_state()
        self._maybe_run_teacher_forward()
        self._compute_kd_loss()
        self._combine_losses()
        self._final_checks()
        return self.total, self.kd_loss_scalar, self.ce_loss_scalar, self.extra_metrics

    def _maybe_filter_cache_missing_samples(self) -> None:
        if not bool(getattr(self.distiller.config, "offline_cache", False)):
            return
        if not bool(getattr(self.distiller.config, "_allow_partial_offline_cache", False)):
            return
        if self.distiller.cache is None:
            return
        input_ids = self.batch.get("input_ids") if isinstance(self.batch, dict) else None
        if input_ids is None:
            return
        if not torch.is_tensor(input_ids):
            return

        missing: List[int] = []
        keep: List[int] = []
        batch_size = int(input_ids.size(0))
        for i in range(batch_size):
            key = TeacherOfflineCache.key_from_ids(input_ids[i])
            if self.distiller.cache.has(key):
                keep.append(i)
            else:
                missing.append(i)

        if not missing:
            return

        if not keep:
            raise SkipBatch("all samples in batch were missing from the offline cache")

        if not getattr(self.distiller, "_printed_cache_missing_skip", False):
            if (not self.distiller.ddp_enabled) or (self.distiller.ddp_rank == 0):
                missing_total = int(getattr(self.distiller.config, "_offline_cache_missing_items", 0) or 0)
                tolerance = int(getattr(self.distiller.config, "_offline_cache_missing_tolerance", 0) or 0)
                print(
                    f"[logits-cache] Skipping samples missing from cache "
                    f"({missing_total} <= {tolerance} total missing)."
                )
            setattr(self.distiller, "_printed_cache_missing_skip", True)

        self.batch = self._filter_batch_by_indices(self.batch, keep, batch_size)

    @staticmethod
    def _filter_batch_by_indices(batch: Dict[str, Any], keep: List[int], batch_size: int) -> Dict[str, Any]:
        if not keep:
            return batch
        keep_idx = torch.tensor(keep, dtype=torch.long)
        filtered: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and value.size(0) == batch_size:
                filtered[key] = value.index_select(0, keep_idx)
            elif isinstance(value, list) and len(value) == batch_size:
                filtered[key] = [value[i] for i in keep]
            else:
                filtered[key] = value
        return filtered

    def _should_use_selective_lm_head(self) -> bool:
        """Check if either selective lm_head flag is set."""
        cfg = self.distiller.config
        sel_flags = (
            bool(getattr(cfg, "teacher_selective_lm_head", False))
            or bool(getattr(cfg, "student_selective_lm_head", False))
            or bool(getattr(cfg, "selective_lm_head_same_flow", False))
        )
        if not sel_flags:
            return False
        if bool(getattr(cfg, "offline_cache", False)):
            raise RuntimeError(
                "offline_cache=True is incompatible with selective_lm_head flows because they require an online teacher. "
                "Disable selective LM head or disable offline_cache."
            )
        if not self.distiller.teacher_available:
            return False
        return True

    def _run_selective_lm_head_flow(self) -> Tuple[torch.Tensor, float, float, Optional[Dict[str, float]]]:
        """Memory-efficient flow with independently controlled teacher/student selective lm_head."""
        cfg = self.distiller.config
        sel_teacher = bool(getattr(cfg, "teacher_selective_lm_head", False))
        sel_student = bool(getattr(cfg, "student_selective_lm_head", False))

        t_start = time.perf_counter()

        student_base = self.distiller._student_base
        if not hasattr(student_base, "model"):
            raise RuntimeError("Student model must have .model and .lm_head attributes for selective lm_head.")
        student_transformer = student_base.model
        student_lm_head = student_base.lm_head

        # Step 1: Run student base transformer (with grad)
        t_student_base_start = time.perf_counter()
        with self.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            student_outputs = student_transformer(
                self.input_ids_s,
                attention_mask=self.attn_mask_s,
                output_hidden_states=False,
            )
            student_hidden_states = student_outputs.last_hidden_state  # [B, T, D]
        t_student_base = time.perf_counter() - t_student_base_start

        # Step 2: Compute student entropy (no grad) and select top-k%
        t_entropy_start = time.perf_counter()
        k_percent = float(cfg.k_percent)
        normalize_topk = bool(getattr(cfg, "normalize_topk_by_length", False))
        chunk_size = int(getattr(cfg, "entropy_streaming_chunk_size", 128))
        selected_mask, _ = compute_student_entropy_and_select(
            student_hidden_states=student_hidden_states,
            lm_head=student_lm_head,
            valid_next=self.valid_next,
            k_percent=k_percent,
            normalize_topk_by_length=normalize_topk,
            chunk_size=chunk_size,
        )
        t_entropy = time.perf_counter() - t_entropy_start

        n_selected = int(selected_mask.sum().item())
        if n_selected == 0:
            return student_hidden_states.sum() * 0.0, 0.0, 0.0, None

        # Step 3: Student logits — selective or full
        t_student_logits_start = time.perf_counter()
        alpha_ce = float(cfg.alpha_ce) if cfg.enable_ce else 0.0
        needs_full_hidden_states = alpha_ce > 0.0 and self.ce_all_tokens and sel_student
        
        if sel_student:
            with self.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                s_logits_for_kd = selective_student_logits(
                    student_hidden_states=student_hidden_states,
                    lm_head=student_lm_head,
                    selected_mask=selected_mask,
                )  # [N_sel, V]
            # Free hidden states early if not needed for CE loss (saves ~2-3 GB for large models)
            if not needs_full_hidden_states:
                del student_hidden_states
        else:
            with self.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                s_full_logits = student_lm_head(student_hidden_states[:, :-1, :])  # [B, T-1, V]
            batch_idx, pos_idx = torch.nonzero(selected_mask, as_tuple=True)
            s_logits_for_kd = s_full_logits[batch_idx, pos_idx]  # [N_sel, V]
        t_student_logits = time.perf_counter() - t_student_logits_start
        s_logits_for_kd = self.distiller._sanitize_logits(s_logits_for_kd, "student")

        # Step 4: Teacher logits — selective or full
        t_teacher_logits_start = time.perf_counter()
        if sel_teacher:
            t_logits_for_kd = selective_teacher_logits(
                teacher_model=self.distiller.teacher,
                input_ids=self.input_ids,
                attention_mask=self.attn_mask,
                selected_mask=selected_mask,
                teacher_device=torch.device(self.distiller.teacher_device),
                amp_enabled=self.amp_enabled,
                amp_dtype=self.amp_dtype,
            )  # [N_sel, V_teacher]
        else:
            with torch.no_grad():
                with self.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                    t_dev = torch.device(self.distiller.teacher_device)
                    t_full = self.distiller.teacher(
                        self.input_ids.to(t_dev),
                        attention_mask=self.attn_mask.to(t_dev),
                ).logits[:, :-1, :]  # [B, T-1, V_teacher]
            batch_idx, pos_idx = torch.nonzero(selected_mask.to(t_full.device), as_tuple=True)
            t_logits_for_kd = t_full[batch_idx, pos_idx]  # [N_sel, V_teacher]
        t_teacher_logits = time.perf_counter() - t_teacher_logits_start
        t_logits_for_kd = self.distiller._sanitize_logits(t_logits_for_kd, "teacher")
        t_logits_for_kd = t_logits_for_kd.to(self.distiller.student_device)

        # Step 5: KD loss on selected positions
        # Use F.kl_div with log_target=True to avoid materializing intermediate prob tensors
        T, T2 = self.T, self.T2
        s_log_probs = torch.log_softmax(s_logits_for_kd.float() / T, dim=-1)
        t_log_probs = torch.log_softmax(t_logits_for_kd.float() / T, dim=-1)

        use_fused_kl = bool(getattr(cfg, "fused_kl_loss", True))
        if use_fused_kl:
            # Memory-efficient: F.kl_div avoids creating intermediate probability tensors
            if getattr(cfg, "kd_objective", "forward") == "reverse":
                # KL(student || teacher) = sum(student * (log_student - log_teacher))
                kd_loss = F.kl_div(t_log_probs, s_log_probs, log_target=True, reduction='batchmean') * T2
            else:
                # KL(teacher || student) = sum(teacher * (log_teacher - log_student))
                kd_loss = F.kl_div(s_log_probs, t_log_probs, log_target=True, reduction='batchmean') * T2
        else:
            # Original implementation (kept for backward compatibility / debugging)
            if getattr(cfg, "kd_objective", "forward") == "reverse":
                s_probs = s_log_probs.exp()
                kd_per_pos = (s_probs * (s_log_probs - t_log_probs)).sum(dim=-1)
            else:
                t_probs = t_log_probs.exp()
                kd_per_pos = (t_probs * (t_log_probs - s_log_probs)).sum(dim=-1)
            kd_loss = kd_per_pos.mean() * T2

        if self.log_debug:
            kd_val = float(kd_loss.detach().item())
            print(
                f"[DEBUG] _forward_batch: kd_loss (after T^2)={kd_val:.10f}, T={T}, T2={T2}",
                flush=True,
            )

        # Step 6: CE loss (alpha_ce already computed earlier in Step 3)
        if alpha_ce > 0.0:
            if self.ce_all_tokens:
                if sel_student:
                    with self.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
                        ce_logits = student_lm_head(student_hidden_states[:, :-1, :])
                else:
                    ce_logits = s_full_logits
                ce_logits = self.distiller._sanitize_logits(ce_logits, "student")
                targets = self.input_ids_s[:, 1:]
                V = ce_logits.size(-1)
                targets = targets.clamp(min=-1, max=V - 1).masked_fill(~self.ce_mask, -100)
                ce_loss = F.nll_loss(
                    torch.log_softmax(ce_logits.float(), dim=-1).reshape(-1, V),
                    targets.reshape(-1).long(), ignore_index=-100, reduction="mean")
            else:
                batch_idx, pos_idx = torch.nonzero(selected_mask, as_tuple=True)
                targets_sel = self.input_ids_s[:, 1:][batch_idx, pos_idx]
                V = s_logits_for_kd.size(-1)
                targets_sel = targets_sel.clamp(min=0, max=V - 1)
                ce_loss = F.nll_loss(
                    torch.log_softmax(s_logits_for_kd.float(), dim=-1),
                    targets_sel.long(), reduction="mean")
            use_unbounded = bool(getattr(cfg, "unbounded_to_1_loss", False))
            total = (kd_loss + ce_loss) if use_unbounded else ((1.0 - alpha_ce) * kd_loss + alpha_ce * ce_loss)
            ce_val = float(ce_loss.item())
        else:
            total = kd_loss
            ce_val = 0.0

        if not torch.isfinite(total) or not torch.isfinite(kd_loss):
            return s_logits_for_kd.sum() * 0.0, 0.0, 0.0, None

        t_total = time.perf_counter() - t_start
        if self.distiller.logger:
            self.distiller.logger.log(
                {
                    "train/selective_lm_head_n_selected": float(n_selected),
                    "train/selective_lm_head_student_base_time_s": t_student_base,
                    "train/selective_lm_head_entropy_time_s": t_entropy,
                    "train/selective_lm_head_student_logits_time_s": t_student_logits,
                    "train/selective_lm_head_teacher_logits_time_s": t_teacher_logits,
                    "train/selective_lm_head_total_time_s": t_total,
                    "train/selective_lm_head_student_enabled": float(sel_student),
                    "train/selective_lm_head_teacher_enabled": float(sel_teacher),
                },
                self.distiller.global_step,
            )
        return total, float(kd_loss.item()), ce_val, None

    def _init_debug(self) -> None:
        debug_interval = getattr(self.distiller, "_debug_log_interval", 0)
        self.distiller._debug_forward_calls += 1
        self.log_debug = False
        if debug_interval > 0:
            if self.distiller._debug_forward_calls == 1 or (self.distiller._debug_forward_calls % debug_interval == 0):
                self.log_debug = True

    def _move_inputs(self) -> None:
        batch = self.batch
        self.input_ids = batch["input_ids"]
        self.attn_mask = batch["attention_mask"]
        self.kd_mask_tensor = batch.get("kd_mask")
        self.input_ids_s = self.input_ids.to(self.distiller.student_device)
        self.attn_mask_s = self.attn_mask.to(self.distiller.student_device)
        self.kd_mask_s = None
        if self.kd_mask_tensor is not None:
            self.kd_mask_s = self.kd_mask_tensor.to(self.distiller.student_device)
            if self.kd_mask_s.dtype != torch.bool:
                self.kd_mask_s = self.kd_mask_s.bool()
        self.ce_mask = self.attn_mask_s[:, 1:].bool()
        self.valid_next = self.ce_mask.clone()
        if self.kd_mask_s is not None:
            self.valid_next = self.valid_next & self.kd_mask_s[:, 1:].bool()

    def _setup_temperature(self) -> None:
        self.T = float(getattr(self.distiller.config, "kd_temperature", 1.0))
        self.T2 = self.T * self.T

    def _setup_amp(self) -> None:
        from torch.cuda.amp import autocast

        self.autocast = autocast
        self.amp_enabled = getattr(self.distiller, "_use_amp", False)
        self.amp_dtype = getattr(self.distiller, "_amp_dtype", torch.float32)
        self.rank_is_zero = (not self.distiller.ddp_enabled) or (self.distiller.ddp_rank == 0)

    def _student_forward(self) -> None:
        with self.autocast(enabled=self.amp_enabled, dtype=self.amp_dtype):
            s_logits = self.distiller.student(self.input_ids_s, attention_mask=self.attn_mask_s).logits
        self.s_logits = self.distiller._sanitize_logits(s_logits, "student")
        self.s_pred = self.s_logits[:, :-1, :]
        self.s_log_probs: Optional[torch.Tensor] = None

    def _setup_cache_state(self) -> None:
        offline_cache = bool(getattr(self.distiller.config, "offline_cache", False))
        self.cached_items = self.distiller._lookup_cache_batch(self.input_ids) if offline_cache else None
        self.cache_mode = self.distiller._cache_mode() or "entropy"
        self.cached_target_probs: Optional[torch.Tensor] = None
        self.ids_U: Optional[torch.Tensor] = None
        self.probs_U: Optional[torch.Tensor] = None
        self.rs_batch_idx: Optional[torch.Tensor] = None
        self.rs_pos_idx: Optional[torch.Tensor] = None

        if not self.distiller.teacher_available:
            if not offline_cache:
                raise RuntimeError("Teacher-less training requires offline_cache=True.")
            if self.cached_items is None:
                raise RuntimeError(
                    "Teacher is unavailable but the offline cache is missing entries for this batch. Rebuild the cache before training."
                )

        self.distill_type = getattr(self.distiller.config, "distill_type", "vanilla")
        self.score_enabled_flag = bool(getattr(self.distiller.config, "score_token_selection", False))
        cache_only_supported = {
            "vanilla",
            "top-k-tok",
            "top-k-tok-dkd",
            "bucket",
            "random",
            "random-dkd",
            "dkd",
            "pos-rs-kd-dkd",
            "linucb",
            "linucb-dkd",
        }
        if self.cache_mode == "unc":
            cache_only_supported.add("atkd")
        self.use_vocab_rs_kd = bool(offline_cache)
        if self.distill_type in {"dkd", "top-k-tok-dkd", "random-dkd", "pos-rs-kd-dkd"} and not (self.distiller.teacher_available or self.use_vocab_rs_kd):
            raise RuntimeError(f"distill_type='{self.distill_type}' requires teacher logits from cache or an online teacher.")
        if not self.distiller.teacher_available:
            if self.distill_type == "atkd" and self.cache_mode != "unc":
                raise RuntimeError(
                    "distill_type='atkd' without a teacher requires offline_cache_mode='unc' so cached target probabilities are available."
                )
            if self.distill_type not in cache_only_supported:
                raise RuntimeError(
                    f"distill_type='{self.distill_type}' requires a teacher; supported cache-only modes: {sorted(cache_only_supported)}."
                )

        if (
            offline_cache
            and self.cached_items is None
            and self.distiller.teacher_available
            and not getattr(self.distiller, "_cache_build_on_miss_done", False)
        ):
            build_cache_on_rank = (not self.distiller.ddp_enabled) or (self.distiller.ddp_rank == 0)
            self.distiller._prepare_offline_cache(build_cache_on_rank)
            self.distiller._cache_build_on_miss_done = True
            self.cached_items = self.distiller._lookup_cache_batch(self.input_ids)
            self.cache_mode = self.distiller._cache_mode() or self.cache_mode

        if offline_cache and self.cached_items is None:
            raise RuntimeError(
                "offline_cache=True but cached logits are missing for this batch. "
                "Rebuild the offline cache; online teacher fallback is disabled in offline_cache mode."
            )

        self.supports_cached_teacher_logits = (
            self.use_vocab_rs_kd
            and self.cached_items is not None
            and self.distill_type in cache_only_supported
        )

        if offline_cache and not self.supports_cached_teacher_logits:
            raise RuntimeError(
                f"offline_cache=True forbids online teacher forward, but distill_type='{self.distill_type}' "
                f"(cache_mode='{self.cache_mode}') is not supported by cached logits. "
                "Disable offline_cache or use a cache-supported distill_type."
            )

    def _maybe_run_teacher_forward(self) -> None:
        self.t_pred = None
        self.t_log_probs = None
        offline_cache = bool(getattr(self.distiller.config, "offline_cache", False))

        if self.supports_cached_teacher_logits and not self.distiller._printed_cache_info:
            if self.rank_is_zero:
                print("[logits-cache] Using offline cache (vanilla KD on all positions) → computing KD from cache.", flush=True)
            self.distiller._printed_cache_info = True

        if not self.supports_cached_teacher_logits:
            if offline_cache:
                raise RuntimeError(
                    "offline_cache=True forbids online teacher forward. "
                    "Rebuild the offline cache or disable offline_cache."
                )
            if not self.distiller.teacher_available:
                raise RuntimeError(
                    "Offline cache must provide logits for the configured distillation mode; teacher is not available for fallback."
                )
            t_logits = self.distiller._teacher_forward_logits(
                self.input_ids,
                self.attn_mask,
                self.amp_enabled,
                self.amp_dtype,
            )
            self.t_pred = t_logits[:, :-1, :]
            self.t_log_probs = torch.log_softmax((self.t_pred.float() / self.T), dim=-1)
            if not self.distiller._printed_cache_info:
                if self.rank_is_zero:
                    print("[logits-cache] Running online teacher forward.")
                self.distiller._printed_cache_info = True
        else:
            if not self.distiller._printed_cache_info:
                if self.rank_is_zero:
                    print("[logits-cache] Using offline cache → skipping online teacher forward.")
                self.distiller._printed_cache_info = True

    def _compute_kd_loss(self) -> None:
        self._init_kd_loss_state()
        self._dispatch_kd_loss_strategy()

    def _init_kd_loss_state(self) -> None:
        self.extra_metrics = None
        self.ce_selection_mask = None

    def _dispatch_kd_loss_strategy(self) -> None:
        if self._should_use_cached_kd_loss():
            self._kd_loss_with_cache()
        else:
            self._kd_loss_without_cache()

    def _should_use_cached_kd_loss(self) -> bool:
        return self.use_vocab_rs_kd

    def _kd_loss_without_cache(self) -> None:
        self.ce_loss_override = None
        if self.distill_type == "atkd":
            cache_bundle = ATKDCacheBundle(cache_mode="none")
            self.kd_loss, atkd_metrics = compute_atkd_loss(
                config=self.distiller.config,
                student_device=self.distiller.student_device,
                input_ids_s=self.input_ids_s,
                valid_next=self.valid_next,
                student_logits=self.s_pred,
                teacher_logits=self.t_pred,
                cache_bundle=cache_bundle,
            )
            if self.distiller.logger and atkd_metrics:
                self.distiller.logger.log(atkd_metrics, self.distiller.global_step)
        else:
            if self.s_log_probs is None:
                self.s_log_probs = torch.log_softmax((self.s_pred.float() / self.T), dim=-1)
            self.kd_loss, kd_extra = self.distiller._compute_kd_loss(
                self.t_pred,
                self.t_log_probs,
                self.s_pred,
                self.s_log_probs,
                self.valid_next,
                self.input_ids,
                self.attn_mask,
                self.T,
                debug_log=self.log_debug,
            )
            if kd_extra:
                self.extra_metrics = kd_extra
            self.ce_selection_mask = getattr(self.distiller, "_last_ce_selection_mask", None)

    def _kd_loss_with_cache(self) -> None:
        (
            cached_items,
            cache_mode,
            supports_cached_teacher_logits,
            distill_type,
            score_enabled_flag,
            T,
        ) = self._cache_loss_context()
        if cached_items is not None:
            self._handle_cached_kd_loss(
                cached_items,
                cache_mode,
                supports_cached_teacher_logits,
                distill_type,
                score_enabled_flag,
                T,
            )
            return
        self._handle_cache_miss_kd_loss()

    def _cache_loss_context(self):
        cached_items = self.cached_items
        cache_mode = self.cache_mode
        supports_cached_teacher_logits = self.supports_cached_teacher_logits
        distill_type = self.distill_type
        score_enabled_flag = self.score_enabled_flag
        T = self.T
        return cached_items, cache_mode, supports_cached_teacher_logits, distill_type, score_enabled_flag, T

    def _handle_cached_kd_loss(
        self,
        cached_items,
        cache_mode,
        supports_cached_teacher_logits,
        distill_type,
        score_enabled_flag,
        T,
    ) -> None:
        """Handle KD loss computation using offline cache with sampled teacher vocabulary."""
        V = int(self.s_pred.size(-1))

        valid_mask = self.valid_next
        batch_idx, pos_idx = torch.nonzero(valid_mask, as_tuple=True)
        P_total = int(batch_idx.numel())

        if P_total == 0:
            self.kd_loss = self.s_pred.sum() * 0.0
            self.ce_loss_override = None
            return

        B = self.s_pred.size(0)
        packed_by_b, U_by_b, sen_by_b = [], [], []
        for b in range(B):
            rs = cached_items[b]["rs"]
            packed_by_b.append(torch.as_tensor(rs["packed"], device=self.distiller.student_device, dtype=torch.uint8))
            U_by_b.append(int(rs["U"]))
            sen_by_b.append(int(rs["sentinel_id"]))

        if cache_mode == "unc":
            target_prob_tensors = []
            for b in range(B):
                entry = cached_items[b].get("target_prob_fp16")
                if entry is None:
                    raise RuntimeError("Cache item missing target_prob_fp16 in UNC mode.")
                target_prob_tensors.append(
                    torch.as_tensor(entry, dtype=torch.float16).to(self.distiller.student_device).float()
                )
            if target_prob_tensors:
                self.cached_target_probs = torch.stack(target_prob_tensors, dim=0)

        U_max = max(U_by_b) if len(U_by_b) > 0 else 0
        ids_U = torch.zeros((P_total, U_max), dtype=torch.long, device=self.distiller.student_device)
        probs_U = torch.zeros((P_total, U_max), dtype=torch.float32, device=self.distiller.student_device)

        # Vectorized unpacking: gather all blocks into a single tensor for batch processing
        if U_max > 0:
            # Preallocate buffer for all packed blocks [P_total, U_max, 3]
            all_blocks = torch.zeros((P_total, U_max, 3), dtype=torch.uint8, device=self.distiller.student_device)
            sentinel_ids = torch.zeros(P_total, dtype=torch.int32, device=self.distiller.student_device)
            
            for r in range(P_total):
                b = int(batch_idx[r].item())
                p = int(pos_idx[r].item())
                U = U_by_b[b]
                sentinel_ids[r] = sen_by_b[b]
                if U == 0:
                    continue
                packed = packed_by_b[b]
                block = packed[p * U * 3:(p + 1) * U * 3]
                all_blocks[r, :U, :] = block.view(U, 3)
            
            # Vectorized unpack: [P_total, U_max, 3] -> [P_total, U_max] ids and q7
            b_flat = all_blocks.to(torch.int64)  # [P_total, U_max, 3]
            x = b_flat[:, :, 0] | (b_flat[:, :, 1] << 8) | (b_flat[:, :, 2] << 16)  # [P_total, U_max]
            ids17 = x & ((1 << 17) - 1)  # 17-bit IDs
            q7 = (x >> 17) & ((1 << 7) - 1)  # 7-bit probs
            
            # Filter sentinels and zero probabilities
            sentinel_mask = ids17 != sentinel_ids.unsqueeze(1)  # [P_total, U_max]
            q7_mask = q7 > 0
            keep = sentinel_mask & q7_mask
            
            # Convert to final format
            ids_U_cpu = ids17.to(torch.int64)
            ids_U_cpu[~keep] = 0  # Zero out invalid entries
            probs_U_cpu = (q7.float() / 127.0).clamp_min(0.0)
            probs_U_cpu[~keep] = 0.0
            
            # Normalize probabilities per row
            row_sums = probs_U_cpu.sum(dim=1, keepdim=True).clamp_min(1e-12)
            probs_U_cpu = probs_U_cpu / row_sums
            
            # Already on student_device, no transfer needed
            ids_U = ids_U_cpu
            probs_U = probs_U_cpu

        self.rs_batch_idx = batch_idx
        self.rs_pos_idx = pos_idx
        self.ids_U = ids_U
        self.probs_U = probs_U

        if self._handle_cached_distill_atkd(
            distill_type=distill_type,
            cache_mode=cache_mode,
            batch_idx=batch_idx,
            pos_idx=pos_idx,
            ids_U=ids_U,
            probs_U=probs_U,
        ):
            return

        if self._handle_cached_distill_teacher_logits(
            supports_cached_teacher_logits=supports_cached_teacher_logits,
            batch_idx=batch_idx,
            pos_idx=pos_idx,
            ids_U=ids_U,
            probs_U=probs_U,
        ):
            return

        # For other distill types, compute KD loss with sampled teacher vocab
        # CE will be computed normally over full student vocabulary
        kd_pos_proxy = torch.zeros((self.valid_next.size(0), self.valid_next.size(1)), device=self.distiller.student_device, dtype=torch.float32)
        if self.s_log_probs is None:
            self.s_log_probs = torch.log_softmax((self.s_pred.float() / self.T), dim=-1)
        student_targets = self.input_ids_s[:, 1:]
        student_logp_targets = torch.gather(self.s_log_probs, -1, student_targets.unsqueeze(-1)).squeeze(-1)
        student_ce_full = (-student_logp_targets).detach().masked_fill(~self.valid_next, 0.0)
        
        s_rows_logp = self.s_log_probs[batch_idx, pos_idx]
        s_logp_on_U_exact = torch.gather(s_rows_logp, 1, ids_U)
        kd_rows_exact = -(probs_U * s_logp_on_U_exact).sum(dim=1)
        kd_pos_proxy[batch_idx, pos_idx] = kd_rows_exact.float()

        # Build ce_pos_proxy for score context only (not used as CE override)
        ce_pos_proxy = student_ce_full.float()

        self._handle_cached_distill_by_type(
            distill_type=distill_type,
            score_enabled=score_enabled_flag,
            kd_pos_proxy=kd_pos_proxy,
            ce_pos_proxy=ce_pos_proxy,
            T=T,
        )

    def _handle_cached_distill_atkd(
        self,
        distill_type,
        cache_mode,
        batch_idx,
        pos_idx,
        ids_U,
        probs_U,
    ) -> bool:
        if distill_type != "atkd":
            return False
        if self.t_pred is None and cache_mode != "unc":
            raise RuntimeError(
                "AT-KD with offline cache requires offline_cache_mode='unc' or a live teacher."
            )
        cache_bundle = ATKDCacheBundle(
            cache_mode=cache_mode,
            target_probs=self.cached_target_probs,
            rs_ids=ids_U,
            rs_probs=probs_U,
            rs_batch_idx=batch_idx,
            rs_pos_idx=pos_idx,
        )
        kd_loss, atkd_metrics = compute_atkd_loss(
            config=self.distiller.config,
            student_device=self.distiller.student_device,
            input_ids_s=self.input_ids_s,
            valid_next=self.valid_next,
            student_logits=self.s_pred,
            teacher_logits=self.t_pred,
            cache_bundle=cache_bundle,
        )
        self.kd_loss = kd_loss
        self.ce_loss_override = None
        if self.distiller.logger and atkd_metrics:
            self.distiller.logger.log(atkd_metrics, self.distiller.global_step)
        return True

    def _handle_cached_distill_teacher_logits(
        self,
        supports_cached_teacher_logits,
        batch_idx,
        pos_idx,
        ids_U,
        probs_U,
    ) -> bool:
        """
        Compute vanilla KD loss (all positions) using sampled teacher vocabulary from cache.
        
        This is the simplest cached KD: no token selection, just distill on all valid positions.
        Teacher probs come from cached sampled vocabulary (RS-KD).
        Student computes full softmax, but we only gather at sampled teacher indices.
        """
        if self.s_log_probs is None:
            self.s_log_probs = torch.log_softmax((self.s_pred.float() / self.T), dim=-1)
        s_rows_logp = self.s_log_probs[batch_idx, pos_idx]
        s_logp_on_U_exact = torch.gather(s_rows_logp, 1, ids_U)
        kd_rows_exact = -(probs_U * s_logp_on_U_exact).sum(dim=1)
        self.ce_loss_override = None

        valid_count = self.valid_next.sum().clamp_min(1).to(torch.float32)
        if supports_cached_teacher_logits:
            numerator = kd_rows_exact.float().sum()
        else:
            kd_pos_proxy = torch.zeros_like(self.valid_next, dtype=torch.float32)
            kd_pos_proxy[batch_idx, pos_idx] = kd_rows_exact.float()
            numerator = (kd_pos_proxy * self.valid_next).sum()

        kd_loss_value = numerator / valid_count
        self.kd_loss = kd_loss_value.to(self.distiller.student_dtype)
        return True

    def _cached_dkd_loss_from_keepmask(self, keep_mask: torch.Tensor, weight_mask: Optional[torch.Tensor] = None) -> None:
        """Compute DKD loss using cached teacher distributions (ids_U/probs_U)."""
        if self.ids_U is None or self.probs_U is None or self.rs_batch_idx is None or self.rs_pos_idx is None:
            raise RuntimeError("DKD cached computation requires cached ids/probs.")

        if self.s_log_probs is None:
            self.s_log_probs = torch.log_softmax((self.s_pred.float() / self.T), dim=-1)

        # Map (b, t) -> row index in ids_U/probs_U
        L = self.valid_next.size(1)
        keys = self.rs_batch_idx.to(torch.int64) * L + self.rs_pos_idx.to(torch.int64)
        key_to_row = {int(keys[i].item()): i for i in range(keys.numel())}

        sel = keep_mask.nonzero(as_tuple=False)
        if sel.numel() == 0:
            self.kd_loss = self.s_pred.sum() * 0.0
            self.ce_loss_override = None
            return

        kd_terms: List[torch.Tensor] = []
        weights: List[torch.Tensor] = []
        for b_idx, t_idx in sel:
            key = int(b_idx.item() * L + t_idx.item())
            row = key_to_row.get(key)
            if row is None:
                continue
            ids_row = self.ids_U[row]
            probs_row = self.probs_U[row]
            valid = probs_row > 0
            if not valid.any():
                continue
            ids_sel = ids_row[valid]
            t_probs_sel = probs_row[valid]
            t_probs_sel = t_probs_sel / t_probs_sel.sum().clamp_min(1e-12)

            s_log_row = self.s_log_probs[b_idx, t_idx, :]
            s_probs_sel = s_log_row.gather(-1, ids_sel).exp()
            s_probs_sel = s_probs_sel / s_probs_sel.sum().clamp_min(1e-12)

            target_id = self.input_ids[b_idx, t_idx + 1]
            target_mask = ids_sel == target_id

            tckd = s_probs_sel.sum() * 0.0
            nckd = s_probs_sel.sum() * 0.0

            if target_mask.any():
                t_pt = t_probs_sel[target_mask].sum()
                s_pt = s_probs_sel[target_mask].sum()
                t_rest = (t_probs_sel.sum() - t_pt).clamp_min(1e-12)
                s_rest = (s_probs_sel.sum() - s_pt).clamp_min(1e-12)

                t_bin = torch.stack([t_pt, t_rest])
                s_bin = torch.stack([s_pt, s_rest])
                log_t_bin = torch.log(t_bin.clamp_min(1e-12))
                log_s_bin = torch.log(s_bin.clamp_min(1e-12))
                tckd = self._kl_directional(log_t_bin, log_s_bin)

                non_target_mask = ~target_mask
            else:
                non_target_mask = torch.ones_like(target_mask, dtype=torch.bool)

            if non_target_mask.any():
                t_hat = t_probs_sel[non_target_mask]
                s_hat = s_probs_sel[non_target_mask]
                t_hat = t_hat / t_hat.sum().clamp_min(1e-12)
                s_hat = s_hat / s_hat.sum().clamp_min(1e-12)
                log_t_hat = torch.log(t_hat.clamp_min(1e-12))
                log_s_hat = torch.log(s_hat.clamp_min(1e-12))
                nckd = self._kl_directional(log_t_hat, log_s_hat)

            alpha = float(getattr(self.distiller.config, "dkd_alpha", 1.0))
            beta = float(getattr(self.distiller.config, "dkd_beta", 8.0))
            kd_row = alpha * tckd + beta * nckd
            kd_terms.append(kd_row)
            if weight_mask is not None:
                weights.append(weight_mask[b_idx, t_idx])

        if not kd_terms:
            self.kd_loss = self.s_pred.sum() * 0.0
        else:
            kd_stack = torch.stack(kd_terms)
            if weight_mask is not None and weights:
                w = torch.stack(weights).float().to(self.distiller.student_device)
                self.kd_loss = (kd_stack * w).sum() / w.sum().clamp_min(1e-12)
            else:
                self.kd_loss = kd_stack.mean()
        self.ce_loss_override = None

    def _handle_cached_distill_linucb(
        self,
        kd_pos_proxy,
        ce_pos_proxy,
        T,
    ) -> None:
        if self.distiller.bandit_manager is None:
            raise RuntimeError("LinUCB bandit is not initialized.")
        ent_for_bandit = self.distiller._entropy_for_selection(self.input_ids, t_pred=None)
        if self.s_log_probs is None:
            self.s_log_probs = torch.log_softmax((self.s_pred.float() / self.T), dim=-1)
        student_entropy = (-(self.s_log_probs.exp() * self.s_log_probs).sum(-1)).detach()
        teacher_ce = None
        if self.t_log_probs is not None:
            targets_t = self.input_ids[:, 1:].to(self.t_log_probs.device, non_blocking=True)
            teacher_logp = self.t_log_probs.gather(-1, targets_t.unsqueeze(-1)).squeeze(-1)
            teacher_ce = (-teacher_logp).to(self.distiller.student_device).detach()
        student_ce = ce_pos_proxy.detach()
        kd_terms, metrics, selection = self.distiller.bandit_manager.select_tokens(
            input_ids=self.input_ids,
            attention_mask=self.attn_mask,
            ent_raw=ent_for_bandit.detach(),
            student_entropy=student_entropy,
            teacher_ce=teacher_ce,
            student_ce=student_ce,
            kl_pos=kd_pos_proxy,
            valid_next=self.valid_next,
            temperature=T,
        )
        use_dkd = getattr(self.distiller.config, "distill_type", "linucb") == "linucb-dkd"

        if use_dkd:
            if selection is None or selection[0].numel() == 0:
                self.kd_loss = self.s_pred.sum() * 0.0
                self.ce_loss_override = None
            else:
                keep_mask = torch.zeros_like(self.valid_next, dtype=torch.bool)
                b_idx, t_idx = selection
                b_idx = b_idx.to(keep_mask.device)
                t_idx = t_idx.to(keep_mask.device)
                keep_mask[b_idx, t_idx] = True
                if keep_mask.any():
                    self._cached_dkd_loss_from_keepmask(keep_mask)
                else:
                    self.kd_loss = self.s_pred.sum() * 0.0
                    self.ce_loss_override = None
        else:
            kd_loss = torch.cat(kd_terms).mean() if kd_terms else self.s_pred.sum() * 0.0
            self.kd_loss = kd_loss
            self.ce_loss_override = None
        self.extra_metrics = metrics or None

    def _rs_bucket_bounds(self):
        """Return (lower_q, upper_q) if pos-rs bucket mode is active, else None."""
        return self.distiller._rs_bucket_bounds()

    def _apply_rs_bucket_filter(self, vec, idx, bounds):
        """Restrict candidate tensors to the requested percentile band."""
        return self.distiller._apply_rs_bucket_filter(vec, idx, bounds)

    def _rs_selection_fraction(self, bucket_bounds: Optional[Tuple[float, float]]) -> float:
        """Return selection fraction (0-1) for RS-KD token sampling."""
        pct = max(0.0, min(1.0, float(getattr(self.distiller.config, "k_percent", 0)) / 100.0))
        if bucket_bounds is None:
            return pct
        lower_q, upper_q = bucket_bounds
        return max(0.0, min(1.0, upper_q - lower_q))

    @staticmethod
    def _rs_target_sample_count(
        total_valid_tokens: int,
        candidate_count: int,
        pct: float,
        shared_quota: Optional[int],
    ) -> int:
        """Translate selection fraction into an integer sample quota."""
        if candidate_count <= 0 or total_valid_tokens <= 0:
            return 0
        desired = max(1, math.ceil(pct * total_valid_tokens))
        if shared_quota is not None:
            desired = min(desired, shared_quota)
        return min(candidate_count, desired)

    def _handle_cached_distill_pos_rs_kd(
        self,
        kd_pos_proxy,
        ce_pos_proxy,
    ) -> None:
        alpha = float(getattr(self.distiller.config, "rs_alpha", 1.0))
        q_floor = float(getattr(self.distiller.config, "rs_floor", 1e-6))
        weight_mask = torch.zeros_like(self.valid_next, dtype=torch.float32)
        use_score = bool(getattr(self.distiller.config, "score_token_selection", False))
        bucket_bounds = self._rs_bucket_bounds()
        pct = self._rs_selection_fraction(bucket_bounds)
        normalize_topk = bool(getattr(self.distiller.config, "normalize_topk_by_length", False))
        shared_quota = None
        if normalize_topk:
            total_valid = int(self.valid_next.sum().item())
            avg_valid = total_valid / max(1, self.valid_next.size(0))
            shared_quota = max(1, math.ceil(pct * avg_valid))
        if use_score:
            ent_for_score = self.distiller._entropy_for_selection(self.input_ids, t_pred=None).to(self.distiller.student_device)
            score_ctx = self.distiller._prepare_score_context(
                ent_raw=ent_for_score,
                kl_pos=kd_pos_proxy,
                s_log_probs=None,
                valid_next=self.valid_next,
                input_ids=self.input_ids,
                student_ce_override=ce_pos_proxy,
            )
        else:
            score_ctx = None
        ent_for_rs = None
        entropy_logger = getattr(self.distiller, "_pos_rs_entropy_logger", None)
        for i in range(self.valid_next.size(0)):
            mask_i = self.valid_next[i]
            total_valid_i = int(mask_i.sum().item())
            if total_valid_i < 3:
                continue
            if use_score:
                combined = self.distiller._build_score_vector(score_ctx, i, mask_i)
                if combined is None:
                    continue
                vec = combined[mask_i].float()
            else:
                if ent_for_rs is None:
                    ent_for_rs = self.distiller._entropy_for_selection(self.input_ids, t_pred=None)
                ent_i = ent_for_rs[i]
                vec = ent_i[mask_i].float()
            valid_idx_i = torch.where(mask_i)[0]
            filtered = False
            if bucket_bounds is not None:
                filtered_vec, filtered_idx, filtered = self._apply_rs_bucket_filter(vec, valid_idx_i, bucket_bounds)
                if filtered:
                    vec = filtered_vec
                    valid_idx_i = filtered_idx
            n_valid = int(valid_idx_i.numel())
            if n_valid < 1:
                continue
            vec = torch.clamp(vec, min=1e-8)
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
            quota = shared_quota if normalize_topk else None
            k_count = self._rs_target_sample_count(total_valid_i, n_valid, pct, quota)
            if k_count < 1:
                continue
            if entropy_logger is not None:
                entropy_logger.record(
                    batch_index=i,
                    valid_indices=valid_idx_i,
                    entropies=vec,
                    probabilities=q,
                    source="score" if use_score else "entropy",
                    bucket_filtered=filtered,
                    bucket_bounds=bucket_bounds,
                    selection_fraction=pct,
                    alpha=alpha,
                    q_floor=q_floor,
                    quota=k_count,
                    input_ids=self.input_ids,
                    valid_next=self.valid_next,
                    global_step=self.distiller.global_step,
                    k_percent=float(getattr(self.distiller.config, "k_percent", 0.0)),
                )
            rel_sel = torch.multinomial(q, num_samples=k_count, replacement=True)
            abs_sel = valid_idx_i[rel_sel]
            q_sel = q[rel_sel]
            w = 1.0 / torch.clamp(q_sel, min=q_floor)
            weight_mask[i, abs_sel] += w
        kd_loss = (
            kd_pos_proxy * weight_mask
        ).sum() / weight_mask.sum().clamp_min(1e-12)
        self.kd_loss = kd_loss
        self.ce_loss_override = None

    def _handle_cached_distill_by_type(
        self,
        distill_type,
        score_enabled,
        kd_pos_proxy,
        ce_pos_proxy,
        T,
    ) -> None:
        if distill_type in {"linucb", "linucb-dkd"}:
            self._handle_cached_distill_linucb(kd_pos_proxy, ce_pos_proxy, T)
            return

        if distill_type == "pos-rs-kd":
            self._handle_cached_distill_pos_rs_kd(kd_pos_proxy, ce_pos_proxy)
            return
        if distill_type in {"top-k-tok-dkd", "dkd"}:
            if distill_type == "dkd":
                keep_mask = self.valid_next.clone()
            else:
                keep_mask = self._compute_cached_keep_mask(
                    distill_type="top-k-tok",
                    score_enabled=score_enabled,
                    kd_pos_proxy=kd_pos_proxy,
                    ce_pos_proxy=ce_pos_proxy,
                )
            self._cached_dkd_loss_from_keepmask(keep_mask)
            return
        if distill_type == "random-dkd":
            keep_mask = self._compute_cached_keep_mask(
                distill_type="random",
                score_enabled=score_enabled,
                kd_pos_proxy=kd_pos_proxy,
                ce_pos_proxy=ce_pos_proxy,
            )
            self._cached_dkd_loss_from_keepmask(keep_mask)
            return
        if distill_type == "pos-rs-kd-dkd":
            weight_mask = self._compute_cached_weight_mask_pos_rs()
            self._cached_dkd_loss_from_keepmask(weight_mask > 0, weight_mask=weight_mask)
            return

        handlers = {
            "top-k-tok": self._handle_cached_distill_top_k_tok,
            "bucket": self._handle_cached_distill_bucket,
            "random": self._handle_cached_distill_random,
        }
        handler = handlers.get(distill_type)
        if handler is not None:
            handler(score_enabled, kd_pos_proxy, ce_pos_proxy)
            return

        self._handle_cached_distill_default(distill_type, score_enabled, kd_pos_proxy, ce_pos_proxy)

    def _handle_cached_distill_top_k_tok(
        self,
        score_enabled,
        kd_pos_proxy,
        ce_pos_proxy,
    ) -> None:
        keep_mask = self._compute_cached_keep_mask(
            distill_type="top-k-tok",
            score_enabled=score_enabled,
            kd_pos_proxy=kd_pos_proxy,
            ce_pos_proxy=ce_pos_proxy,
        )

        self._finalize_cached_distill_loss(
            distill_type="top-k-tok",
            keep_mask=keep_mask,
            kd_pos_proxy=kd_pos_proxy,
        )

    def _handle_cached_distill_bucket(
        self,
        score_enabled,
        kd_pos_proxy,
        ce_pos_proxy,
    ) -> None:
        keep_mask = self._compute_cached_keep_mask(
            distill_type="bucket",
            score_enabled=score_enabled,
            kd_pos_proxy=kd_pos_proxy,
            ce_pos_proxy=ce_pos_proxy,
        )

        self._finalize_cached_distill_loss(
            distill_type="bucket",
            keep_mask=keep_mask,
            kd_pos_proxy=kd_pos_proxy,
        )

    def _handle_cached_distill_random(
        self,
        score_enabled,
        kd_pos_proxy,
        ce_pos_proxy,
    ) -> None:
        keep_mask = self._compute_cached_keep_mask(
            distill_type="random",
            score_enabled=score_enabled,
            kd_pos_proxy=kd_pos_proxy,
            ce_pos_proxy=ce_pos_proxy,
        )

        self._finalize_cached_distill_loss(
            distill_type="random",
            keep_mask=keep_mask,
            kd_pos_proxy=kd_pos_proxy,
        )

    def _handle_cached_distill_default(
        self,
        distill_type,
        score_enabled,
        kd_pos_proxy,
        ce_pos_proxy,
    ) -> None:
        keep_mask = self._compute_cached_keep_mask(
            distill_type=distill_type,
            score_enabled=score_enabled,
            kd_pos_proxy=kd_pos_proxy,
            ce_pos_proxy=ce_pos_proxy,
        )

        self._finalize_cached_distill_loss(
            distill_type=distill_type,
            keep_mask=keep_mask,
            kd_pos_proxy=kd_pos_proxy,
        )

    def _compute_cached_keep_mask(
        self,
        distill_type,
        score_enabled,
        kd_pos_proxy,
        ce_pos_proxy,
    ) -> torch.Tensor:
        keep_mask = self.valid_next.clone()
        if distill_type not in {"top-k-tok", "bucket", "random", "top-k-tok-dkd"}:
            return keep_mask
        score_ctx = self._maybe_build_score_context(score_enabled, kd_pos_proxy, ce_pos_proxy)
        handlers = {
            "top-k-tok": self._compute_keep_mask_top_k_tok,
            "top-k-tok-dkd": self._compute_keep_mask_top_k_tok,
            "bucket": self._compute_keep_mask_bucket,
            "random": self._compute_keep_mask_random,
        }
        return handlers[distill_type](score_ctx, score_enabled)

    def _maybe_build_score_context(self, score_enabled, kd_pos_proxy, ce_pos_proxy):
        if not score_enabled:
            return None
        ent_for_score = self.distiller._entropy_for_selection(self.input_ids, t_pred=None).to(self.distiller.student_device)
        return self.distiller._prepare_score_context(
            ent_raw=ent_for_score,
            kl_pos=kd_pos_proxy,
            s_log_probs=None,
            valid_next=self.valid_next,
            input_ids=self.input_ids,
            student_ce_override=ce_pos_proxy,
        )

    def _compute_cached_weight_mask_pos_rs(self) -> torch.Tensor:
        """RS-KD style importance weights using cached entropy."""
        ent_raw = self.distiller._entropy_for_selection(self.input_ids, t_pred=None)
        weight_mask = torch.zeros_like(self.valid_next, dtype=torch.float32)
        Bsz = ent_raw.size(0)
        alpha = float(getattr(self.distiller.config, "rs_alpha", 1.0))
        q_floor = float(getattr(self.distiller.config, "rs_floor", 1e-6))
        pct = max(0.0, min(1.0, self.distiller.config.k_percent / 100.0))
        bucket_bounds = self._rs_bucket_bounds()
        normalize_topk = bool(getattr(self.distiller.config, "normalize_topk_by_length", False))
        shared_quota = None
        if normalize_topk:
            total_valid = int(self.valid_next.sum().item())
            avg_valid = total_valid / max(1, self.valid_next.size(0))
            shared_quota = max(1, math.ceil(pct * avg_valid))

        for i in range(Bsz):
            valid_next_i = self.valid_next[i]
            if valid_next_i.sum().item() < 1:
                continue
            vec = ent_raw[i][valid_next_i].float()
            valid_idx = torch.where(valid_next_i)[0]
            if bucket_bounds is not None:
                vec, valid_idx, _ = self._apply_rs_bucket_filter(vec, valid_idx, bucket_bounds)
            valid_count = int(valid_idx.numel())
            if valid_count < 1:
                continue

            vec = torch.clamp(vec, min=1e-8)
            logits = vec if alpha == 1.0 else vec * alpha
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
            rel_sel = torch.multinomial(q, num_samples=k_count, replacement=True)
            abs_sel = valid_idx[rel_sel]
            q_sel = q[rel_sel]
            w = 1.0 / torch.clamp(q_sel, min=q_floor)
            weight_mask[i, abs_sel] += w

        return weight_mask

    def _compute_keep_mask_top_k_tok(self, score_ctx, score_enabled):
        ent_cached = self.distiller._entropy_for_selection(self.input_ids, t_pred=None).to(self.distiller.student_device)
        if score_enabled and score_ctx is not None:
            stat_elim = torch.full_like(ent_cached, float("-inf"))
            for i in range(self.valid_next.size(0)):
                mask_i = self.valid_next[i]
                combined = self.distiller._build_score_vector(score_ctx, i, mask_i)
                if combined is not None:
                    stat_elim[i] = combined
            stat_elim = stat_elim.masked_fill(~self.valid_next, float("-inf"))
        else:
            stat_elim = ent_cached.masked_fill(~self.valid_next, float("-inf"))

        normalize_topk = bool(getattr(self.distiller.config, "normalize_topk_by_length", False))
        use_gls = bool(getattr(self.distiller.config, "gls_enabled", False)) and not normalize_topk
        sel_topk_count = 0
        sel_gls_count = 0
        if not use_gls:
            pct = max(0.0, min(1.0, self.distiller.config.k_percent / 100.0))
            shared_quota = None
            if normalize_topk:
                total_valid = int(self.valid_next.sum().item())
                avg_valid = total_valid / max(1, self.valid_next.size(0))
                shared_quota = max(1, math.ceil(pct * avg_valid))
            keep_mask = torch.zeros_like(self.valid_next, dtype=torch.bool)
            for i in range(self.valid_next.size(0)):
                mask_i = self.valid_next[i]
                n_valid = int(mask_i.sum().item())
                if n_valid < 3:
                    continue
                if normalize_topk and shared_quota is not None:
                    k = min(n_valid, shared_quota)
                else:
                    k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                sel_topk_count += int(k)
                valid_idx_i = torch.where(mask_i)[0]
                scores = stat_elim[i][mask_i].float()
                if scores.numel() == 0:
                    continue
                _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
                sel_abs = valid_idx_i[rel]
                keep_mask[i, sel_abs] = True
        else:
            self.distiller._gls_init_if_needed()
            thr = self.distiller._gls_threshold(top_percent=self.distiller.config.k_percent)
            if thr is None:
                pct = max(0.0, min(1.0, self.distiller.config.k_percent / 100.0))
                shared_quota = None
                if normalize_topk:
                    total_valid = int(self.valid_next.sum().item())
                    avg_valid = total_valid / max(1, self.valid_next.size(0))
                    shared_quota = max(1, math.ceil(pct * avg_valid))
                keep_mask = torch.zeros_like(self.valid_next, dtype=torch.bool)
                for i in range(self.valid_next.size(0)):
                    mask_i = self.valid_next[i]
                    n_valid = int(mask_i.sum().item())
                    if n_valid < 3:
                        continue
                    if normalize_topk and shared_quota is not None:
                        k = min(n_valid, shared_quota)
                    else:
                        k = max(1, min(n_valid, math.ceil(pct * n_valid)))
                    sel_topk_count += int(k)
                    valid_idx_i = torch.where(mask_i)[0]
                    scores = stat_elim[i][mask_i].float()
                    if scores.numel() == 0:
                        continue
                    _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
                    sel_abs = valid_idx_i[rel]
                    keep_mask[i, sel_abs] = True
            else:
                keep_mask = (stat_elim >= thr) & self.valid_next
                sel_gls_count = int(keep_mask.sum().item())
            assert stat_elim is not None
            vals = stat_elim[self.valid_next].detach().float().to("cpu")
            vals = vals[torch.isfinite(vals)]
            self.distiller._gls_push(vals)
            if getattr(self.distiller.config, "gls_log_threshold", False) and ('thr' in locals()) and thr is not None and self.distiller.logger:
                self.distiller.logger.log_scalar("train/gls_threshold", float(thr), self.distiller.global_step)
        if self.distiller.logger:
            self.distiller.logger.log(
                {
                    "train/selected_tokens_topk": float(sel_topk_count),
                    "train/selected_tokens_gls": float(sel_gls_count),
                },
                self.distiller.global_step,
            )
        return keep_mask

    def _compute_keep_mask_bucket(self, score_ctx, score_enabled):
        keep_mask = torch.zeros_like(self.valid_next, dtype=torch.bool)
        ent_for_bucket = None
        for i in range(self.valid_next.size(0)):
            mask_i = self.valid_next[i]
            if mask_i.sum() < 3:
                continue
            if score_enabled and score_ctx is not None:
                combined = self.distiller._build_score_vector(score_ctx, i, mask_i)
                if combined is None:
                    continue
                vec = combined[mask_i].float()
            else:
                if ent_for_bucket is None:
                    ent_for_bucket = self.distiller._entropy_for_selection(self.input_ids, t_pred=None).to(self.distiller.student_device)
                vec = ent_for_bucket[i][mask_i].float()
            low = torch.quantile(vec, self.distiller.config.bucket_lower_percent / 100.0)
            high = torch.quantile(vec, self.distiller.config.bucket_upper_percent / 100.0)
            rel = torch.where(mask_i)[0]
            sel = (vec >= low) & (vec <= high)
            if sel.any():
                keep_mask[i, rel[sel]] = True
        return keep_mask

    def _compute_keep_mask_random(self, score_ctx, score_enabled):
        keep_mask = torch.zeros_like(self.valid_next, dtype=torch.bool)
        pct = max(0.0, min(1.0, self.distiller.config.k_percent / 100.0))
        for i in range(self.valid_next.size(0)):
            mask_i = self.valid_next[i]
            n_valid = int(mask_i.sum().item())
            if n_valid < 2:
                continue
            k = max(1, int(n_valid * pct))
            valid_idx_i = torch.where(mask_i)[0]
            if score_enabled and score_ctx is not None:
                combined = self.distiller._build_score_vector(score_ctx, i, mask_i)
                if combined is None:
                    continue
                scores = combined[mask_i].float()
                if scores.numel() == 0:
                    continue
                scores = scores - scores.min()
                scores = torch.clamp(scores, min=1e-8)
                probs = scores / scores.sum()
                rel = torch.multinomial(probs, num_samples=k, replacement=False)
                sel_abs = valid_idx_i[rel]
            else:
                perm = torch.randperm(valid_idx_i.numel(), device=self.distiller.student_device)
                sel_abs = valid_idx_i[perm[:k]]
            keep_mask[i, sel_abs] = True
        return keep_mask

    def _finalize_cached_distill_loss(
        self,
        distill_type,
        keep_mask,
        kd_pos_proxy,
    ) -> None:
        if distill_type == "top-k-tok":
            self.ce_selection_mask = keep_mask
            kd_loss = (
                kd_pos_proxy.to(self.distiller.student_dtype) * keep_mask
            ).sum() / keep_mask.sum().clamp_min(1)
            self.kd_loss = kd_loss
            self.ce_loss_override = None
            return
        kd_loss = (
            kd_pos_proxy.to(self.distiller.student_dtype) * keep_mask
        ).sum() / keep_mask.sum().clamp_min(1)
        self.kd_loss = kd_loss
        self.ce_loss_override = None

    def _handle_cache_miss_kd_loss(self) -> None:
        assert self.t_log_probs is not None and self.t_pred is not None, "Teacher logits required for online RS-KD when cache missing"
        t_logp_Tkd = self.t_log_probs.to(self.distiller.student_device)
        p_Tkd = t_logp_Tkd.exp()

        mask = self.valid_next
        P_total = int(mask.sum().item())
        if P_total == 0:
            self.kd_loss = self.s_pred.sum() * 0.0
            self.ce_loss_override = None
            return

        if self.s_log_probs is None:
            self.s_log_probs = torch.log_softmax((self.s_pred.float() / self.T), dim=-1)
        s_rows = self.s_log_probs[mask]
        t_rows = t_logp_Tkd[mask]
        p_rows = p_Tkd[mask]

        kd_per_pos = (p_rows * (t_rows - s_rows)).sum(dim=-1)
        kd_loss = kd_per_pos.mean().to(self.distiller.student_dtype)
        self.kd_loss = kd_loss
        self.ce_loss_override = None

    def _combine_losses(self) -> None:
        self.kd_loss = self.kd_loss * self.T2
        if self.log_debug:
            kd_val = float(self.kd_loss.detach().item())
            print(
                f"[DEBUG] _forward_batch: kd_loss (after T^2)={kd_val:.10f}, T={self.T}, T2={self.T2}",
                flush=True,
            )

        if self.distiller.config.enable_ce:
            if self.ce_loss_override is not None and not self.ce_all_tokens:
                self.ce_loss = self.ce_loss_override
            else:
                targets = self.input_ids_s[:, 1:]
                V = self.s_pred.size(-1)
                targets = targets.clamp(min=-1, max=V - 1)
                if self.ce_selection_mask is not None and not self.ce_all_tokens:
                    mask_override = self.ce_selection_mask.to(self.ce_mask.device).bool()
                    ce_mask_effective = self.ce_mask & mask_override
                else:
                    ce_mask_effective = self.ce_mask
                targets = targets.masked_fill(~ce_mask_effective, -100)
                s_log_probs_T1 = torch.log_softmax(self.s_pred.float(), dim=-1)
                V = s_log_probs_T1.size(-1)
                flat_log_probs = s_log_probs_T1.reshape(-1, V)
                flat_targets = targets.reshape(-1).long().clone()

                ignore_mask = flat_targets == -100
                valid_range_mask = (flat_targets >= 0) & (flat_targets < V)
                invalid_range_mask = ~(ignore_mask | valid_range_mask)
                if invalid_range_mask.any():
                    bad_vals = flat_targets[invalid_range_mask].detach().to("cpu", non_blocking=True)
                    bad_count = int(bad_vals.numel())
                    flat_targets = flat_targets.masked_fill(invalid_range_mask, -100)
                    if bad_count > 0 and not self.distiller._warned_invalid_targets:
                        min_bad = int(bad_vals.min().item()) if bad_vals.numel() else 0
                        max_bad = int(bad_vals.max().item()) if bad_vals.numel() else 0
                        sample_vals = bad_vals.unique()
                        if sample_vals.numel() > 5:
                            sample_vals = sample_vals[:5]
                        sample_list = ", ".join(str(int(v.item())) for v in sample_vals)
                        print(
                            "[warn] CE targets out of range (count="
                            f"{bad_count}, min={min_bad}, max={max_bad}, sample=[{sample_list}])"
                            " → masking from loss.",
                            flush=True,
                        )
                        self.distiller._warned_invalid_targets = True

                finite_row_mask = torch.isfinite(flat_log_probs).all(dim=-1)
                drop_mask = (~finite_row_mask) & (flat_targets != -100)
                if drop_mask.any():
                    drop_count = int(drop_mask.sum().item())
                    flat_targets[drop_mask] = -100
                    if drop_count > 0 and not self.distiller._warned_invalid_logprob:
                        print(
                            f"[warn] CE log-probs contained {drop_count} non-finite rows → masking from loss.",
                            flush=True,
                        )
                        self.distiller._warned_invalid_logprob = True

                if (flat_targets != -100).any():
                    self.ce_loss = F.nll_loss(flat_log_probs, flat_targets, ignore_index=-100, reduction="mean")
                else:
                    self.ce_loss = torch.zeros((), device=self.distiller.student_device, dtype=self.s_pred.dtype)

            use_unbounded = bool(getattr(self.distiller.config, "unbounded_to_1_loss", False)) and self.distill_type == "dkd"
            if use_unbounded:
                self.total = self.kd_loss + self.ce_loss
            else:
                self.total = (1.0 - self.distiller.config.alpha_ce) * self.kd_loss + self.distiller.config.alpha_ce * self.ce_loss
        else:
            self.ce_loss = torch.tensor(0.0, device=self.distiller.student_device)
            self.total = self.kd_loss

    def _final_checks(self) -> None:
        if (not torch.isfinite(self.total)) or (not torch.isfinite(self.kd_loss)) or (not torch.isfinite(self.ce_loss)):
            print(
                "[warn] skipping batch due to non-finite loss "
                f"(total={self.total.item()}, kd={self.kd_loss.item()}, ce={self.ce_loss.item()})"
            )
            zero = self.s_pred.sum() * 0.0
            self.total = zero + zero
            self.kd_loss_scalar = 0.0
            self.ce_loss_scalar = 0.0
            self.extra_metrics = None
            return

        self.kd_loss_scalar = self.kd_loss.item()
        self.ce_loss_scalar = self.ce_loss.item()

__all__ = ["ForwardBatchRunner"]
