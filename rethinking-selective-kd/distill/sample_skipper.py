from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

class FrozenStudentSampleSkipper:
    """Gate distillation to a fixed subset of documents based on a frozen pre-pass.

    Supported strategies:
    - entropy: frozen-student mean token entropy per document (default)
    - kl: mean token KL divergence between frozen teacher and frozen student per document
    - ce_ratio: mean token CE_s / (CE_t + eps) between frozen student/teacher per document
    - random: deterministic random subset (no forward pass)

    This matches the behavior described by SKIP_SAMPLES_BY_STUDENT + L_PERCENT_SAMPLES_TO_KEEP:
    - Collect a length-N list of per-document scores (N documents processed)
    - Select top-l% (or random l%)
    - Train using that selection mask (implemented as batch filtering)
    """

    def __init__(
        self,
        *,
        config,
        student,
        student_device: torch.device,
        dataloader,
        teacher=None,
        teacher_device: Optional[torch.device] = None,
    ) -> None:
        self.config = config
        self.student = student
        self.student_device = student_device
        self.dataloader = dataloader
        self.teacher = teacher
        self.teacher_device = teacher_device
        # Selected doc ids (top entropy) that we keep for distillation.
        self.selected_sample_ids: Optional[set[int]] = None
        # Optional debugging/inspection buffers (only populated when enabled).
        self.entropy_by_sample_id: Optional[Dict[int, float]] = None
        self.entropy_list: Optional[List[float]] = None
        self.selection_mask_list: Optional[List[int]] = None
        # Prepass timing and token counters (for efficiency logging).
        self.prepass_forward_s: float = 0.0
        self.prepass_total_s: float = 0.0
        self.prepass_tokens: int = 0
        self.prepass_strategy: Optional[str] = None
        self.prepass_selected_ids: Optional[List[int]] = None

    @property
    def enabled(self) -> bool:
        return bool(getattr(self.config, "skip_by_frozen_student", False))

    @property
    def l_percent(self) -> float:
        value = float(getattr(self.config, "L_PERCENT_SAMPLES_TO_KEEP", 20.0))
        return max(0.0, min(100.0, value))

    def prepare(self, *, rank_is_zero: bool = True) -> None:
        if not self.enabled:
            self.selected_sample_ids = None
            self.entropy_by_sample_id = None
            self.entropy_list = None
            self.selection_mask_list = None
            self.prepass_forward_s = 0.0
            self.prepass_total_s = 0.0
            self.prepass_tokens = 0
            self.prepass_strategy = None
            self.prepass_selected_ids = None
            return

        l_percent = self.l_percent
        if l_percent <= 0.0:
            self.selected_sample_ids = None
            self.entropy_by_sample_id = None
            self.entropy_list = None
            self.selection_mask_list = None
            self.prepass_forward_s = 0.0
            self.prepass_total_s = 0.0
            self.prepass_tokens = 0
            self.prepass_strategy = None
            self.prepass_selected_ids = None
            return
        try:
            dl_for_scoring = DataLoader(
                self.dataloader.dataset,
                batch_size=int(getattr(self.config, "batch_size", 1)),
                shuffle=False,
                collate_fn=self.dataloader.collate_fn,
                num_workers=0,
                pin_memory=False,
                persistent_workers=False,
            )
        except Exception:
            dl_for_scoring = self.dataloader

        strategy = str(getattr(self.config, "skip_samples_strategy", "entropy")).strip().lower()
        if strategy not in {"entropy", "random", "kl", "ce_ratio"}:
            strategy = "entropy"

        # Random selection: choose a deterministic random subset of sample_ids.
        # This intentionally does NOT run a frozen-student forward pass.
        if strategy == "random":
            all_ids: list[int] = []
            for batch in dl_for_scoring:
                sample_ids = batch.get("sample_id")
                if sample_ids is None:
                    raise RuntimeError(
                        "skip_by_frozen_student requires batches to include 'sample_id'. "
                        "Ensure the dataset/collator provides it."
                    )
                if torch.is_tensor(sample_ids):
                    batch_ids = [int(x) for x in sample_ids.detach().cpu().tolist()]
                else:
                    batch_ids = [int(x) for x in list(sample_ids)]
                all_ids.extend(batch_ids)

            # Deduplicate while preserving order.
            uniq_ids = list(dict.fromkeys(all_ids))
            n_total = len(uniq_ids)
            n_select = int((n_total * l_percent) // 100)
            n_select = max(0, min(n_total, n_select))

            seed = int(getattr(self.config, "seed", 1337))
            rng = random.Random(seed)
            selected_ids = set(rng.sample(uniq_ids, k=n_select)) if n_select > 0 else set()

            # No entropy buffers in random mode.
            self.entropy_by_sample_id = None
            self.entropy_list = None
            self.selection_mask_list = None
            self.prepass_forward_s = 0.0
            self.prepass_total_s = 0.0
            self.prepass_tokens = 0
            self.prepass_strategy = strategy
            self.prepass_selected_ids = sorted(selected_ids) if selected_ids else None

            if selected_ids:
                max_id = max(selected_ids)
                if set(uniq_ids) == set(range(max_id + 1)):
                    try:
                        mask_list = [0] * (max_id + 1)
                        for sid_int in selected_ids:
                            mask_list[sid_int] = 1
                        self.selection_mask_list = mask_list
                    except Exception:
                        self.selection_mask_list = None

            if rank_is_zero:
                print(
                    f"[skip] Random selection: will distill on {len(selected_ids)}/{n_total} samples ({l_percent:.2f}%), seed={seed}.",
                    flush=True,
                )

            self.selected_sample_ids = selected_ids if selected_ids else None
            self._maybe_log_selected_indices(rank_is_zero=rank_is_zero)
            return

        if strategy in {"kl", "ce_ratio"} and self.teacher is None:
            raise RuntimeError(
                f"skip_samples_strategy='{strategy}' requires a frozen teacher model."
            )

        was_training = bool(getattr(self.student, "training", False))
        was_teacher_training = bool(getattr(self.teacher, "training", False)) if self.teacher is not None else False
        try:
            self.student.eval()
        except Exception:
            pass
        if self.teacher is not None:
            try:
                self.teacher.eval()
            except Exception:
                pass

        scores: List[Tuple[float, int]] = []
        entropy_by_id: Dict[int, float] = {}
        total_batches: Optional[int]
        try:
            total_batches = len(dl_for_scoring)
        except TypeError:
            total_batches = None
        log_points: set[int] = set()
        if total_batches:
            for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
                idx = max(0, int(round(frac * total_batches)) - 1)
                log_points.add(idx)
        prepass_start = time.perf_counter()
        forward_start = time.perf_counter()
        prepass_tokens = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dl_for_scoring):
                sample_ids = batch.get("sample_id")
                if sample_ids is None:
                    raise RuntimeError(
                        "skip_by_frozen_student requires batches to include 'sample_id'. "
                        "Ensure the dataset/collator provides it."
                    )

                if torch.is_tensor(sample_ids):
                    batch_ids = [int(x) for x in sample_ids.detach().cpu().tolist()]
                else:
                    batch_ids = [int(x) for x in list(sample_ids)]

                input_ids_cpu = batch["input_ids"]
                attn_cpu = batch["attention_mask"]
                labels_cpu = batch["labels"]

                if strategy == "entropy":
                    input_ids = input_ids_cpu.to(self.student_device)
                    attn = attn_cpu.to(self.student_device)
                    labels = labels_cpu.to(self.student_device)
                    logits = self.student(input_ids, attention_mask=attn).logits
                    shift_logits = logits[:, :-1, :].float()
                    shift_labels = labels[:, 1:]

                    # Token entropy: H(p) = -sum_v p(v) log p(v)
                    # Compute per-token entropy then average across valid tokens (labels != -100)
                    log_p = F.log_softmax(shift_logits, dim=-1)
                    entropy = -(log_p.exp() * log_p).sum(dim=-1)  # (B, T-1)

                    valid = shift_labels.ne(-100)
                    denom = valid.sum(dim=1).clamp_min(1)
                    mean_scores = (entropy * valid).sum(dim=1) / denom
                    prepass_tokens += int(valid.sum().item())
                elif strategy == "kl":
                    assert self.teacher is not None
                    teacher_device = self.teacher_device or self.student_device
                    input_ids_s = input_ids_cpu.to(self.student_device)
                    attn_s = attn_cpu.to(self.student_device)
                    input_ids_t = input_ids_cpu.to(teacher_device)
                    attn_t = attn_cpu.to(teacher_device)
                    labels_t = labels_cpu.to(teacher_device)

                    t_logits = self.teacher(input_ids_t, attention_mask=attn_t).logits
                    s_logits = self.student(input_ids_s, attention_mask=attn_s).logits

                    if teacher_device != s_logits.device:
                        s_logits = s_logits.to(teacher_device)

                    shift_t = t_logits[:, :-1, :].float()
                    shift_s = s_logits[:, :-1, :].float()
                    shift_labels = labels_t[:, 1:]

                    temperature = float(getattr(self.config, "kd_temperature", 1.0))
                    temperature = max(temperature, 1e-6)
                    log_t = F.log_softmax(shift_t / temperature, dim=-1)
                    log_s = F.log_softmax(shift_s / temperature, dim=-1)
                    if getattr(self.config, "kd_objective", "forward") == "reverse":
                        log_p = log_s
                        log_q = log_t
                    else:
                        log_p = log_t
                        log_q = log_s

                    kl_rows = (log_p.exp() * (log_p - log_q)).sum(dim=-1)
                    valid = shift_labels.ne(-100)
                    denom = valid.sum(dim=1).clamp_min(1)
                    mean_scores = (kl_rows * valid).sum(dim=1) / denom
                    prepass_tokens += int(valid.sum().item())
                else:
                    assert self.teacher is not None
                    teacher_device = self.teacher_device or self.student_device
                    input_ids_s = input_ids_cpu.to(self.student_device)
                    attn_s = attn_cpu.to(self.student_device)
                    input_ids_t = input_ids_cpu.to(teacher_device)
                    attn_t = attn_cpu.to(teacher_device)
                    labels_t = labels_cpu.to(teacher_device)

                    t_logits = self.teacher(input_ids_t, attention_mask=attn_t).logits
                    s_logits = self.student(input_ids_s, attention_mask=attn_s).logits

                    if teacher_device != s_logits.device:
                        s_logits = s_logits.to(teacher_device)

                    shift_t = t_logits[:, :-1, :].float()
                    shift_s = s_logits[:, :-1, :].float()
                    shift_labels = labels_t[:, 1:]

                    temperature = float(getattr(self.config, "kd_temperature", 1.0))
                    temperature = max(temperature, 1e-6)
                    log_t = F.log_softmax(shift_t / temperature, dim=-1)
                    log_s = F.log_softmax(shift_s / temperature, dim=-1)

                    vocab_size = log_s.size(-1)
                    invalid_mask = (shift_labels < 0) | (shift_labels >= vocab_size)
                    valid = shift_labels.ne(-100) & (~invalid_mask)
                    safe_labels = shift_labels.masked_fill(~valid, 0).clamp(min=0, max=vocab_size - 1)

                    student_target_logp = log_s.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                    teacher_target_logp = log_t.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
                    student_ce = (-student_target_logp)
                    teacher_ce = (-teacher_target_logp)
                    eps = 1e-6
                    ce_ratio = student_ce / (teacher_ce + eps)

                    denom = valid.sum(dim=1).clamp_min(1)
                    mean_scores = (ce_ratio * valid).sum(dim=1) / denom
                    prepass_tokens += int(valid.sum().item())

                for sid, val in zip(batch_ids, mean_scores.detach().cpu().tolist()):
                    sid_int = int(sid)
                    val_f = float(val)
                    scores.append((val_f, sid_int))
                    entropy_by_id[sid_int] = val_f
                if rank_is_zero and total_batches and batch_idx in log_points:
                    pct = int(round(100 * (batch_idx + 1) / total_batches))
                    elapsed = time.perf_counter() - forward_start
                    print(
                        f"[skip] Frozen-student prepass progress: {pct}% "
                        f"({batch_idx + 1}/{total_batches} batches), {elapsed:.1f}s elapsed.",
                        flush=True,
                    )
        forward_elapsed = time.perf_counter() - forward_start

        if was_training:
            try:
                self.student.train()
            except Exception:
                pass
        if was_teacher_training and self.teacher is not None:
            try:
                self.teacher.train()
            except Exception:
                pass

        n_total = len(scores)
        n_select = int((n_total * l_percent) // 100)
        n_select = max(0, min(n_total, n_select))

        # Select top entropy (descending)
        scores.sort(key=lambda x: x[0], reverse=True)
        selected_ids = {sid for _, sid in scores[:n_select]}

        # Materialize optional contiguous lists (index by sample_id) when possible.
        self.entropy_by_sample_id = entropy_by_id
        self.entropy_list = None
        self.selection_mask_list = None
        if entropy_by_id:
            max_id = max(entropy_by_id.keys())
            if len(entropy_by_id) == (max_id + 1):
                try:
                    ent_list = [0.0] * (max_id + 1)
                    mask_list = [0] * (max_id + 1)
                    for sid_int, ent in entropy_by_id.items():
                        ent_list[sid_int] = float(ent)
                        mask_list[sid_int] = 1 if sid_int in selected_ids else 0
                    self.entropy_list = ent_list
                    self.selection_mask_list = mask_list
                except Exception:
                    self.entropy_list = None
                    self.selection_mask_list = None

        total_elapsed = time.perf_counter() - prepass_start
        if rank_is_zero:
            print(
                f"[skip] Frozen-student prepass: will distill on top-entropy {len(selected_ids)}/{n_total} samples ({l_percent:.1f}%).",
                flush=True,
            )
            print(
                f"[skip] Frozen-student prepass timing: forward={forward_elapsed:.1f}s total={total_elapsed:.1f}s.",
                flush=True,
            )

        self.selected_sample_ids = selected_ids if selected_ids else None
        self.prepass_forward_s = float(forward_elapsed)
        self.prepass_total_s = float(total_elapsed)
        self.prepass_tokens = int(prepass_tokens)
        self.prepass_strategy = strategy
        self.prepass_selected_ids = sorted(selected_ids) if selected_ids else None
        self._maybe_log_selected_indices(rank_is_zero=rank_is_zero)

    def _maybe_log_selected_indices(self, *, rank_is_zero: bool) -> None:
        if not rank_is_zero:
            return
        if not bool(getattr(self.config, "log_skipping_indices", False)):
            return
        selected_ids = self.prepass_selected_ids
        if not selected_ids:
            return
        out_path = Path(getattr(self.config, "skipping_indices_path", "results/skipping_indices.json"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        display_name = str(getattr(self.config, "display_name", "") or "").strip()
        if not display_name:
            display_name = str(os.getenv("RUN_DISPLAY_NAME", "")).strip()
        slurm_job_id = str(os.getenv("SLURM_JOB_ID", "")).strip()
        cuda_visible = str(os.getenv("CUDA_VISIBLE_DEVICES", "")).strip()
        gpu_name = None
        gpu_index = None
        if self.student_device is not None and self.student_device.type == "cuda":
            try:
                gpu_index = int(self.student_device.index) if self.student_device.index is not None else None
            except Exception:
                gpu_index = None
            try:
                gpu_name = torch.cuda.get_device_name(self.student_device)
            except Exception:
                gpu_name = None
        gpu_label = None
        if gpu_name is not None:
            gpu_label = f"{gpu_name}" if gpu_index is None else f"{gpu_name} (device {gpu_index})"
        payload = {
            "display_name": display_name,
            "slurm_job_id": slurm_job_id,
            "cuda_visible_devices": cuda_visible,
            "gpu_index": gpu_index,
            "gpu_name": gpu_name,
            "gpu": gpu_label,
            "skip_strategy": self.prepass_strategy,
            "indices": selected_ids,
        }
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def maybe_filter_batch(self, batch: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.enabled:
            return batch
        selected_ids = self.selected_sample_ids
        if not selected_ids:
            return batch

        sample_ids = batch.get("sample_id")
        if sample_ids is None:
            return batch

        if torch.is_tensor(sample_ids):
            ids_list = [int(x) for x in sample_ids.detach().cpu().tolist()]
        else:
            try:
                ids_list = [int(x) for x in list(sample_ids)]
            except Exception:
                return batch

        keep_mask = [sid in selected_ids for sid in ids_list]
        if all(keep_mask):
            return batch
        if not any(keep_mask):
            return None

        keep_idx = torch.tensor([i for i, keep in enumerate(keep_mask) if keep], dtype=torch.long)
        filtered: Dict[str, Any] = {}
        B = len(ids_list)
        for key, value in batch.items():
            if torch.is_tensor(value) and value.dim() >= 1 and value.size(0) == B:
                filtered[key] = value.index_select(0, keep_idx.to(value.device))
            else:
                filtered[key] = value
        return filtered
