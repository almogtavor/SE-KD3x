from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from ..config import TrainingConfig


@dataclass
class ATKDCacheBundle:
    """Container for cached signals used by AT-KD when the teacher is offline."""

    cache_mode: str
    target_probs: Optional[torch.Tensor] = None  # [B, L-1] teacher p_t(g)
    rs_ids: Optional[torch.Tensor] = None  # [P_total, U]
    rs_probs: Optional[torch.Tensor] = None  # [P_total, U]
    rs_batch_idx: Optional[torch.Tensor] = None  # [P_total]
    rs_pos_idx: Optional[torch.Tensor] = None  # [P_total]


def compute_atkd_loss(
    config: TrainingConfig,
    student_device: torch.device,
    input_ids_s: torch.Tensor,
    valid_next: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    cache_bundle: ATKDCacheBundle,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute the AT-KD loss given student logits and (optionally) teacher logits.

    When the teacher is unavailable, cached target probabilities plus RS samples from the offline
    cache are used to approximate the diversity-oriented term.
    """
    # Flatten valid positions across the batch
    batch_idx, pos_idx = torch.nonzero(valid_next, as_tuple=True)
    P_total = int(batch_idx.numel())
    if P_total == 0:
        kd_zero = student_logits.sum() * 0.0
        return kd_zero, {}

    target_ids = input_ids_s[batch_idx, pos_idx + 1]
    student_rows = student_logits[batch_idx, pos_idx, :].to(student_device)
    student_log_probs = torch.log_softmax(student_rows.float(), dim=-1)
    log_qtg = student_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
    student_target_prob = torch.exp(log_qtg)

    teacher_log_probs_rows: Optional[torch.Tensor]
    teacher_target_prob: torch.Tensor

    if teacher_logits is not None:
        teacher_rows = teacher_logits[
            batch_idx.to(teacher_logits.device), pos_idx.to(teacher_logits.device), :
        ].to(student_device)
        teacher_log_probs_rows = torch.log_softmax(teacher_rows.float(), dim=-1)
        log_target = teacher_log_probs_rows.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        teacher_target_prob = torch.exp(log_target)
    else:
        teacher_log_probs_rows = None
        if cache_bundle.target_probs is None:
            raise RuntimeError("AT-KD requires cached target probabilities when teacher logits are unavailable.")
        teacher_target_prob = cache_bundle.target_probs[batch_idx, pos_idx].clamp(0.0, 1.0)

    teacher_target_prob = teacher_target_prob.to(torch.float32)
    student_target_prob = student_target_prob.to(torch.float32)
    uncertainty = (1.0 - teacher_target_prob).clamp_min(0.0)

    teacher_rest = uncertainty.clamp_min(1e-8)
    student_rest = (1.0 - student_target_prob).clamp_min(1e-8)
    teacher_bin = torch.stack([teacher_target_prob.clamp_min(1e-8), teacher_rest], dim=-1)
    student_bin = torch.stack([student_target_prob.clamp_min(1e-8), student_rest], dim=-1)
    kl_bin = (teacher_bin * (torch.log(teacher_bin) - torch.log(student_bin))).sum(dim=-1)

    if teacher_log_probs_rows is not None:
        teacher_probs_nt = torch.exp(teacher_log_probs_rows)
        student_probs_nt = torch.exp(student_log_probs)
        mask_target = torch.zeros_like(teacher_probs_nt, dtype=torch.bool)
        mask_target.scatter_(1, target_ids.unsqueeze(1), True)
        teacher_probs_nt = teacher_probs_nt.masked_fill(mask_target, 0.0)
        student_probs_nt = student_probs_nt.masked_fill(mask_target, 0.0)
        rest_teacher = teacher_probs_nt.sum(dim=-1)
        rest_student = student_probs_nt.sum(dim=-1).clamp_min(1e-12)
        kl_off = torch.zeros(P_total, device=student_device, dtype=torch.float32)
        valid_rest = rest_teacher > 1e-12
        if valid_rest.any():
            teacher_hat = teacher_probs_nt[valid_rest] / rest_teacher[valid_rest].unsqueeze(1)
            student_hat = student_probs_nt[valid_rest] / rest_student[valid_rest].unsqueeze(1)
            log_teacher_hat = torch.log(teacher_hat.clamp_min(1e-12))
            log_student_hat = torch.log(student_hat.clamp_min(1e-12))
            kl_vals = (teacher_hat * (log_teacher_hat - log_student_hat)).sum(dim=-1)
            kl_off[valid_rest] = kl_vals
    else:
        if cache_bundle.rs_ids is None or cache_bundle.rs_probs is None:
            raise RuntimeError("Cached RS samples are required for AT-KD when the teacher is offline.")
        if cache_bundle.rs_batch_idx is None or cache_bundle.rs_pos_idx is None:
            raise RuntimeError("Cached RS indices are missing for AT-KD in UNC mode.")
        if cache_bundle.rs_batch_idx.numel() != P_total or cache_bundle.rs_pos_idx.numel() != P_total:
            raise RuntimeError("Cached RS indices do not align with AT-KD positions.")
        if not torch.equal(cache_bundle.rs_batch_idx, batch_idx) or not torch.equal(
            cache_bundle.rs_pos_idx, pos_idx
        ):
            raise RuntimeError("Cached RS ordering mismatch for AT-KD.")

        kl_off = torch.zeros(P_total, device=student_device, dtype=torch.float32)
        for row_idx in range(P_total):
            probs_row = cache_bundle.rs_probs[row_idx]
            ids_row = cache_bundle.rs_ids[row_idx]
            valid_mask = probs_row > 0
            if not valid_mask.any():
                continue
            ids_sel = ids_row[valid_mask]
            probs_sel = probs_row[valid_mask]
            mask_nt = ids_sel != target_ids[row_idx]
            ids_nt = ids_sel[mask_nt]
            probs_nt = probs_sel[mask_nt]
            if probs_nt.numel() == 0:
                continue
            probs_nt = probs_nt / probs_nt.sum().clamp_min(1e-12)
            log_q_subset = student_log_probs[row_idx, ids_nt]
            log_p_subset = torch.log(probs_nt.clamp_min(1e-12))
            q_subset = torch.exp(log_q_subset).clamp_min(1e-12)
            q_hat = q_subset / q_subset.sum().clamp_min(1e-12)
            log_q_hat = torch.log(q_hat.clamp_min(1e-12))
            kl_off[row_idx] = torch.sum(probs_nt * (log_p_subset - log_q_hat))

    hard_percent = float(getattr(config, "atkd_hard_percent", getattr(config, "k_percent", 50)))
    hard_percent = max(0.0, min(100.0, hard_percent))
    hard_mask = torch.zeros(P_total, dtype=torch.bool, device=student_device)
    if hard_percent >= 100.0:
        hard_mask[:] = True
    elif hard_percent > 0.0:
        hard_target = max(1, int(math.ceil((hard_percent / 100.0) * P_total)))
        hard_target = min(P_total, hard_target)
        _, hard_indices = torch.topk(uncertainty, hard_target, largest=True, sorted=False)
        hard_mask[hard_indices] = True
    easy_mask = ~hard_mask
    hard_count = int(hard_mask.sum().item())
    easy_count = int(easy_mask.sum().item())

    lambda_easy = float(getattr(config, "atkd_loss_lambda", 0.2))
    lambda_easy = float(min(max(lambda_easy, 0.0), 1.0))

    zero_float = torch.zeros((), device=student_device, dtype=torch.float32)
    easy_sum_val = zero_float.clone()
    hard_sum_val = zero_float.clone()
    easy_avg = None
    hard_avg = None
    if easy_count > 0:
        easy_values = kl_off[easy_mask]
        easy_sum_val = easy_values.sum()
        easy_avg = easy_values.mean()
    if hard_count > 0:
        hard_values = (kl_bin + kl_off)[hard_mask]
        hard_sum_val = hard_values.sum()
        hard_avg = hard_values.mean()

    total_tokens = easy_count + hard_count
    if total_tokens == 0:
        kd_loss = student_logits.sum() * 0.0
    else:
        kd_loss_float = (lambda_easy * easy_sum_val + (1.0 - lambda_easy) * hard_sum_val) / float(total_tokens)
        kd_loss = kd_loss_float.to(student_logits.dtype)

    metrics: Dict[str, float] = {
        "train/atkd_easy_tokens": float(easy_count),
        "train/atkd_hard_tokens": float(hard_count),
        "train/atkd_lambda_easy": lambda_easy,
        "train/atkd_hard_percent": hard_percent,
    }
    if hard_count > 0:
        metrics["train/atkd_unc_threshold"] = float(uncertainty[hard_mask].min().item())
    if easy_avg is not None:
        metrics["train/atkd_easy_loss"] = float(easy_avg.item())
    if hard_avg is not None:
        metrics["train/atkd_hard_loss"] = float(hard_avg.item())

    return kd_loss, metrics
