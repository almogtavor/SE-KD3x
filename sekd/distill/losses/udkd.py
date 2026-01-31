"""
Uncertainty-Driven Decoupled Knowledge Distillation (UDKD) Loss.

This module implements several variants of UDKD where the uncertainty gate can be
derived from teacher or student probabilities, or from their KL divergence:
1. UDKD-UNC: g_t = 1 - p_teacher(target)
2. UDKD-ENT: g_t = H_teacher(p) / log(|V|)
3. UDKD-STUDENT-ENT: g_t = H_student(q) / log(|V|)
4. UDKD-KL: g_t = KL(p_teacher || q_student)
5. UDKD-REVERSE-KL: g_t = KL(q_student || p_teacher)

All variants compute:
- TCKD (Target "gold" Class KD): -p_g * log(q_g) for target class
- NCKD (Non-target Class KD): KL divergence over normalized non-target distributions

Final loss:
    L = L_NCKD + gate * L_TCKD
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class UDKDUncertaintyMetric(Enum):
    """Uncertainty metric for UDKD loss."""
    UNC = "unc"      # 1 - p(target)
    ENTROPY = "entropy"  # H_teacher(p) / log(|V|)
    STUDENT_ENTROPY = "student_entropy"  # H_student(q) / log(|V|)
    KL = "kl"  # KL(teacher || student)
    REVERSE_KL = "reverse_kl"  # KL(student || teacher)


class UDKDLoss:
    """Uncertainty-Driven Decoupled Knowledge Distillation Loss.
    
    This class computes the UDKD loss which combines:
    - TCKD: Target Class KD loss (cross-entropy on target class)
    - NCKD: Non-target Class KD loss (KL on non-target distribution)
    
    The combination is weighted by an uncertainty gate:
    - UNC mode: gate = 1 - p(target)
    - ENT mode: gate = entropy / log(vocab_size)
    """
    
    def __init__(
        self,
        uncertainty_metric: str = "unc",
        eps: float = 1e-12,
    ):
        """Initialize UDKD loss.
        
        Args:
            uncertainty_metric: Determines how the uncertainty gate is computed
            eps: Small constant for numerical stability
        """
        valid_metrics = {m.value for m in UDKDUncertaintyMetric}
        if uncertainty_metric not in valid_metrics:
            raise ValueError(f"uncertainty_metric must be one of {sorted(valid_metrics)}, got {uncertainty_metric}")
        self.uncertainty_metric = UDKDUncertaintyMetric(uncertainty_metric)
        self.eps = eps
    
    def compute_tckd(
        self,
        teacher_probs: torch.Tensor,
        student_probs: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Target Class KD loss: -p_g * log(q_g).
        
        Args:
            teacher_probs: Teacher probabilities [N, V]
            student_probs: Student probabilities [N, V]
            target_ids: Ground truth target token IDs [N]
            
        Returns:
            TCKD loss per position [N]
        """
        # Get teacher and student probability for target class
        row_indices = torch.arange(teacher_probs.size(0), device=teacher_probs.device)
        teacher_target_prob = teacher_probs[row_indices, target_ids]  # p_g [N]
        student_target_prob = student_probs[row_indices, target_ids]  # q_g [N]
        
        # L_TCKD = -p_g * log(q_g)
        tckd_loss = -teacher_target_prob * torch.log(student_target_prob.clamp_min(self.eps))
        
        return tckd_loss
    
    def compute_nckd(
        self,
        teacher_probs: torch.Tensor,
        student_probs: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Non-target Class KD loss: KL(p_hat || q_hat) over non-target classes.
        
        p_hat_i = p_i / (1 - p_g) for i != g
        q_hat_i = q_i / (1 - q_g) for i != g
        
        Args:
            teacher_probs: Teacher probabilities [N, V]
            student_probs: Student probabilities [N, V]
            target_ids: Ground truth target token IDs [N]
            
        Returns:
            NCKD loss per position [N]
        """
        N, V = teacher_probs.shape
        row_indices = torch.arange(N, device=teacher_probs.device)
                
        # Zero out target class to get non-target distribution
        teacher_nontarget = teacher_probs.clone()
        student_nontarget = student_probs.clone()
        teacher_nontarget[row_indices, target_ids] = 0.0
        student_nontarget[row_indices, target_ids] = 0.0

        # Sum the remaining mass; skip rows with no non-target support to avoid 0/0 explosions
        teacher_nontarget_sum = teacher_nontarget.sum(dim=-1, keepdim=True)
        student_nontarget_sum = student_nontarget.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        valid_rows = (teacher_nontarget_sum.squeeze(-1) > self.eps)
        nckd_loss = teacher_probs.new_zeros(N)

        if valid_rows.any():
            idx = valid_rows.nonzero(as_tuple=False).squeeze(-1)
            teacher_hat = teacher_nontarget[idx] / teacher_nontarget_sum[idx].clamp_min(self.eps)
            student_hat = student_nontarget[idx] / student_nontarget_sum[idx]

            # KL(p_hat || q_hat) = sum_i p_hat_i * log(p_hat_i / q_hat_i)
            log_teacher_hat = torch.log(teacher_hat.clamp_min(self.eps))
            log_student_hat = torch.log(student_hat.clamp_min(self.eps))

            kl_vals = F.kl_div(
                log_student_hat,
                log_teacher_hat,
                log_target=True,
                reduction="none",
            ).sum(dim=-1)
            nckd_loss[idx] = kl_vals

        return nckd_loss
    
    def compute_uncertainty_gate(
        self,
        teacher_probs: torch.Tensor,
        student_probs: Optional[torch.Tensor],
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute uncertainty gate based on the selected metric.
        
        Args:
            teacher_probs: Teacher probabilities [N, V]
            student_probs: Student probabilities [N, V] (required for some metrics)
            target_ids: Ground truth target token IDs [N]
            
        Returns:
            Uncertainty gate values [N]
        """
        metric = self.uncertainty_metric
        if metric == UDKDUncertaintyMetric.UNC:
            # g_t = 1 - p_g (uncertainty = 1 - probability of target)
            row_indices = torch.arange(teacher_probs.size(0), device=teacher_probs.device)
            teacher_target_prob = teacher_probs[row_indices, target_ids]
            uncertainty_gate = 1.0 - teacher_target_prob
        elif metric == UDKDUncertaintyMetric.ENTROPY:
            # g_t = H(p) / log(|V|) (normalized entropy)
            vocab_size = teacher_probs.size(-1)
            max_entropy = math.log(vocab_size)
            
            # H(p) = -sum_i p_i * log(p_i)
            log_probs = torch.log(teacher_probs.clamp_min(self.eps))
            entropy = -(teacher_probs * log_probs).sum(dim=-1)  # [N]
            
            # Normalize to [0, 1]
            uncertainty_gate = entropy / max_entropy
        elif metric == UDKDUncertaintyMetric.STUDENT_ENTROPY:
            if student_probs is None:
                raise ValueError("student_probs must be provided for student_entropy gate")
            vocab_size = student_probs.size(-1)
            max_entropy = math.log(vocab_size)
            log_probs = torch.log(student_probs.clamp_min(self.eps))
            entropy = -(student_probs * log_probs).sum(dim=-1)
            uncertainty_gate = entropy / max_entropy
        else:
            if student_probs is None:
                raise ValueError("student_probs must be provided for KL-based gates")
            log_teacher = torch.log(teacher_probs.clamp_min(self.eps))
            log_student = torch.log(student_probs.clamp_min(self.eps))
            if metric == UDKDUncertaintyMetric.KL:
                # KL(p || q) = sum_i p_i * (log p_i - log q_i)
                uncertainty_gate = (teacher_probs * (log_teacher - log_student)).sum(dim=-1)
            elif metric == UDKDUncertaintyMetric.REVERSE_KL:
                # KL(q || p) = sum_i q_i * (log q_i - log p_i)
                uncertainty_gate = (student_probs * (log_student - log_teacher)).sum(dim=-1)
            else:
                raise ValueError(f"Unsupported uncertainty metric: {metric}")

        return uncertainty_gate
    
    def forward(
        self,
        teacher_probs: torch.Tensor,
        student_probs: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute UDKD loss.
        
        UDKD loss = L_NCKD + uncertainty_gate * L_TCKD
        
        Args:
            teacher_probs: Teacher probabilities [N, V]
            student_probs: Student probabilities [N, V]
            target_ids: Ground truth target token IDs [N]
            
        Returns:
            Tuple of (total_loss, tckd_loss, nckd_loss, uncertainty_gate), all [N]
        """
        # Compute components
        tckd_loss = self.compute_tckd(teacher_probs, student_probs, target_ids)
        nckd_loss = self.compute_nckd(teacher_probs, student_probs, target_ids)
        uncertainty_gate = self.compute_uncertainty_gate(teacher_probs, student_probs, target_ids)
        
        # Final loss: L_NCKD + g * L_TCKD
        total_loss = nckd_loss + uncertainty_gate * tckd_loss
        
        return total_loss, tckd_loss, nckd_loss, uncertainty_gate
    
    def __call__(
        self,
        teacher_probs: torch.Tensor,
        student_probs: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute UDKD loss (callable interface)."""
        return self.forward(teacher_probs, student_probs, target_ids)


def compute_udkd_loss(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
    target_ids: torch.Tensor,
    uncertainty_metric: str = "unc",
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience function to compute UDKD loss.
    
    Args:
        teacher_probs: Teacher probabilities [N, V]
        student_probs: Student probabilities [N, V]
        target_ids: Ground truth target token IDs [N]
        uncertainty_metric: See UDKDUncertaintyMetric for options ("unc", "entropy", "student_entropy", "kl", "reverse_kl")
        eps: Small constant for numerical stability
        
    Returns:
        Tuple of (total_loss, tckd_loss, nckd_loss, uncertainty_gate), all [N]
    """
    loss_fn = UDKDLoss(uncertainty_metric=uncertainty_metric, eps=eps)
    return loss_fn(teacher_probs, student_probs, target_ids)
