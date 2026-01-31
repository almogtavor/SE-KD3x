"""Selective LM Head utilities for memory-efficient token-selective KD.

When teacher_selective_lm_head is enabled:
- Student: run base transformer with grad, compute entropy on all positions without grad
  to select top-k%, then apply lm_head with grad only on selected positions.
- Teacher: run base transformer without grad, apply lm_head only on selected positions.

This avoids materializing the full [B, T, V] logits tensors for both models.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _streaming_entropy_from_hidden_states(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Compute entropy in a streaming/chunked fashion to avoid full [B, T, V] materialization.

    Processes `chunk_size` positions at a time for GPU efficiency while keeping memory
    bounded. Default chunk_size=128 balances ~78 MB peak (for V=152k) with good throughput.

    Args:
        hidden_states: [B, T, D] hidden states (positions for next-token prediction).
        lm_head: The lm_head linear layer.
        chunk_size: Number of positions to process at a time (default 128).

    Returns:
        entropy: [B, T] entropy values (float32).
    """
    B, T, D = hidden_states.shape
    device = hidden_states.device
    entropy = torch.zeros(B, T, device=device, dtype=torch.float32)

    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        # Process chunk: [B, chunk, D] -> [B, chunk, V]
        hs_chunk = hidden_states[:, start:end, :]  # [B, chunk, D]
        logits_chunk = lm_head(hs_chunk)  # [B, chunk, V]
        log_probs = F.log_softmax(logits_chunk.float(), dim=-1)
        probs = log_probs.exp()
        ent_chunk = -(probs * log_probs).sum(dim=-1)  # [B, chunk]
        entropy[:, start:end] = ent_chunk
        # logits_chunk goes out of scope here, memory freed

    return entropy


def compute_student_entropy_and_select(
    student_hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    valid_next: torch.Tensor,
    k_percent: float,
    normalize_topk_by_length: bool = False,
    chunk_size: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute student entropy (no grad) on all positions and select top-k%.

    Uses chunked streaming entropy computation to avoid materializing full [B, T, V]
    logits. Processes `chunk_size` positions at a time for GPU efficiency.

    Args:
        student_hidden_states: [B, T, D] hidden states from student base transformer (has grad).
        lm_head: The student's lm_head linear layer.
        valid_next: [B, T-1] boolean mask of valid next-token positions.
        k_percent: Percentage of tokens to select (0-100).
        normalize_topk_by_length: Use batch-average length for quota.
        chunk_size: Number of positions per chunk for streaming entropy (default 128).

    Returns:
        selected_mask: [B, T-1] boolean mask of selected positions.
        student_entropy: [B, T-1] entropy values (detached).
    """
    B, T, D = student_hidden_states.shape

    # Chunked streaming entropy: process chunk_size positions at a time
    with torch.no_grad():
        hs_for_entropy = student_hidden_states[:, :-1, :]  # [B, T-1, D]
        entropy = _streaming_entropy_from_hidden_states(hs_for_entropy, lm_head, chunk_size)  # [B, T-1]

    # Select top-k% by entropy within valid positions
    pct = max(0.0, min(1.0, k_percent / 100.0))
    selected_mask = torch.zeros_like(valid_next, dtype=torch.bool)

    if normalize_topk_by_length:
        total_valid = int(valid_next.sum().item())
        avg_valid = total_valid / max(1, B)
        shared_quota = max(1, math.ceil(pct * avg_valid))
    else:
        shared_quota = None

    for i in range(B):
        mask_i = valid_next[i]
        n_valid = int(mask_i.sum().item())
        if n_valid < 1:
            continue
        if shared_quota is not None:
            k = min(n_valid, shared_quota)
        else:
            k = max(1, min(n_valid, math.ceil(pct * n_valid)))
        valid_idx = torch.where(mask_i)[0]
        scores = entropy[i][mask_i].float()
        if scores.numel() == 0:
            continue
        _, rel = torch.topk(scores, k=k, largest=True, sorted=False)
        sel_abs = valid_idx[rel]
        selected_mask[i, sel_abs] = True

    return selected_mask, entropy.detach()


def selective_student_logits(
    student_hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    selected_mask: torch.Tensor,
    contiguous_selected: bool = True,
) -> torch.Tensor:
    """Apply student lm_head WITH grad only on selected positions.

    Args:
        student_hidden_states: [B, T, D] from base transformer (has grad).
        lm_head: Student's lm_head.
        selected_mask: [B, T-1] boolean mask.
        contiguous_selected: If True, make selected_hs contiguous before lm_head.
            This can improve memory efficiency by allowing the original tensor
            to be freed if no other references exist.

    Returns:
        selected_logits: [N_selected, V] logits with grad for KD loss.
    """
    # Hidden states for next-token prediction (shift by 1)
    hs_pred = student_hidden_states[:, :-1, :]  # [B, T-1, D]
    # Gather selected hidden states
    batch_idx, pos_idx = torch.nonzero(selected_mask, as_tuple=True)
    if batch_idx.numel() == 0:
        V = lm_head.weight.shape[0] if hasattr(lm_head, 'weight') else lm_head.out_features
        return torch.zeros((0, V), device=student_hidden_states.device, dtype=student_hidden_states.dtype)
    selected_hs = hs_pred[batch_idx, pos_idx]  # [N_selected, D]
    # Make contiguous to allow freeing the original tensor's memory
    if contiguous_selected:
        selected_hs = selected_hs.contiguous()
    # Apply lm_head with grad
    selected_logits = lm_head(selected_hs)  # [N_selected, V]
    return selected_logits


def selective_teacher_logits(
    teacher_model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    selected_mask: torch.Tensor,
    teacher_device: torch.device,
    amp_enabled: bool = False,
    amp_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Run teacher base transformer and apply lm_head only on selected positions.

    Args:
        teacher_model: The full teacher model (must have .model and .lm_head).
        input_ids: [B, T] input token IDs.
        attention_mask: [B, T] attention mask.
        selected_mask: [B, T-1] boolean mask of positions to compute logits for.
        teacher_device: Device where teacher resides.
        amp_enabled: Whether to use AMP.
        amp_dtype: AMP dtype.

    Returns:
        teacher_selected_logits: [N_selected, V_teacher] logits (no grad).
    """
    from torch.cuda.amp import autocast

    input_ids_t = input_ids.to(teacher_device)
    attn_t = attention_mask.to(teacher_device)

    with torch.no_grad():
        with autocast(enabled=amp_enabled, dtype=amp_dtype):
            # Run base transformer only (output hidden states, no lm_head)
            base_model = teacher_model.model  # Qwen3 uses .model for the transformer
            outputs = base_model(
                input_ids_t,
                attention_mask=attn_t,
                output_hidden_states=False,
            )
            hidden_states = outputs.last_hidden_state  # [B, T, D_teacher]

            # Gather hidden states at selected positions (shifted for next-token)
            hs_pred = hidden_states[:, :-1, :]  # [B, T-1, D_teacher]
            selected_mask_t = selected_mask.to(teacher_device)
            batch_idx, pos_idx = torch.nonzero(selected_mask_t, as_tuple=True)

            if batch_idx.numel() == 0:
                V_teacher = teacher_model.lm_head.weight.shape[0]
                return torch.zeros(
                    (0, V_teacher), device=teacher_device, dtype=hidden_states.dtype
                )

            selected_hs = hs_pred[batch_idx, pos_idx]  # [N_selected, D_teacher]

            # Apply lm_head only on selected positions
            teacher_logits = teacher_model.lm_head(selected_hs)  # [N_selected, V_teacher]

    return teacher_logits


def log_peak_memory(device: torch.device) -> float:
    """Get peak memory allocated in GB and reset stats.

    Call torch.cuda.reset_peak_memory_stats() before the step,
    then call this after the step.

    Returns:
        Peak memory in GB.
    """
    if device.type != "cuda":
        return 0.0
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 3)
