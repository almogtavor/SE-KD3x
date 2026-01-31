"""Compute-skip optimizations for selective knowledge distillation.

This module provides utilities for:
1. Truncating sequences to the last selected position (cut_after_last_selected)
2. Computing logits only for selected positions (logits_on_selected_only)

These optimizations reduce compute when distilling on a small fraction of tokens.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


class SelectiveForwardHelper:
    """Helper for compute-skip optimizations in selective KD.
    
    This class provides methods to:
    1. Pre-compute which positions will be selected (from cached entropy/uncertainty)
    2. Truncate sequences before transformer forward
    3. Compute logits only on selected positions
    """
    
    def __init__(
        self,
        config,
        student_device: torch.device,
        cached_items: Optional[List[Dict]] = None,
    ):
        self.config = config
        self.student_device = student_device
        self.cached_items = cached_items
        
        self.cut_after_last_selected = bool(getattr(config, "cut_after_last_selected", False))
        self.logits_on_selected_only = bool(getattr(config, "logits_on_selected_only", False))
        
        # Pre-computed selection info
        self.selected_positions: Optional[List[Tuple[int, int]]] = None
        self.max_selected_per_batch: Optional[List[int]] = None
        self.truncated_length: Optional[int] = None
        
    @property
    def optimizations_enabled(self) -> bool:
        """Check if any optimizations are enabled."""
        return self.cut_after_last_selected or self.logits_on_selected_only
    
    @property
    def has_cached_entropy(self) -> bool:
        """Check if cached entropy/uncertainty is available for pre-selection."""
        return self.cached_items is not None and len(self.cached_items) > 0
    
    def precompute_selections_from_cache(
        self,
        valid_next: torch.Tensor,
        k_percent: float,
        alpha: float = 1.0,
        normalize_topk: bool = False,
    ) -> Optional[torch.Tensor]:
        """Pre-compute selected positions from cached entropy/uncertainty.
        
        Args:
            valid_next: [B, L-1] mask of valid positions
            k_percent: Percentage of positions to select (0-100)
            alpha: RS-KD alpha parameter for softmax temperature
            normalize_topk: Whether to use batch-normalized quota
            
        Returns:
            keep_mask: [B, L-1] boolean mask of selected positions, or None if caching unavailable
        """
        if not self.has_cached_entropy:
            return None
            
        B = valid_next.size(0)
        L_minus_1 = valid_next.size(1)
        
        pct = max(0.0, min(1.0, k_percent / 100.0))
        
        # Extract cached entropy/uncertainty
        cache_mode = getattr(self.config, "offline_cache_mode", "entropy")
        selection_scores = torch.zeros((B, L_minus_1), device=self.student_device, dtype=torch.float32)
        
        for b in range(B):
            if b >= len(self.cached_items) or self.cached_items[b] is None:
                continue
            item = self.cached_items[b]
            
            if cache_mode == "unc":
                # Use cached target probabilities to compute uncertainty
                if "target_prob_fp16" in item:
                    target_probs = torch.as_tensor(
                        item["target_prob_fp16"], 
                        dtype=torch.float16
                    ).to(self.student_device).float()
                    # unc = 1 - max(p), but we have p(target), so unc ≈ 1 - p(target)
                    unc = 1.0 - target_probs[:L_minus_1]
                    selection_scores[b, :len(unc)] = unc
            else:
                # Use cached entropy
                if "H_u8" in item:
                    H_u8 = torch.as_tensor(item["H_u8"], dtype=torch.uint8).to(self.student_device)
                    H = H_u8.float() / 25.5  # Decode from uint8
                    selection_scores[b, :len(H)] = H[:L_minus_1] if len(H) > L_minus_1 else H
                elif "H_fp16" in item:
                    H = torch.as_tensor(item["H_fp16"], dtype=torch.float16).to(self.student_device).float()
                    selection_scores[b, :len(H)] = H[:L_minus_1] if len(H) > L_minus_1 else H
        
        # Now sample positions using the same logic as pos-rs-kd
        keep_mask = torch.zeros_like(valid_next, dtype=torch.bool)
        selected_positions = []
        max_selected_per_batch = []
        
        q_floor = float(getattr(self.config, "rs_floor", 1e-6))
        
        # Compute shared quota if normalize_topk
        shared_quota = None
        if normalize_topk:
            total_valid = int(valid_next.sum().item())
            avg_valid = total_valid / max(1, B)
            shared_quota = max(1, math.ceil(pct * avg_valid))
        
        for b in range(B):
            valid_next_b = valid_next[b]
            n_valid = int(valid_next_b.sum().item())
            if n_valid < 1:
                max_selected_per_batch.append(0)
                continue
            
            vec = selection_scores[b][valid_next_b].float()
            valid_idx = torch.where(valid_next_b)[0]
            
            if normalize_topk and shared_quota is not None:
                k_count = min(n_valid, shared_quota)
            else:
                k_count = max(1, min(n_valid, math.ceil(pct * n_valid)))
            
            # Build sampling distribution
            vec = torch.clamp(vec, min=1e-8)
            logits = vec if alpha == 1.0 else vec * alpha
            q = torch.softmax(logits, dim=0)
            if not torch.isfinite(q).all():
                q = torch.full_like(vec, 1.0 / max(1, vec.numel()))
            else:
                q = torch.clamp(q, min=q_floor)
                q = q / q.sum().clamp_min(1e-12)
            
            # Sample positions
            sel_rel = torch.multinomial(q, num_samples=k_count, replacement=True)
            selected_abs = valid_idx[sel_rel]
            
            for pos in selected_abs:
                pos_val = int(pos.item())
                keep_mask[b, pos_val] = True
                selected_positions.append((b, pos_val))
            
            max_pos = int(selected_abs.max().item()) if len(selected_abs) > 0 else 0
            max_selected_per_batch.append(max_pos)
        
        self.selected_positions = selected_positions
        self.max_selected_per_batch = max_selected_per_batch
        
        # Compute truncated length (max of all max positions + 2 for safety with next-token prediction)
        if max_selected_per_batch:
            self.truncated_length = max(max_selected_per_batch) + 2
        
        return keep_mask
    
    def truncate_inputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Truncate inputs to the last selected position.
        
        Returns:
            truncated_input_ids: [B, L'] where L' <= L
            truncated_attention_mask: [B, L']
            original_length: Original L for reference
        """
        original_length = input_ids.size(1)
        
        if not self.cut_after_last_selected or self.truncated_length is None:
            return input_ids, attention_mask, original_length
        
        # Truncate to computed length
        new_length = min(self.truncated_length, original_length)
        
        return (
            input_ids[:, :new_length],
            attention_mask[:, :new_length],
            original_length,
        )
    
    def compute_logits_selected_only(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        autocast_ctx,
        amp_enabled: bool,
        amp_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute logits only for selected positions.
        
        This avoids materializing the full [B, T, V] logits tensor.
        
        Returns:
            s_pred: Full logits tensor [B, T-1, V] (with zeros at non-selected positions)
            selected_logits: Logits only at selected positions [N_sel, V]
        """
        if not self.logits_on_selected_only or self.selected_positions is None:
            # Fall back to normal forward
            with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
                logits = model(input_ids, attention_mask=attention_mask).logits
            return logits[:, :-1, :], None
        
        # Get hidden states from base transformer
        B, T = input_ids.shape
        V = model.config.vocab_size if hasattr(model.config, 'vocab_size') else model.lm_head.out_features
        
        # Access the base transformer (model.model for most HF models)
        base_model = getattr(model, 'model', None)
        if base_model is None:
            # Fall back if we can't find base model
            with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
                logits = model(input_ids, attention_mask=attention_mask).logits
            return logits[:, :-1, :], None
        
        with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
            # Run base transformer to get hidden states
            outputs = base_model(
                input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state  # [B, T, H]
        
        # Gather hidden states for selected positions
        if len(self.selected_positions) == 0:
            # No selections - return zeros
            s_pred = torch.zeros((B, T - 1, V), device=input_ids.device, dtype=hidden_states.dtype)
            return s_pred, None
        
        b_indices = torch.tensor([p[0] for p in self.selected_positions], device=input_ids.device)
        t_indices = torch.tensor([p[1] for p in self.selected_positions], device=input_ids.device)
        
        # Gather selected hidden states [N_sel, H]
        h_selected = hidden_states[b_indices, t_indices, :]
        
        # Compute logits only for selected positions
        with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
            selected_logits = model.lm_head(h_selected)  # [N_sel, V]
        
        # Create sparse output tensor with zeros at non-selected positions
        s_pred = torch.zeros((B, T - 1, V), device=input_ids.device, dtype=selected_logits.dtype)
        s_pred[b_indices, t_indices, :] = selected_logits
        
        return s_pred, selected_logits


def log_compute_skip_metrics(
    logger,
    global_step: int,
    original_length: int,
    truncated_length: Optional[int],
    n_selected: int,
    n_total_valid: int,
) -> None:
    """Log metrics about compute savings from optimizations."""
    if logger is None:
        return
    
    metrics = {
        "compute_skip/original_length": original_length,
        "compute_skip/n_selected": n_selected,
        "compute_skip/n_total_valid": n_total_valid,
        "compute_skip/selection_ratio": n_selected / max(1, n_total_valid),
    }
    
    if truncated_length is not None:
        metrics["compute_skip/truncated_length"] = truncated_length
        metrics["compute_skip/length_reduction"] = 1.0 - (truncated_length / max(1, original_length))
    
    logger.log(metrics, global_step)
