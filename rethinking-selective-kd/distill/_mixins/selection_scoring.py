from __future__ import annotations

from typing import Any, Dict, Optional

import torch


class SelectionScoringMixin:
    @staticmethod
    def _normalize_component(values: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
        """Normalize `values` over the True positions of `mask` according to `mode`.

        Returns a tensor of the same shape as `values` (no masking applied).
        """
        if mode == "none":
            return values

        valid_vals = values[mask]
        if valid_vals.numel() == 0:
            return values

        if mode == "z":
            mean = valid_vals.mean()
            std = valid_vals.std(unbiased=False)
            if std < 1e-6:
                return values - mean
            return (values - mean) / std
        elif mode == "minmax":
            min_val = valid_vals.min()
            max_val = valid_vals.max()
            denom = (max_val - min_val).clamp_min(1e-6)
            return (values - min_val) / denom
        else:
            raise ValueError(f"Unknown normalization mode: {mode}")

    def _prepare_score_context(
        self,
        ent_raw: torch.Tensor,
        kl_pos: torch.Tensor,
        s_log_probs: Optional[torch.Tensor],
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        student_ce_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        weights = (
            float(getattr(self.config, "score_entropy_weight", 1.0)),
            float(getattr(self.config, "score_ce_weight", 1.0)),
            float(getattr(self.config, "score_kl_weight", 1.0)),
        )
        if all(abs(w) < 1e-8 for w in weights):
            raise ValueError("Score-based selection requires at least one non-zero component weight.")

        norm_mode = getattr(self.config, "score_normalize", "z")

        # entropy / kl components
        ent_for_score = ent_raw.detach().masked_fill(~valid_next, 0.0)
        kl_for_score = kl_pos.detach().masked_fill(~valid_next, 0.0)

        # student CE component
        if student_ce_override is not None:
            student_ce = student_ce_override.detach().masked_fill(~valid_next, 0.0)
        else:
            if s_log_probs is None:
                raise ValueError("s_log_probs is required unless student_ce_override is provided.")
            targets = input_ids[:, 1:].to(self.student_device)
            V = s_log_probs.size(-1)
            invalid_mask = (targets < 0) | (targets >= V)
            invalid_count = int(invalid_mask.sum().item())
            if invalid_count > 0:
                if not getattr(self, "_warned_selection_invalid_targets", False):
                    bad_vals = targets[invalid_mask].detach().to("cpu", non_blocking=True)
                    min_bad = int(bad_vals.min().item()) if bad_vals.numel() else 0
                    max_bad = int(bad_vals.max().item()) if bad_vals.numel() else 0
                    sample_vals = bad_vals.unique()
                    if sample_vals.numel() > 5:
                        sample_vals = sample_vals[:5]
                    sample_list = ", ".join(str(int(v.item())) for v in sample_vals)
                    print(
                        "[warn] Selection-scoring targets out of range "
                        f"(count={invalid_count}, min={min_bad}, max={max_bad}, sample=[{sample_list}]) "
                        "→ skipping those positions.",
                        flush=True,
                    )
                    self._warned_selection_invalid_targets = True
                valid_next = valid_next & (~invalid_mask)
                ent_for_score = ent_for_score.masked_fill(invalid_mask, 0.0)
                kl_for_score = kl_for_score.masked_fill(invalid_mask, 0.0)
                targets = targets.masked_fill(invalid_mask, 0)
            targets = targets.masked_fill(~valid_next, 0)
            targets = targets.clamp(min=0, max=V - 1)
            target_logp = s_log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            student_ce = (-target_logp).detach().masked_fill(~valid_next, 0.0)

        return {
            "weights": weights,
            "norm_mode": norm_mode,
            "entropy": ent_for_score,
            "student_ce": student_ce,
            "kl": kl_for_score,
        }

    def _build_score_vector(
        self,
        score_ctx: Dict[str, Any],
        example_idx: int,
        mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Construct combined score for one example or return None if no components are active."""
        weights = score_ctx["weights"]
        norm_mode = score_ctx["norm_mode"]

        combined = torch.zeros_like(score_ctx["entropy"][example_idx])
        component_used = False

        if abs(weights[0]) > 0:
            comp = self._normalize_component(score_ctx["entropy"][example_idx].clone(), mask, norm_mode)
            combined = combined + weights[0] * comp
            component_used = True

        if abs(weights[1]) > 0:
            comp = self._normalize_component(score_ctx["student_ce"][example_idx].clone(), mask, norm_mode)
            combined = combined + weights[1] * comp
            component_used = True

        if abs(weights[2]) > 0:
            comp = self._normalize_component(score_ctx["kl"][example_idx].clone(), mask, norm_mode)
            combined = combined + weights[2] * comp
            component_used = True

        if not component_used:
            return None

        combined = combined.masked_fill(~mask, float('-inf'))
        return combined
