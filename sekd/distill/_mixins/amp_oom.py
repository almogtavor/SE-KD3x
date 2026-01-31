from __future__ import annotations

import torch


class AmpOomMixin:
    @staticmethod
    def _sanitize_logits(x: torch.Tensor, name: str) -> torch.Tensor:
        """Sanitize logits to prevent NaNs/Infs during training.
        We might train with lower precision (e.g., fp16), so instability might occur.
        """
        # cast to fp32 for stability, clamp, and replace NaN/Inf
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = torch.clamp(x, min=-1e4, max=1e4)
        # The implementation of x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4) was buggy, so we do manually:
        finite_mask = torch.isfinite(x)
        if not finite_mask.all():
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                x = x.masked_fill(nan_mask, 0.0)
            inf_mask = torch.isinf(x)
            if inf_mask.any():
                pos_inf_mask = inf_mask & (x > 0)
                if pos_inf_mask.any():
                    x = x.masked_fill(pos_inf_mask, 1e4)
                neg_inf_mask = inf_mask & (x < 0)
                if neg_inf_mask.any():
                    x = x.masked_fill(neg_inf_mask, -1e4)
            # ensure values remain within bounds after replacements
            x = torch.clamp(x, min=-1e4, max=1e4)
        if not torch.isfinite(x).all():
            print(f"[warn] non-finite after sanitize in {name}")
        if orig_dtype != torch.float32:
            x = x.to(orig_dtype)
        return x
