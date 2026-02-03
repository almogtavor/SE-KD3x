from __future__ import annotations

import math
from typing import Optional

import torch

from ...training.entropy_utils import token_entropy


class EntropyMixin:
    def _entropy_for_selection(self, input_ids: torch.Tensor, t_pred: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Return per-position entropy used only for *selection*.
        If cache is available for the whole batch, use cached H_hat (truncated entropy approximation).
        Otherwise, compute exact entropy online via t_pred.
        Output shape: [B, L-1]
        """
        # If t_pred is None, we are *expecting* to use the cache. Hard-fail on any miss.
        if t_pred is None:
            if self.cache is None:
                raise RuntimeError("Entropy requested with t_pred=None but no cache is set.")
            cache_mode = self._cache_mode()
            if cache_mode == "unc":
                raise RuntimeError("Entropy lookup requested but cache is configured for 'unc' mode (no entropy stored).")
            items = self._lookup_cache_batch(input_ids)
            if items is None:
                raise RuntimeError("Cache miss: at least one example not present in the offline cache.")
            H_list = []
            for it in items:
                if cache_mode == "entropy":
                    if "entropy_fp16" not in it:
                        raise RuntimeError("Cache item missing entropy_fp16 in entropy mode.")
                    H_list.append(torch.as_tensor(it["entropy_fp16"], dtype=torch.float16).float())
                else:
                    if "H_hat_u8" in it:
                        H_u8 = torch.as_tensor(it["H_hat_u8"], dtype=torch.uint8)
                        V = int(it.get("rs", {}).get("sentinel_id", 0))
                        H_cap = math.log(max(2, V)) if V > 0 else 1.0
                        H_f = (H_u8.float() / 255.0) * H_cap
                        H_list.append(H_f)
                    elif "H_hat" in it:
                        H_list.append(torch.as_tensor(it["H_hat"], dtype=torch.float16).float())
                    else:
                        raise RuntimeError("Cache item lacks H_hat / H_hat_u8; cache is incomplete.")
            return torch.stack(H_list, dim=0).to(self.student_device)
        assert t_pred is not None, "Exact entropy requested but teacher logits are unavailable."
        B = t_pred.size(0)
        ent_list = []
        for i in range(B):
            ent_list.append(token_entropy(t_pred[i]).to(self.student_device))  # [L-1]
        return torch.stack(ent_list, dim=0)  # [B, L-1]

    def _unc_from_cache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return p_t(g) stored in UNC cache mode as float32 tensor [B, L-1]."""
        if self.cache is None:
            raise RuntimeError("Uncertainty requested but no cache is available.")
        cache_mode = self._cache_mode()
        if cache_mode != "unc":
            raise RuntimeError(f"Uncertainty lookup requested but cache mode is '{cache_mode}'.")
        items = self._lookup_cache_batch(input_ids)
        if items is None:
            raise RuntimeError("Cache miss: at least one example not present in the offline cache.")
        ptg_list = []
        for it in items:
            data = it.get("target_prob_fp16")
            if data is None:
                raise RuntimeError("Cache item missing target_prob_fp16 data in unc mode.")
            ptg_list.append(torch.as_tensor(data, dtype=torch.float16).float())
        return torch.stack(ptg_list, dim=0).to(self.student_device)
