from __future__ import annotations

from typing import Optional

import torch


class GLSMixin:
    def _gls_init_if_needed(self):
        if not bool(getattr(self.config, "gls_enabled", False)):
            return
        if self._gls_buf is None:
            self._gls_cap = max(1, int(getattr(self.config, "gls_queue_size", 30000)))
            # CPU float32 buffer; we store only finite stats
            self._gls_buf = torch.empty(self._gls_cap, dtype=torch.float32, device="cpu")
            self._gls_count = 0
            self._gls_head = 0

    def _gls_push(self, values_1d: torch.Tensor):
        # values_1d: 1D CPU float32 finite values; may be empty
        if self._gls_buf is None:
            return
        if values_1d is None or values_1d.numel() == 0:
            return
        # ensure on CPU float32
        vals = values_1d.detach().to("cpu", dtype=torch.float32)
        if vals.numel() == 0:
            return
        n = int(vals.numel())
        idx = 0
        while idx < n:
            take = min(self._gls_cap - self._gls_head, n - idx)
            self._gls_buf[self._gls_head:self._gls_head + take] = vals[idx:idx + take]
            self._gls_head = (self._gls_head + take) % self._gls_cap
            self._gls_count = min(self._gls_cap, self._gls_count + take)
            idx += take

    def _gls_threshold(self, top_percent: float) -> Optional[float]:
        # Compute percentile over history BEFORE current batch is pushed.
        if self._gls_buf is None or self._gls_count == 0:
            return None
        # We want the threshold such that tokens with stat >= thr are in the top k_percent
        q = max(0.0, min(1.0, 1.0 - (top_percent / 100.0)))
        view = self._gls_buf if self._gls_count == self._gls_cap else self._gls_buf[:self._gls_count]
        try:
            thr = torch.quantile(view, q).item()
        except Exception:
            thr = float(view.min().item())
        return float(thr)
