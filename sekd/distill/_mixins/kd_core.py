from __future__ import annotations

import torch
import torch.nn.functional as F


class KDCoreMixin:
    @staticmethod
    def _kl_loss(log_p: torch.Tensor, log_q: torch.Tensor, chunk_tokens: int = 0):
        """KL(P||Q) where log_p are teacher log-probs, log_q are student log-probs.

        If tensors already share a device we compute in one shot. Otherwise we stream
        teacher chunks to the student's device to avoid a full copy of V-sized rows.
        """
        if log_p.device == log_q.device:
            log_q32 = log_q.float()
            log_p32 = log_p.float()
            kl = F.kl_div(log_q32, log_p32, log_target=True, reduction="none").sum(dim=-1)
            return kl.to(log_q.dtype)

        # Different devices: stream small chunks of teacher rows onto student device.
        V = log_q.size(-1)
        flat_q32 = log_q.float().reshape(-1, V)
        flat_p = log_p.reshape(-1, V)
        total = flat_q32.size(0)
        if chunk_tokens <= 0:
            # By default keep ~32 token rows per chunk (≈20 MB for V≈150k).
            chunk_tokens = max(1, min(total, 32))

        out_chunks = []
        device_q = log_q.device
        for start in range(0, total, chunk_tokens):
            end = min(total, start + chunk_tokens)
            p_chunk = flat_p[start:end].to(device_q, non_blocking=True).float()
            q_chunk = flat_q32[start:end]
            kl_chunk = F.kl_div(q_chunk, p_chunk, log_target=True, reduction="none").sum(dim=-1)
            out_chunks.append(kl_chunk.to(log_q.dtype))

        kl_flat = torch.cat(out_chunks, dim=0)
        return kl_flat.view(*log_q.shape[:-1])

    @staticmethod
    def _proposal_sample_negatives(V: int, M: int, device: torch.device) -> torch.Tensor:
        """Draw M uniform 'negatives' from [0, V). Overlap with U/avoid is allowed."""
        if M <= 0 or V <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        return torch.randint(low=0, high=V, size=(M,), device=device, dtype=torch.long)

    @staticmethod
    def _student_log_probs_sampled(z_sel: torch.Tensor, q_sel: torch.Tensor, T: float) -> torch.Tensor:
        """Return log softmax over the sampled set S with importance correction: log s(i) ∝ z_i/T - log q(i)."""
        z_corr = z_sel / T - torch.log(q_sel.clamp_min(1e-12))
        logZ = torch.logsumexp(z_corr, dim=-1, keepdim=False)
        return z_corr - logZ
