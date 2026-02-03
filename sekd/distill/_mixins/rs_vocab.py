from __future__ import annotations

import torch


def _kl_importance_estimate(
    t_logp_sel: torch.Tensor,
    s_logp_sel: torch.Tensor,
    q_sel: torch.Tensor,
    *,
    self_norm: bool = True,
    T_kd: float = 1.0,
) -> torch.Tensor:
    """Importance-sampled KL estimate for a single position."""
    with torch.no_grad():
        p1 = t_logp_sel.exp()
        gamma = 1.0 / max(1e-12, T_kd)
        pT_un = p1.pow(gamma)
        w = pT_un / q_sel.clamp_min(1e-12)
        if self_norm:
            w = w / w.sum().clamp_min(1e-12)

    logpT_approx = torch.log(pT_un) - torch.log(pT_un.sum().clamp_min(1e-12))
    diff = logpT_approx - s_logp_sel
    if self_norm:
        return (w * diff).sum()
    return (w * diff).mean()
