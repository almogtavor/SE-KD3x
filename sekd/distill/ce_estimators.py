from __future__ import annotations

import torch


def ce_is_estimator(
    s_rows: torch.Tensor,
    ids_U: torch.Tensor,
    probs_U: torch.Tensor,
    ids_M_shared: torch.Tensor,
    M_neg: int,
    y_rows: torch.Tensor,
) -> torch.Tensor:
    """Self-normalized IS estimate of CE over ``U ∪ M ∪ {y}`` at temperature 1."""

    P, V = s_rows.shape
    device = s_rows.device

    # Gold logits at T=1
    z_y = s_rows.gather(1, y_rows.view(-1, 1)).squeeze(1)  # [P]

    # Cached/top tokens
    if ids_U.numel() == 0:
        z_U = s_rows.new_zeros((P, 0))
        qU_raw = s_rows.new_zeros((P, 0))
        valid_U_mask = s_rows.new_zeros((P, 0), dtype=torch.bool)
        y_eq_U = s_rows.new_zeros((P, 0), dtype=torch.bool)
        y_in_U = torch.zeros(P, dtype=torch.bool, device=device)
    else:
        z_U = torch.gather(s_rows, 1, ids_U)
        qU_raw = probs_U.clamp_min(0.0)
        valid_U_mask = qU_raw > 0
        y_eq_U = (ids_U == y_rows.view(-1, 1)) & valid_U_mask
        y_in_U = y_eq_U.any(dim=1)

    qM_raw = 1.0 / float(max(1, V))

    if ids_M_shared.numel() == 0 or M_neg == 0:
        z_M = s_rows.new_zeros((P, 0))
    else:
        if ids_M_shared.dim() == 1:
            z_M = s_rows[:, ids_M_shared]
        else:
            z_M = torch.gather(s_rows, 1, ids_M_shared)

    # Importance weights (log-space) for each subset
    qy_raw = torch.where(y_in_U, (qU_raw * y_eq_U.float()).sum(dim=1), torch.full((P,), qM_raw, device=device, dtype=s_rows.dtype))
    qy = qy_raw.clamp_min(1e-12)
    log_w_y = z_y - qy.log()

    if ids_U.numel() == 0:
        log_w_U = s_rows.new_zeros((P, 0))
    else:
        qU_safe = qU_raw.clamp_min(1e-12)
        log_qU = qU_safe.log().to(z_U.dtype)
        log_w_U = torch.full_like(z_U, float('-inf'))
        include_mask = valid_U_mask & (~y_eq_U)
        log_w_U[include_mask] = z_U[include_mask] - log_qU[include_mask]

    if ids_M_shared.numel() == 0 or M_neg == 0:
        log_w_M = s_rows.new_zeros((P, 0))
    else:
        log_qM = torch.log(torch.tensor(qM_raw, device=device, dtype=s_rows.dtype)).item()
        log_w_M = z_M - log_qM

    parts = [log_w_y.unsqueeze(1)]
    if ids_U.numel() != 0:
        parts.append(log_w_U)
    if ids_M_shared.numel() != 0 and M_neg != 0:
        parts.append(log_w_M)

    log_sum_w = torch.logsumexp(torch.cat(parts, dim=1), dim=1)
    ce_hat = log_sum_w - log_w_y
    return ce_hat
