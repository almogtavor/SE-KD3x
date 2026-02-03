import torch


def token_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Compute per-token entropy (natural log base) row by row to limit peak memory.

    Accepts logits with shape [..., V] and returns entropies with shape [...].
    """
    x = logits.float()
    seq_shape = x.shape[:-1]
    V = x.shape[-1]
    flat = x.reshape(-1, V)
    out = torch.empty(flat.size(0), dtype=x.dtype, device=x.device)

    blocks = 32
    block_size = max(1, (flat.size(0) + blocks - 1) // blocks)
    start = 0
    while start < flat.size(0):
        end = min(flat.size(0), start + block_size)
        block = flat[start:end]
        probs = torch.softmax(block, dim=-1)
        log_probs = probs.clamp_min(1e-12).log()
        out[start:end] = -(probs * log_probs).sum(dim=-1)
        start = end

    return out.reshape(*seq_shape)

def truncated_entropy_topk_tail_midpoint(logits: torch.Tensor, k: int = 20) -> torch.Tensor:
    x = logits.float()
    topk_logits, _ = torch.topk(x, k=k, dim=-1)
    Z = torch.logsumexp(x, dim=-1, keepdim=True)
    p_topk = torch.exp(topk_logits - Z)
    p_tail = (1.0 - p_topk.sum(-1)).clamp_min(0.0)

    p_topk_safe = p_topk.clamp_min(1e-12)
    h_topk = -(p_topk * p_topk_safe.log()).sum(-1)

    # lower: single bucket
    lb = h_topk + (-(p_tail * p_tail.clamp_min(1e-12).log()))
    # upper: uniform tail over V-k
    V_minus_k = x.size(-1) - k
    ub = h_topk + (-(p_tail * p_tail.clamp_min(1e-12).log()) + p_tail * torch.log(torch.tensor(V_minus_k, dtype=x.dtype, device=x.device)))
    return 0.5 * (lb + ub)


def truncated_entropy_topk_tail_uniform(logits: torch.Tensor, k: int = 20) -> torch.Tensor:
    # Top-m + uniform tail (upper bound on the true entropy of the tail)
    x = logits.float()                                 # [L, V]
    topk_logits, _ = torch.topk(x, k=k, dim=-1)        # [L, k]
    Z = torch.logsumexp(x, dim=-1, keepdim=True)       # [L, 1]
    p_topk = torch.exp(topk_logits - Z)                # [L, k]
    p_tail = (1.0 - p_topk.sum(-1)).clamp_min(0.0)     # [L]

    p_topk_safe = p_topk.clamp_min(1e-12)
    h_topk = -(p_topk * p_topk_safe.log()).sum(-1)     # [L]

    V_minus_k = x.size(-1) - k
    # uniform tail entropy contribution
    tail_term = (-p_tail * p_tail.clamp_min(1e-12).log()) + p_tail * torch.log(torch.tensor(V_minus_k, dtype=x.dtype, device=x.device))
    return h_topk + tail_term


# truncated entropy lower bound (Top-k + Tail), base-e (natural log)
def truncated_entropy_topk_tail(logits: torch.Tensor, k: int = 20) -> torch.Tensor:
    """
    H_hat = - sum_{j=1..k} p_j ln p_j - p_tail ln p_tail
    logits: [L, V]
    returns: [L]  (natural log base; multiply by log(2) if you want bits later)
    """
    # stable top-k on logits then convert to probabilities
    topk_vals, _ = torch.topk(logits, k=k, dim=-1)  # [L, k]
    # compute logsumexp for normalization using the same k window via trick:
    # We need full Z = logsumexp over V; approximate Z via topk + tail mass estimate:
    # Here we'll get probabilities by softmax over ALL via logits (safe enough with autocast disabled offline).
    # This function is only used offline/cached, so do the exact softmax:
    p_full = torch.softmax(logits, dim=-1)  # [L, V]
    p_topk, _ = torch.topk(p_full, k=k, dim=-1)  # [L, k]
    p_sum_topk = p_topk.sum(-1)  # [L]
    p_tail = (1.0 - p_sum_topk).clamp_min(0.0)  # [L]
    # -sum p log p on topk
    h_topk = -(p_topk * (p_topk.clamp_min(1e-12)).log()).sum(-1)  # [L]
    # tail bucket
    h_tail = -(p_tail * (p_tail.clamp_min(1e-12)).log())  # [L]
    return h_topk + h_tail  # [L]


@torch.no_grad()
def entropy_topm_plus_tail_is(
    logits: torch.Tensor,
    m: int = 20,
    s: int = 5,
) -> torch.Tensor:
    """
    Approx entropy with exact head (top-(m-s)) and IS-estimated tail using s samples.
    Returns natural-log entropy per position. Shape: [L] if logits is [L, V].
    """
    assert m > 0 and s >= 0 and s <= m, "Require 0 <= s <= m"
    x = logits.float()
    L, V = x.shape

    k_head = max(0, m - s)
    if k_head > 0:
        topk_logits, topk_idx = torch.topk(x, k=k_head, dim=-1)
    else:
        topk_logits = x.new_zeros((L, 0))
        topk_idx = torch.empty((L, 0), dtype=torch.long, device=x.device)

    Z = torch.logsumexp(x, dim=-1, keepdim=True)

    if k_head > 0:
        p_head = torch.exp(topk_logits - Z)
        H_head = -(p_head * p_head.clamp_min(1e-12).log()).sum(-1)
    else:
        H_head = torch.zeros(L, dtype=x.dtype, device=x.device)

    logp_full = x - Z
    p_full = torch.exp(logp_full)

    if k_head > 0:
        tail_mask = torch.ones_like(p_full, dtype=torch.bool)
        tail_mask.scatter_(1, topk_idx, False)
        p_tail_vec = p_full.masked_fill(~tail_mask, 0.0)
    else:
        p_tail_vec = p_full

    P_T = p_tail_vec.sum(-1)
    no_tail = P_T.le(1e-12)

    if s > 0:
        q = torch.where(
            P_T.unsqueeze(-1) > 0,
            p_tail_vec / P_T.unsqueeze(-1),
            p_tail_vec,
        )

        logp_samples = []
        for i in range(L):
            if no_tail[i] or (q[i].sum() <= 0):
                logp_samples.append(torch.tensor(0.0, device=x.device, dtype=x.dtype))
                continue
            idx_i = torch.multinomial(q[i], num_samples=s, replacement=False)
            logp_i = (x[i, idx_i] - Z[i, 0]).mean()
            logp_samples.append(logp_i)
        logp_samples = torch.stack(logp_samples, dim=0)
        S_T_est = P_T * logp_samples
        H_tail = -S_T_est
    else:
        H_tail = torch.zeros(L, dtype=x.dtype, device=x.device)

    return H_head + H_tail


@torch.no_grad()
def entropy_topm_plus_cv_is(logits: torch.Tensor, m: int = 20, s: int = 5) -> torch.Tensor:
    """
    Exact head for k_head=m-s, tail via control-variated IS residual around uniform.
    logits: [L, V] -> returns [L] (natural-log entropy)
    """
    x = logits.float()
    L, V = x.shape
    k_head = max(0, m - s)

    if k_head > 0:
        topk_logits, topk_idx = torch.topk(x, k=k_head, dim=-1)
    else:
        topk_logits = x.new_zeros((L, 0))
        topk_idx = torch.empty((L, 0), dtype=torch.long, device=x.device)
    Z = torch.logsumexp(x, dim=-1, keepdim=True)

    if k_head > 0:
        p_head = torch.exp(topk_logits - Z)
        H_head = -(p_head * p_head.clamp_min(1e-12).log()).sum(-1)
    else:
        H_head = torch.zeros(L, dtype=x.dtype, device=x.device)

    logp_full = x - Z
    p_full = torch.exp(logp_full)
    if k_head > 0:
        tail_mask = torch.ones_like(p_full, dtype=torch.bool)
        tail_mask.scatter_(1, topk_idx, False)
        p_tail_vec = p_full.masked_fill(~tail_mask, 0.0)
    else:
        p_tail_vec = p_full

    P_T = p_tail_vec.sum(-1)
    no_tail = P_T.le(1e-12)

    Vmk = max(1, V - k_head)
    baseline = -P_T * (P_T.clamp_min(1e-12).log() - torch.log(torch.tensor(Vmk, dtype=x.dtype, device=x.device)))

    if s == 0:
        return H_head + baseline
    q = torch.where(P_T.unsqueeze(-1) > 0, p_tail_vec / P_T.unsqueeze(-1), p_tail_vec)
    est_residual = []
    for i in range(L):
        if no_tail[i] or q[i].sum() <= 0:
            est_residual.append(torch.tensor(0.0, dtype=x.dtype, device=x.device))
            continue
        idx_i = torch.multinomial(q[i], num_samples=s, replacement=False)
        logp_i = logp_full[i, idx_i]
        logu_i = (P_T[i].clamp_min(1e-12).log() - torch.log(torch.tensor(Vmk, dtype=x.dtype, device=x.device)))
        avg_res = (logp_i - logu_i).mean()
        est_residual.append(P_T[i] * avg_res)
    est_residual = torch.stack(est_residual, dim=0)

    H_tail = baseline - est_residual
    return H_head + H_tail
