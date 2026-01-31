import torch

from sekd.distill.ce_estimators import ce_is_estimator


def _true_ce(s_rows: torch.Tensor, y_rows: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(s_rows, dim=-1)
    return -log_probs[torch.arange(s_rows.size(0)), y_rows]


def _shadow_truth(s_rows: torch.Tensor, indices: torch.Tensor, y_rows: torch.Tensor) -> torch.Tensor:
    sorted_indices, _ = torch.sort(indices)
    s_subset = s_rows[:, sorted_indices]
    y_subset = torch.searchsorted(sorted_indices, y_rows)
    return _true_ce(s_subset, y_subset)


def test_ce_is_equals_truth_when_s_equals_vocab():
    torch.manual_seed(0)
    P, V = 8, 64

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))
    ids_U = torch.arange(V, dtype=torch.long).view(1, V).expand(P, V)
    probs_U = torch.full((P, V), 1.0 / V)
    ids_M = torch.tensor([], dtype=torch.long)

    ce_hat = ce_is_estimator(s_rows, ids_U, probs_U, ids_M, ids_M.numel(), y_rows)
    ce_true = _true_ce(s_rows, y_rows)
    assert torch.allclose(ce_hat, ce_true, atol=1e-6, rtol=1e-6)


def test_ce_is_matches_shadow_truth():
    torch.manual_seed(1)
    V, P, U, M = 12000, 192, 64, 1024

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))

    topk = torch.topk(s_rows, k=U, dim=1)
    ids_U = topk.indices
    probs_U = torch.softmax(topk.values, dim=1)

    ids_M = torch.randperm(V, dtype=torch.long)[:M]

    ce_hat = ce_is_estimator(s_rows, ids_U, probs_U, ids_M, ids_M.numel(), y_rows)

    mask = torch.zeros(V, dtype=torch.bool)
    mask[ids_M] = True
    mask[ids_U.reshape(-1)] = True
    mask[y_rows.unique()] = True
    extra = torch.randperm(V, dtype=torch.long)[:6000]
    mask[extra] = True
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

    ce_shadow = _shadow_truth(s_rows, indices, y_rows)

    diff = ce_hat - ce_shadow
    mae = diff.abs().mean().item()
    corr = torch.corrcoef(torch.stack([ce_hat, ce_shadow]))[0, 1].item()

    assert corr > 0.995
    assert mae < 0.05


def test_ce_is_error_decreases_with_more_negatives():
    torch.manual_seed(2)
    V, P, U = 10000, 160, 64

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))
    ids_U = torch.topk(s_rows, k=U, dim=1).indices
    probs_U = torch.softmax(s_rows.gather(1, ids_U), dim=1)

    ids_M_small = torch.randperm(V, dtype=torch.long)[:512]
    ids_M_big = torch.randperm(V, dtype=torch.long)[:4096]

    mask = torch.zeros(V, dtype=torch.bool)
    mask[ids_U.reshape(-1)] = True
    mask[y_rows.unique()] = True
    mask[torch.randperm(V, dtype=torch.long)[:5000]] = True
    indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
    ce_shadow = _shadow_truth(s_rows, indices, y_rows)

    ce_small = ce_is_estimator(s_rows, ids_U, probs_U, ids_M_small, ids_M_small.numel(), y_rows)
    ce_big = ce_is_estimator(s_rows, ids_U, probs_U, ids_M_big, ids_M_big.numel(), y_rows)

    mae_small = torch.mean(torch.abs(ce_small - ce_shadow)).item()
    mae_big = torch.mean(torch.abs(ce_big - ce_shadow)).item()
    assert mae_big <= mae_small + 1e-6


def test_ce_is_temperature_mismatch_detected():
    torch.manual_seed(3)
    V, P, U, M = 6000, 96, 48, 1024

    s_rows = torch.randn(P, V)
    y_rows = torch.randint(0, V, (P,))
    topk = torch.topk(s_rows, k=U, dim=1)
    ids_U = topk.indices
    probs_U = torch.softmax(topk.values, dim=1)
    ids_M = torch.randperm(V, dtype=torch.long)[:M]

    ce_true = _true_ce(s_rows, y_rows)
    ce_correct = ce_is_estimator(s_rows, ids_U, probs_U, ids_M, ids_M.numel(), y_rows)
    ce_bad = ce_is_estimator(s_rows / 2.0, ids_U, probs_U, ids_M, ids_M.numel(), y_rows)

    corr = torch.corrcoef(torch.stack([ce_correct, ce_true]))[0, 1].item()
    mae = torch.mean(torch.abs(ce_correct - ce_true)).item()

    assert corr > 0.995 or mae < 0.05
    assert not torch.allclose(ce_bad, ce_true, atol=0.05, rtol=0.05)


def test_ce_is_numerics_and_backward():
    torch.manual_seed(4)
    V, P, U, M = 8000, 64, 32, 1024

    s_rows = (50.0 * torch.randn(P, V)).requires_grad_(True)
    y_rows = torch.randint(0, V, (P,))
    ids_U = torch.topk(s_rows.detach(), k=U, dim=1).indices
    probs_U = torch.softmax(s_rows.detach().gather(1, ids_U), dim=1)
    ids_M = torch.randperm(V, dtype=torch.long)[:M]

    ce_hat = ce_is_estimator(s_rows, ids_U, probs_U, ids_M, ids_M.numel(), y_rows)
    loss = ce_hat.mean()
    loss.backward()

    assert torch.isfinite(ce_hat).all()
    assert s_rows.grad is not None and torch.isfinite(s_rows.grad).all()
