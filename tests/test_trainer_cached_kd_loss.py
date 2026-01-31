import torch
from types import SimpleNamespace

from sekd.distill.forward_batch_runner import ForwardBatchRunner as _ForwardBatchRunner


def _setup_runner():
    runner = _ForwardBatchRunner.__new__(_ForwardBatchRunner)  # bypass __init__
    runner.distiller = SimpleNamespace(student_dtype=torch.float32)
    runner.valid_next = torch.tensor([[True, True]])
    runner.s_log_probs = torch.tensor(
        [[[-1.0, -2.0], [-1.5, -2.5]]],
        dtype=torch.float32,
    )
    runner.ce_loss_override = None
    runner.kd_loss = None
    runner.T = 1.0
    return runner


def _common_args():
    batch_idx = torch.tensor([0, 0], dtype=torch.long)
    pos_idx = torch.tensor([0, 1], dtype=torch.long)
    ids_U = torch.tensor([[0], [1]], dtype=torch.long)
    probs_U = torch.ones_like(ids_U, dtype=torch.float32)
    return batch_idx, pos_idx, ids_U, probs_U


def test_cached_teacher_logits_compute_mean_kl_loss():
    runner = _setup_runner()
    args = _common_args()

    handled = runner._handle_cached_distill_teacher_logits(True, *args)
    assert handled is True
    assert torch.isclose(runner.kd_loss, torch.tensor(1.75, dtype=torch.float32))


def test_cached_kd_loss_matches_non_cached_path():
    runner = _setup_runner()
    args = _common_args()

    runner._handle_cached_distill_teacher_logits(False, *args)
    assert torch.isclose(runner.kd_loss, torch.tensor(1.75, dtype=torch.float32))
