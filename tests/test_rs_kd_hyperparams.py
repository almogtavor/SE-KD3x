import types

from sekd.distill.forward_batch_runner import ForwardBatchRunner as _ForwardBatchRunner


class DummyDistiller:
    def __init__(self, k_percent: float):
        self.config = types.SimpleNamespace(k_percent=k_percent, enable_ce_on_all_tokens=False)


def test_bucket_selection_fraction_ignores_k_percent():
    runner = _ForwardBatchRunner(DummyDistiller(k_percent=5), batch={})
    pct = runner._rs_selection_fraction((0.8, 0.95))
    assert abs(pct - 0.15) < 1e-6


def test_bucket_sample_quota_matches_bucket_span():
    runner = _ForwardBatchRunner(DummyDistiller(k_percent=80), batch={})
    pct = runner._rs_selection_fraction((0.8, 0.95))
    quota = runner._rs_target_sample_count(total_valid_tokens=100, candidate_count=15, pct=pct, shared_quota=None)
    assert quota == 15


def test_non_bucket_selection_uses_k_percent():
    runner = _ForwardBatchRunner(DummyDistiller(k_percent=25), batch={})
    pct = runner._rs_selection_fraction(None)
    assert abs(pct - 0.25) < 1e-6


def test_shared_quota_caps_bucket_sample_count():
    runner = _ForwardBatchRunner(DummyDistiller(k_percent=40), batch={})
    pct = runner._rs_selection_fraction((0.8, 0.95))
    quota = runner._rs_target_sample_count(total_valid_tokens=100, candidate_count=30, pct=pct, shared_quota=10)
    assert quota == 10
