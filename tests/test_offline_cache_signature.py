from types import SimpleNamespace

from sekd.training.offline_cache import _cache_signature_from_config


def _base_config(fineweb_tokens, *, seed=1337):
    return SimpleNamespace(
        teacher_model="teacher",
        datasets=["fineweb"],
        dataset_config="edu",
        prompt_col="prompt",
        answer_col="answer",
        fineweb_tokens=fineweb_tokens,
        seed=seed,
        max_seq_len=512,
        entropy_approx_m=12,
        kd_temperature=2.0,
        rs_vocab_samples=32,
        rs_samples=64,
        H_hat_u8=True,
        enable_packing=True,
        offline_cache_mode="entropy",
    )


def test_cache_signature_varies_with_fineweb_token_budget():
    cfg_small = _base_config(4_000_000)
    cfg_big = _base_config(100_000_000)

    sig_small = _cache_signature_from_config(cfg_small, tokenizer_name="tok", dataset_len=100, teacher_name="teacher")
    sig_big = _cache_signature_from_config(cfg_big, tokenizer_name="tok", dataset_len=100, teacher_name="teacher")

    assert sig_small["fineweb_tokens"] == 4_000_000
    assert sig_big["fineweb_tokens"] == 100_000_000
    assert sig_small != sig_big


def test_cache_signature_captures_dataset_metadata():
    cfg = _base_config(5_000_000)
    sig = _cache_signature_from_config(cfg, tokenizer_name="tok", dataset_len=256, teacher_name="teacher")

    assert sig["datasets"] == ["fineweb"]
    assert sig["dataset_config"] == "edu"
    assert sig["prompt_col"] == "prompt"
    assert sig["answer_col"] == "answer"


def test_cache_signature_varies_with_seed():
    cfg_seed_a = _base_config(5_000_000, seed=1337)
    cfg_seed_b = _base_config(5_000_000, seed=1339)

    sig_a = _cache_signature_from_config(cfg_seed_a, tokenizer_name="tok", dataset_len=256, teacher_name="teacher")
    sig_b = _cache_signature_from_config(cfg_seed_b, tokenizer_name="tok", dataset_len=256, teacher_name="teacher")

    assert sig_a["seed"] == 1337
    assert sig_b["seed"] == 1339
    assert sig_a != sig_b
