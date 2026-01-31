import types

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sekd.config import TrainingConfig
from sekd.distill import OnPolicyDistiller


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1


class DummyLM(nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)
        self.config = types.SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask=None):
        emb = self.embed(input_ids)
        logits = self.proj(emb)
        return types.SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids,
        attention_mask=None,
        max_new_tokens: int = 1,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        **_: dict,
    ):
        batch_size = input_ids.size(0)
        device = input_ids.device
        append = torch.full(
            (batch_size, max_new_tokens),
            eos_token_id,
            dtype=torch.long,
            device=device,
        )
        return torch.cat([input_ids, append], dim=1)


class PromptDataset(Dataset):
    def __init__(self):
        self.samples = [
            {
                "input_ids": torch.tensor([0, 5, 6], dtype=torch.long),
                "attention_mask": torch.tensor([1, 1, 1], dtype=torch.long),
            }
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _build_config(tmpdir: str) -> TrainingConfig:
    return TrainingConfig(
        teacher_model="dummy",
        student_model="dummy",
        distill_category="on_policy",
        distill_type="vanilla",
        datasets=["dummy"],
        output_dir=tmpdir,
        tensorboard_dir=str(tmpdir) + "/tb",
        batch_size=1,
        epochs=1,
        lr=1e-4,
        enable_ce=False,
        alpha_ce=0.0,
        max_seq_len=16,
        k_percent=40,
        kd_objective="reverse",
        on_policy_max_new_tokens=3,
        on_policy_group_size=1,
        on_policy_curriculum=False,
        offline_cache=False,
    )


def test_on_policy_batch_masks_generated_tokens(tmp_path):
    config = _build_config(str(tmp_path))
    student = DummyLM()
    teacher = DummyLM()
    tokenizer = DummyTokenizer()
    dataloader = DataLoader(PromptDataset(), batch_size=1)

    distiller = OnPolicyDistiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tokenizer,
        dataloader=dataloader,
        config=config,
        teacher_device="cpu",
        student_device="cpu",
        logger=None,
    )

    prompt_batch = next(iter(dataloader))
    batch = distiller._build_on_policy_batch(prompt_batch)

    kd_mask = batch["kd_mask"][0]
    # Original prompt has length 3; ensure first three positions masked out
    assert torch.equal(kd_mask[:3], torch.zeros(3, dtype=torch.bool))
    # Newly generated tokens should be marked for KD
    assert kd_mask[3:].all()


def test_on_policy_forward_batch_logs_metrics(tmp_path):
    config = _build_config(str(tmp_path))
    student = DummyLM()
    teacher = DummyLM()
    tokenizer = DummyTokenizer()
    dataloader = DataLoader(PromptDataset(), batch_size=1)

    distiller = OnPolicyDistiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tokenizer,
        dataloader=dataloader,
        config=config,
        teacher_device="cpu",
        student_device="cpu",
        logger=None,
    )

    prompt_batch = next(iter(dataloader))
    batch = distiller._build_on_policy_batch(prompt_batch)
    loss, kl_val, ce_val, metrics = distiller._forward_batch(batch)

    assert torch.isfinite(loss)
    assert torch.isfinite(kl_val)
    assert torch.isfinite(ce_val)
    assert metrics is not None
    assert "train/on_policy_prompt_tokens" in metrics
    assert "train/on_policy_generated_tokens" in metrics


def test_curriculum_updates_k_percent(tmp_path):
    config = _build_config(str(tmp_path))
    config.k_percent = 40
    config.on_policy_curriculum = True
    config.on_policy_curriculum_start_k = 5.0
    config.on_policy_curriculum_steps = 10
    config.on_policy_curriculum_power = 1.0

    student = DummyLM()
    teacher = DummyLM()
    tokenizer = DummyTokenizer()
    dataloader = DataLoader(PromptDataset(), batch_size=1)

    distiller = OnPolicyDistiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=tokenizer,
        dataloader=dataloader,
        config=config,
        teacher_device="cpu",
        student_device="cpu",
        logger=None,
    )

    distiller.global_step = 0
    first_k = distiller._maybe_update_curriculum()
    assert first_k is not None
    assert abs(first_k - 5.0) < 1e-3

    distiller.global_step = 10
    final_k = distiller._maybe_update_curriculum()
    assert final_k is not None
    assert abs(final_k - 40.0) < 1e-3
