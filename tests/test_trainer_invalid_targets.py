from types import SimpleNamespace

import torch

from sekd.config import TrainingConfig
from sekd.distill.trainer import Distiller


class DummyTeacher(torch.nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(use_cache=False)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, self.vocab_size, dtype=torch.float32)
        return SimpleNamespace(logits=logits)

    def eval(self):
        super().eval()
        return self


class DummyStudent(torch.nn.Module):
    def __init__(self, vocab_size: int, num_embeddings: int = 64, hidden_dim: int = 16):
        super().__init__()
        self.embedding = torch.nn.Embedding(num_embeddings, hidden_dim)
        self.proj = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        self.config = SimpleNamespace(use_cache=False, use_flash_attention_2=False, vocab_size=vocab_size)

    def forward(self, input_ids, attention_mask=None):
        hidden = self.embedding(input_ids)
        logits = self.proj(hidden)
        return SimpleNamespace(logits=logits)

    def train(self, mode: bool = True):
        super().train(mode)
        return self

    def gradient_checkpointing_enable(self):
        return None


def test_forward_batch_masks_out_of_range_targets(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "enable_mem_efficient_sdp", lambda *_, **__: None, raising=False)
    monkeypatch.setattr(torch.backends.cuda, "enable_flash_sdp", lambda *_, **__: None, raising=False)

    vocab_size = 32
    teacher = DummyTeacher(vocab_size).eval()
    student = DummyStudent(vocab_size)

    config = TrainingConfig(
        teacher_model="dummy-teacher",
        student_model="dummy-student",
        datasets=["fineweb"],
        output_dir=str(tmp_path / "out"),
        tensorboard_dir=str(tmp_path / "tb"),
        offline_cache=False,
        enable_ce=True,
        alpha_ce=0.5,
        batch_size=1,
        max_seq_len=4,
    )

    distiller = Distiller(
        teacher_model=teacher,
        student_model=student,
        tokenizer=None,
        dataloader=[],
        config=config,
        teacher_device=torch.device("cpu"),
        student_device=torch.device("cpu"),
        logger=None,
    )
    distiller._use_amp = False

    input_ids = torch.tensor([[1, 7, vocab_size + 3, 2]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    total, kd_loss, ce_loss, _ = distiller._forward_batch({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })

    captured = capsys.readouterr()

    assert torch.isfinite(total)
    assert torch.isfinite(torch.tensor(kd_loss))
    assert torch.isfinite(torch.tensor(ce_loss))
    assert distiller._warned_invalid_targets
    assert "CE targets out of range" in captured.out
