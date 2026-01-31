from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

from transformers.generation.logits_process import LogitsProcessor, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import torch

from .trainer import Distiller


@dataclass
class _OnPolicyRolloutLoader:
    """Lightweight wrapper that yields on-policy rollouts from an underlying prompt loader."""

    base_loader: Iterable
    builder: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]

    def __iter__(self):
        for prompt_batch in self.base_loader:
            yield self.builder(prompt_batch)

    def __len__(self):  # type: ignore[override]
        return len(self.base_loader)  # type: ignore[arg-type]

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - passthrough
        return getattr(self.base_loader, item)


class _BlockEosUntilSteps(LogitsProcessor):
    """Prevent EOS from being sampled until a minimum number of rollout tokens is generated."""

    def __init__(self, eos_token_id: Optional[int], context_lengths: List[int], min_steps: List[int]):
        self.eos_token_id = eos_token_id
        self.context_lengths = context_lengths
        self.min_steps = min_steps

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.eos_token_id is None:
            return scores
        vocab_size = scores.size(-1)
        if self.eos_token_id < 0 or self.eos_token_id >= vocab_size:
            return scores
        for idx, min_step in enumerate(self.min_steps):
            if min_step <= 0:
                continue
            generated = input_ids[idx].size(-1) - self.context_lengths[idx]
            if generated < min_step:
                scores[idx, self.eos_token_id] = float("-inf")
        return scores


class _FixedStepsStoppingCriteria(StoppingCriteria):
    """Stop generation once each sequence has produced its target number of rollout tokens."""

    def __init__(self, context_lengths: List[int], target_steps: List[int]):
        self.context_lengths = context_lengths
        self.target_steps = target_steps

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: Optional[torch.FloatTensor],
        **kwargs: Any,
    ) -> bool:
        for idx, target in enumerate(self.target_steps):
            if target <= 0:
                continue
            generated = input_ids[idx].size(-1) - self.context_lengths[idx]
            if generated < target:
                return False
        return True


class OnPolicyDistiller(Distiller):
    """Distiller that generates on-policy rollouts from the student before computing KD."""

    def __init__(
        self,
        teacher_model,
        student_model,
        tokenizer,
        dataloader,
        config,
        teacher_device="cuda",
        student_device="cuda",
        logger=None,
    ):
        super().__init__(
            teacher_model=teacher_model,
            student_model=student_model,
            tokenizer=tokenizer,
            dataloader=dataloader,
            config=config,
            teacher_device=teacher_device,
            student_device=student_device,
            logger=logger,
        )

        self._prompt_loader = dataloader
        self._rollout_meta: Optional[Dict[str, float]] = None

        self._reverse_weight = float(getattr(self.config, "on_policy_reverse_kl_weight", 1.0))
        self._forward_weight = float(getattr(self.config, "on_policy_forward_kl_weight", 0.0))
        if self._forward_weight > 0.0:
            # Forward KL weighting is not yet supported in this implementation.
            raise NotImplementedError("on_policy_forward_kl_weight > 0 is not supported yet.")

        self._group_size = max(1, int(getattr(self.config, "on_policy_group_size", 1)))
        self._max_new_tokens_cfg = max(1, int(getattr(self.config, "on_policy_max_new_tokens", 256)))
        self._rollout_temperature = float(getattr(self.config, "on_policy_temperature", 0.7))
        self._rollout_top_p = float(getattr(self.config, "on_policy_top_p", 0.9))
        self._rollout_do_sample = bool(getattr(self.config, "on_policy_do_sample", True))

        self._curriculum_active = bool(getattr(self.config, "on_policy_curriculum", False))
        self._curriculum_steps = max(1, int(getattr(self.config, "on_policy_curriculum_steps", 1000)))
        self._curriculum_start_k = max(0.0, float(getattr(self.config, "on_policy_curriculum_start_k", 5.0)))
        self._curriculum_power = max(0.0, float(getattr(self.config, "on_policy_curriculum_power", 1.0)))
        self._orig_k_percent = float(getattr(self.config, "k_percent", 100.0))
        self._cuts_enabled = bool(getattr(self.config, "enable_cuts_in_the_middle_for_on_policy", True))
        datasets = getattr(self.config, "datasets", [])
        first_dataset = str(datasets[0]).lower() if datasets else ""
        self._cuts_enabled = self._cuts_enabled and first_dataset == "fineweb"
        self._cut_min_tokens = max(1, int(getattr(self.config, "on_policy_cut_min_tokens", 12)))
        self._cut_max_tokens = max(self._cut_min_tokens, int(getattr(self.config, "on_policy_cut_max_tokens", 32)))
        self._cut_min_context = max(1, int(getattr(self.config, "on_policy_cut_min_context", 128)))
        base_seed = int(getattr(self.config, "seed", 1337))
        rank_offset = int(getattr(self.config, "ddp_rank", 0))
        self._cut_rng = random.Random(base_seed + rank_offset)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _maybe_update_curriculum(self) -> Optional[float]:
        if not self._curriculum_active:
            return None

        progress = min(1.0, self.global_step / max(1, self._curriculum_steps))
        if self._curriculum_power > 0.0:
            progress = progress ** self._curriculum_power

        start = min(self._curriculum_start_k, self._orig_k_percent)
        target = self._orig_k_percent
        new_k = start + (target - start) * progress
        new_k = max(0.0, min(100.0, new_k))
        try:
            self.config.k_percent = float(new_k)
        except Exception:
            object.__setattr__(self.config, "k_percent", float(new_k))
        return new_k

    def _prepare_rollout_contexts(
        self,
        prompt_input: torch.Tensor,
        prompt_attn: torch.Tensor,
        prompt_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
        batch_size = prompt_input.size(0)
        if not self._cuts_enabled or batch_size == 0:
            zero_steps = [0] * batch_size
            return prompt_input, prompt_attn, prompt_lengths, zero_steps, zero_steps

        pad_token_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        device = prompt_input.device
        dtype = prompt_input.dtype
        attn_dtype = prompt_attn.dtype

        context_sequences: List[torch.Tensor] = []
        context_lengths: List[int] = []
        target_steps: List[int] = []
        min_steps: List[int] = []
        applied_cut = False
        max_context_len = 0

        for idx in range(batch_size):
            length = int(prompt_lengths[idx].item())
            ids = prompt_input[idx, :length].clone()

            max_target = min(
                self._cut_max_tokens,
                max(0, length - self._cut_min_context - 1),
                self._max_new_tokens_cfg,
            )
            can_cut = max_target >= self._cut_min_tokens
            if not can_cut:
                context_sequences.append(ids)
                context_lengths.append(length)
                target_steps.append(0)
                min_steps.append(0)
                max_context_len = max(max_context_len, length)
                continue

            target = self._cut_rng.randint(self._cut_min_tokens, max_target)
            cut_high = length - target - 1
            if cut_high < self._cut_min_context:
                context_sequences.append(ids)
                context_lengths.append(length)
                target_steps.append(0)
                min_steps.append(0)
                max_context_len = max(max_context_len, length)
                continue

            cut_point = self._cut_rng.randint(self._cut_min_context, cut_high)
            context = ids[:cut_point].clone()
            context_sequences.append(context)
            context_len = context.size(0)
            context_lengths.append(context_len)
            target_steps.append(target)
            min_steps.append(min(self._cut_min_tokens, target))
            max_context_len = max(max_context_len, context_len)
            applied_cut = True

        if not applied_cut:
            return prompt_input, prompt_attn, prompt_lengths, target_steps, min_steps

        if pad_token_id is None:
            pad_token_id = 0

        new_input = torch.full(
            (batch_size, max_context_len),
            pad_token_id,
            dtype=dtype,
            device=device,
        )
        new_attn = torch.zeros((batch_size, max_context_len), dtype=attn_dtype, device=device)
        for idx, ctx in enumerate(context_sequences):
            ctx = ctx.to(device)
            ctx_len = ctx.size(0)
            if ctx_len == 0:
                continue
            new_input[idx, :ctx_len] = ctx
            new_attn[idx, :ctx_len] = 1

        new_lengths = new_attn.sum(dim=1).to(torch.long)
        return new_input, new_attn, new_lengths, target_steps, min_steps

    def _build_on_policy_batch(self, prompt_batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        prompt_input = prompt_batch["input_ids"].to(self.student_device)
        prompt_attn = prompt_batch["attention_mask"].to(self.student_device)
        prompt_lengths = prompt_attn.sum(dim=1).to(torch.long)

        if self._group_size > 1:
            prompt_input = prompt_input.repeat_interleave(self._group_size, dim=0)
            prompt_attn = prompt_attn.repeat_interleave(self._group_size, dim=0)
            prompt_lengths = prompt_lengths.repeat_interleave(self._group_size, dim=0)

        prompt_input, prompt_attn, prompt_lengths, target_steps, min_steps = self._prepare_rollout_contexts(
            prompt_input,
            prompt_attn,
            prompt_lengths,
        )
        lengths_list = prompt_lengths.tolist()

        max_prompt = int(prompt_lengths.max().item()) if prompt_lengths.numel() else prompt_input.size(1)
        available = max(1, self.config.max_seq_len - max_prompt)
        forced_generation = any(step > 0 for step in target_steps)
        if forced_generation:
            target_steps = [min(step, available) for step in target_steps]
            min_steps = [min(min_steps[idx], target_steps[idx]) for idx in range(len(min_steps))]
            forced_max = max(target_steps) if target_steps else 1
            max_new_tokens = max(1, forced_max)
        else:
            max_new_tokens = min(self._max_new_tokens_cfg, available)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": self._rollout_do_sample,
            "temperature": self._rollout_temperature,
            "top_p": self._rollout_top_p,
            "eos_token_id": self.tok.eos_token_id,
            "pad_token_id": self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id,
        }
        if forced_generation:
            generation_kwargs["logits_processor"] = LogitsProcessorList(
                [
                    _BlockEosUntilSteps(
                        generation_kwargs["eos_token_id"],
                        lengths_list,
                        min_steps,
                    )
                ]
            )
            generation_kwargs["stopping_criteria"] = StoppingCriteriaList(
                [
                    _FixedStepsStoppingCriteria(
                        lengths_list,
                        target_steps,
                    )
                ]
            )

        was_training = self.student.training
        self.student.eval()
        with torch.no_grad():
            sequences = self.student.generate(
                input_ids=prompt_input,
                attention_mask=prompt_attn,
                **generation_kwargs,
            )
        if was_training:
            self.student.train()

        if sequences.size(1) > self.config.max_seq_len:
            sequences = sequences[:, : self.config.max_seq_len]

        attention_mask = (sequences != generation_kwargs["pad_token_id"]).to(torch.long)
        kd_mask = attention_mask.bool()

        repeat_prompt_lengths = prompt_lengths.to(torch.long)
        pad_id = generation_kwargs["pad_token_id"]
        for idx, plen in enumerate(repeat_prompt_lengths.tolist()):
            actual_len = int(attention_mask[idx].sum().item())
            cutoff = actual_len
            if idx < len(target_steps) and target_steps[idx] > 0:
                cutoff = min(actual_len, plen + target_steps[idx])
            if cutoff < actual_len:
                sequences[idx, cutoff:] = pad_id
                attention_mask[idx, cutoff:] = 0
            if plen > 0:
                kd_mask[idx, :plen] = False
            if idx < len(target_steps) and target_steps[idx] > 0:
                kd_mask[idx, cutoff:] = False
        # Ensure prompts do not include padding tokens as valid positions
        kd_mask &= attention_mask.bool()

        new_k = self._maybe_update_curriculum()

        prompt_float = prompt_lengths.float()
        rollout_lengths = attention_mask.sum(dim=1).float()
        kd_token_counts = kd_mask[:, 1:].sum(dim=1).float()

        meta = {
            "prompt_tokens_mean": float(prompt_float.mean().item()) if prompt_float.numel() else 0.0,
            "generated_tokens_mean": float((rollout_lengths - prompt_float).mean().item()) if rollout_lengths.numel() else 0.0,
            "kd_tokens_total": float(kd_mask[:, 1:].sum().item()),
            "kd_tokens_mean": float(kd_token_counts.mean().item()) if kd_token_counts.numel() else 0.0,
            "batch_size": float(sequences.size(0)),
        }
        if new_k is not None:
            meta["curriculum_k_percent"] = float(new_k)

        batch = {
            "input_ids": sequences.to("cpu"),
            "attention_mask": attention_mask.to("cpu"),
            "kd_mask": kd_mask.to("cpu"),
            "_on_policy_meta": meta,
        }
        return batch

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------
    def _forward_batch(self, batch: Dict[str, torch.Tensor]):
        meta = batch.pop("_on_policy_meta", None)
        loss, kl_val, ce_val, bandit_metrics = super()._forward_batch(batch)

        if meta is not None:
            metrics = dict(bandit_metrics or {})
            metrics.setdefault("train/on_policy_prompt_tokens", meta.get("prompt_tokens_mean", 0.0))
            metrics.setdefault("train/on_policy_generated_tokens", meta.get("generated_tokens_mean", 0.0))
            metrics.setdefault("train/on_policy_kd_tokens", meta.get("kd_tokens_mean", 0.0))
            if "curriculum_k_percent" in meta:
                metrics.setdefault("train/on_policy_curriculum_k", meta["curriculum_k_percent"])
            metrics.setdefault("train/on_policy_kd_tokens_total", meta.get("kd_tokens_total", 0.0))
            bandit_metrics = metrics

        return loss, kl_val, ce_val, bandit_metrics

    def train(self, epochs: int = 1, log_every: int = 100):
        rollout_loader = _OnPolicyRolloutLoader(self.dataloader, self._build_on_policy_batch)
        original_loader = self.dataloader
        self.dataloader = rollout_loader
        try:
            super().train(epochs=epochs, log_every=log_every)
        finally:
            self.dataloader = original_loader
