import math
import string
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from .linucb import LinUCBBandit


@dataclass
class BanditTokenRecord:
    """Snapshot of a single token decision made by the bandit."""
    example_idx: int
    position: int
    context: torch.Tensor
    kl_before: float


@dataclass
class BanditPendingBatch:
    """Bundle of tokens tied to a batch so we can recompute rewards later."""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    tokens: List[BanditTokenRecord]
    temperature: float


class LinUCBBanditController:
    """Encapsulates LinUCB token selection and reward updates."""

    def __init__(
        self,
        tokenizer,
        config,
        student_device: torch.device,
        teacher_device: torch.device,
        sanitize_logits_fn,
        teacher_available: bool,
        reward_with_teacher: bool,
        offline_only: bool,
    ) -> None:
        self.tok = tokenizer
        self.config = config
        self.student_device = torch.device(student_device)
        self.teacher_device = torch.device(teacher_device)
        self._sanitize_logits = sanitize_logits_fn

        self.teacher_available = bool(teacher_available)
        self._reward_enabled = bool(reward_with_teacher)
        self.offline_only = bool(offline_only)

        if self.teacher_available and not self.offline_only:
            self._feature_order: Tuple[str, ...] = (
                "entropy",
                "student_entropy",
                "teacher_ce",
                "student_ce",
                "kl",
                "pos_code",
                "pos_norm",
            )
        else:
            # Offline-only mode: restrict to features backed by cache/student signals.
            self._feature_order = (
                "entropy",
                "student_entropy",
                "student_ce",
                "kl",
                "pos_norm",
            )

        feature_dim = len(self._feature_order)
        self.bandit = LinUCBBandit(
            feature_dim=feature_dim,
            alpha=self.config.bandit_alpha,
            lambda_=self.config.bandit_lambda,
            device=self.config.bandit_device,
        )

        self._pos_cache: Dict[int, float] = {}
        self._pending: List[BanditPendingBatch] = []

    # ------------------------------------------------------------------
    # Selection helpers
    # ------------------------------------------------------------------
    def _token_pos_code(self, token_id: int) -> float:
        """Map a token ID to a coarse part-of-speech style bucket [0, 1]."""
        cached = self._pos_cache.get(int(token_id))
        if cached is not None:
            return cached

        token = self.tok.convert_ids_to_tokens(int(token_id))
        token = token.replace("Ġ", "").replace("▁", "").strip()

        if not token:
            category = 3
        elif any(ch.isalpha() for ch in token):
            category = 0
        elif token.isdigit():
            category = 1
        elif all(ch in string.punctuation for ch in token):
            category = 2
        else:
            category = 3

        code = category / 3.0
        self._pos_cache[int(token_id)] = code
        return code

    def _prepare_features(
        self,
        ent_raw: torch.Tensor,
        student_entropy: Optional[torch.Tensor],
        teacher_ce: torch.Tensor,
        student_ce: torch.Tensor,
        kl_pos: torch.Tensor,
        input_ids: torch.Tensor,
        valid_next: torch.Tensor,
    ) -> List[Optional[Dict[str, torch.Tensor]]]:
        """Assemble per-token contexts and bookkeeping needed by the bandit."""
        features: List[Optional[Dict[str, torch.Tensor]]] = []
        seq_len = valid_next.size(1)

        for i in range(valid_next.size(0)):
            mask = valid_next[i]
            valid_idx = torch.where(mask)[0]
            if valid_idx.numel() == 0:
                features.append(None)
                continue

            # Ensure indices live on the same device as the tensors we slice from.
            def _gather_row(source: torch.Tensor) -> torch.Tensor:
                row = source[i]
                idx = valid_idx
                if idx.device != row.device:
                    idx = idx.to(row.device)
                return row[idx]

            ent_vals = _gather_row(ent_raw).float()
            value_device = ent_vals.device
            valid_idx_values = valid_idx
            if valid_idx_values.device != value_device:
                valid_idx_values = valid_idx_values.to(value_device)

            components: Dict[str, torch.Tensor] = {
                "entropy": ent_vals,
            }

            if "student_entropy" in self._feature_order:
                if student_entropy is None:
                    components["student_entropy"] = torch.zeros_like(ent_vals)
                else:
                    components["student_entropy"] = _gather_row(student_entropy).float().to(value_device)

            if "teacher_ce" in self._feature_order:
                if teacher_ce is None:
                    components["teacher_ce"] = torch.zeros_like(ent_vals)
                else:
                    components["teacher_ce"] = _gather_row(teacher_ce).float().to(value_device)

            if "student_ce" in self._feature_order:
                if student_ce is None:
                    components["student_ce"] = torch.zeros_like(ent_vals)
                else:
                    components["student_ce"] = _gather_row(student_ce).float().to(value_device)

            if "kl" in self._feature_order:
                if kl_pos is None:
                    components["kl"] = torch.zeros_like(ent_vals)
                else:
                    components["kl"] = _gather_row(kl_pos).float().to(value_device)

            if "pos_code" in self._feature_order:
                valid_idx_cpu = valid_idx.to(input_ids.device)
                target_ids = input_ids[i, 1:][valid_idx_cpu].tolist()
                pos_codes = torch.tensor(
                    [self._token_pos_code(tok_id) for tok_id in target_ids],
                    device=value_device,
                    dtype=torch.float32,
                )
                components["pos_code"] = pos_codes

            if "pos_norm" in self._feature_order:
                denom = max(1, seq_len - 1)
                pos_norm = (valid_idx_values.float() / denom).float()
                components["pos_norm"] = pos_norm

            contexts = torch.stack(
                [components[name] for name in self._feature_order],
                dim=1,
            )

            # Track entropy top quartile to diagnose whether we revisit high-uncertainty tokens.
            top25_count = max(1, min(valid_idx.numel(), math.ceil(0.25 * valid_idx.numel())))
            top25_rel = torch.topk(ent_vals, k=top25_count, largest=True, sorted=False).indices
            top25_abs = valid_idx_values[top25_rel]

            features.append(
                {
                    "contexts": contexts.detach(),
                    "valid_indices": valid_idx,
                    "top25_abs": top25_abs,
                }
            )

        return features

    def select_tokens(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ent_raw: torch.Tensor,
        student_entropy: Optional[torch.Tensor],
        teacher_ce: torch.Tensor,
        student_ce: torch.Tensor,
        kl_pos: torch.Tensor,
        valid_next: torch.Tensor,
        temperature: float,
    ) -> Tuple[List[torch.Tensor], Dict[str, float], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Choose token positions for KD, record selections, and queue reward computations."""
        feature_data = self._prepare_features(
            ent_raw=ent_raw,
            student_entropy=student_entropy,
            teacher_ce=teacher_ce,
            student_ce=student_ce,
            kl_pos=kl_pos,
            input_ids=input_ids,
            valid_next=valid_next,
        )

        kd_terms: List[torch.Tensor] = []
        pending_tokens: List[BanditTokenRecord] = []
        selected_batches: List[int] = []
        selected_positions: List[int] = []
        total_examples = 0
        total_selected = 0.0
        total_selected_fraction = 0.0
        total_overlap_selected = 0.0
        total_overlap_top25 = 0.0

        max_actions = self.config.bandit_max_tokens if self.config.bandit_max_tokens is not None else None

        for idx, data in enumerate(feature_data):
            if not data:
                continue
            contexts = data["contexts"]
            if contexts.size(0) == 0:
                continue

            # Keep contexts on their current device; the bandit will internally
            # move them to its own device and return outputs on the same device
            # as the provided contexts (see LinUCBBandit.select).
            contexts_clean = contexts.detach()
            mask, scores = self.bandit.select(
                contexts_clean,
                threshold=self.config.bandit_threshold,
                max_actions=max_actions,
            )

            n_valid = contexts_clean.size(0)
            eff_min = max(1, min(self.config.bandit_min_tokens, n_valid))
            if max_actions is not None:
                eff_min = min(eff_min, max_actions)

            # Force a minimum number of actions to keep gradients flowing on small batches.
            if mask.sum().item() < eff_min:
                topk = torch.topk(scores, k=eff_min, largest=True, sorted=False).indices
                new_mask = torch.zeros_like(mask, dtype=torch.bool)
                new_mask[topk] = True
                mask = new_mask

            # If the threshold filtered out everything, backstop with the best-scoring token.
            if mask.sum().item() == 0:
                top_idx = torch.topk(scores, k=1, largest=True).indices
                mask = torch.zeros_like(mask, dtype=torch.bool)
                mask[top_idx] = True

            selected_rel = torch.where(mask)[0]
            valid_idx = data["valid_indices"]
            selected_abs = valid_idx[selected_rel.to(valid_idx.device)]
            selected_abs_device = selected_abs.to(kl_pos.device)

            if selected_abs_device.numel() == 0:
                continue

            selected_kls = kl_pos[idx, selected_abs_device]
            if selected_kls.ndim == 0:
                selected_kls = selected_kls.unsqueeze(0)
            kd_terms.append(selected_kls.reshape(-1))

            selected_abs_cpu = selected_abs_device.detach().to("cpu")
            if selected_abs_cpu.numel() > 0:
                selected_batches.extend([idx] * selected_abs_cpu.numel())
                selected_positions.extend(selected_abs_cpu.tolist())

            for rel_idx, abs_pos in zip(selected_rel.tolist(), selected_abs_device.tolist()):
                pending_tokens.append(
                    BanditTokenRecord(
                        example_idx=idx,
                        position=abs_pos,
                        # Store a CPU copy for the later bandit update to avoid holding GPU tensors
                        context=contexts_clean[rel_idx].detach().to("cpu").float(),
                        kl_before=float(kl_pos[idx, abs_pos].detach().item()),
                    )
                )

            total_examples += 1
            selected_count = float(selected_abs_device.numel())
            total_selected += selected_count
            total_selected_fraction += selected_count / n_valid

            top25_abs = data["top25_abs"].to(selected_abs_device.device)
            intersection = torch.isin(selected_abs_device, top25_abs).float().sum().item()
            total_overlap_selected += intersection / max(1.0, selected_count)
            total_overlap_top25 += intersection / max(1, float(top25_abs.numel()))

        if pending_tokens and self._reward_enabled:
            # Store a CPU copy of the batch so we can run reward rollouts after the optimizer step.
            self._pending.append(
                BanditPendingBatch(
                    input_ids=input_ids.detach().cpu(),
                    attention_mask=attention_mask.detach().cpu(),
                    tokens=pending_tokens,
                    temperature=temperature,
                )
            )

        metrics: Dict[str, float] = {}
        if total_examples > 0:
            metrics = {
                "bandit/selected_tokens": total_selected / total_examples,
                "bandit/selected_fraction": total_selected_fraction / total_examples,
                "bandit/overlap_selected": total_overlap_selected / total_examples,
                "bandit/overlap_top25": total_overlap_top25 / total_examples,
            }

        selection_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        if selected_positions:
            selection_data = (
                torch.tensor(selected_batches, dtype=torch.long),
                torch.tensor(selected_positions, dtype=torch.long),
            )

        return kd_terms, metrics, selection_data

    # ------------------------------------------------------------------
    # Reward processing
    # ------------------------------------------------------------------
    @staticmethod
    def _kl_loss(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        return F.kl_div(log_q, log_p, log_target=True, reduction="none").sum(-1)

    def process_rewards(self, student_model, teacher_model) -> Optional[Dict[str, float]]:
        """Re-evaluate queued tokens and feed rewards back into the bandit."""
        if not self._reward_enabled or teacher_model is None:
            self._pending.clear()
            return None

        if not self._pending:
            return None

        contexts: List[torch.Tensor] = []
        rewards: List[float] = []

        with torch.no_grad():
            was_training = student_model.training
            if was_training:
                student_model.eval()

            for record in self._pending:
                if not record.tokens:
                    continue

                input_ids_cpu = record.input_ids
                attn_cpu = record.attention_mask
                input_ids_s = input_ids_cpu.to(self.student_device)
                attn_mask_s = attn_cpu.to(self.student_device)

                s_logits = student_model(input_ids_s, attention_mask=attn_mask_s).logits
                s_logits = self._sanitize_logits(s_logits, "student_eval")
                s_pred = s_logits[:, :-1, :]
                s_log_probs = torch.log_softmax(s_pred / record.temperature, dim=-1)
                valid_next = attn_mask_s[:, 1:].bool()

                input_ids_t = input_ids_cpu.to(self.teacher_device)
                attn_mask_t = attn_cpu.to(self.teacher_device)
                t_logits = teacher_model(input_ids_t, attention_mask=attn_mask_t, output_hidden_states=False).logits
                t_logits = self._sanitize_logits(t_logits, "teacher_eval")
                t_pred = t_logits[:, :-1, :]
                t_log_probs = torch.log_softmax(t_pred / record.temperature, dim=-1)

                kl_pos_after = self._kl_loss(t_log_probs.to(self.student_device), s_log_probs)

                for token in record.tokens:
                    if token.example_idx >= kl_pos_after.size(0):
                        continue
                    if token.position >= kl_pos_after.size(1):
                        continue
                    if not valid_next[token.example_idx, token.position]:
                        continue
                    kl_after = kl_pos_after[token.example_idx, token.position].item()
                    reward = token.kl_before - kl_after
                    clip_val = float(getattr(self.config, "bandit_reward_clip", 0.0))
                    if clip_val > 0.0:
                        reward = max(-clip_val, min(clip_val, reward))
                    contexts.append(token.context.clone())
                    rewards.append(reward)

            if was_training:
                student_model.train()

        self._pending.clear()

        if not rewards:
            return None

        contexts_tensor = torch.stack(contexts)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        self.bandit.update(contexts_tensor, rewards_tensor)

        avg_reward = float(rewards_tensor.mean().item())
        positive_rate = float((rewards_tensor > 0).float().mean().item())
        return {
            "bandit/avg_reward": avg_reward,
            "bandit/positive_reward_rate": positive_rate,
        }

    def reset(self) -> None:
        """Clear cached state so training can restart safely."""
        self._pending.clear()
        self._pos_cache.clear()
