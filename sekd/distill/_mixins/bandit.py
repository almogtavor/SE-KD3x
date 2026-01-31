from __future__ import annotations

from typing import Optional

from ...training.linucb_controller import LinUCBBanditController


class BanditMixin:
    bandit_manager: Optional[LinUCBBanditController]

    def _init_bandit_manager(self):
        self.bandit_manager = None
        if self.config.distill_type in {"linucb", "linucb-dkd"}:
            teacher_present = getattr(self, "teacher", None) is not None
            teacher_accessible = bool(getattr(self, "teacher_available", False))
            offline_enabled = bool(getattr(self.config, "offline_cache", False))
            offline_ready = bool(getattr(self.config, "_cache_is_ready", False))

            if not teacher_accessible and not (offline_enabled and offline_ready):
                raise RuntimeError("LinUCB distillation requires either a teacher or a ready offline cache.")

            self.bandit_manager = LinUCBBanditController(
                tokenizer=self.tok,
                config=self.config,
                student_device=self.student_device,
                teacher_device=self.teacher_device,
                sanitize_logits_fn=self._sanitize_logits,
                teacher_available=teacher_accessible,
                reward_with_teacher=teacher_present,
                offline_only=(offline_enabled and offline_ready and not teacher_accessible),
            )
