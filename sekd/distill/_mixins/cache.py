from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...training.offline_cache import TeacherOfflineCache


class CacheMixin:
    def _lookup_cache_batch(self, input_ids) -> Optional[List[Dict[str, Any]]]:
        if not self.cache:
            return None
        items = []
        for i in range(input_ids.size(0)):
            key = TeacherOfflineCache.key_from_ids(input_ids[i])
            if not self.cache.has(key):
                return None
            items.append(self.cache.read_item(key))
        return items

    def _cache_mode(self) -> Optional[str]:
        if not self.cache:
            return None
        cached = getattr(self, "_cached_cache_mode", None)
        if cached:
            return cached

        sig = getattr(self.cache, "manifest", {}).get("signature", {})
        mode = sig.get("cache_mode")
        if mode is not None:
            cached_mode = str(mode)
            setattr(self, "_cached_cache_mode", cached_mode)
            return cached_mode

        # Legacy caches did not persist cache_mode in the manifest; detect from stored items.
        items = getattr(self.cache, "manifest", {}).get("items", {}) or {}
        for key in items:
            try:
                item = self.cache.read_item(key)
            except Exception:
                continue
            raw_mode = item.get("cache_mode")
            if raw_mode is not None:
                cached_mode = str(raw_mode)
                setattr(self, "_cached_cache_mode", cached_mode)
                return cached_mode
            if "entropy_fp16" in item:
                setattr(self, "_cached_cache_mode", "entropy")
                return "entropy"
            if "target_prob_fp16" in item:
                setattr(self, "_cached_cache_mode", "unc")
                return "unc"
            if "H_hat_u8" in item or "H_hat" in item:
                setattr(self, "_cached_cache_mode", "entropy_approx")
                return "entropy_approx"
        setattr(self, "_cached_cache_mode", "entropy_approx")
        return "entropy_approx"

    def compute_cache_signature(self) -> Dict[str, Any]:
        """Compute a stable signature for the logits cache based on teacher/tokenizer/settings/dataset."""
        override = getattr(self.config, "_expected_cache_signature", None)
        if override:
            return dict(override)
        teacher_name = getattr(getattr(self.teacher, "config", None), "_name_or_path", None)
        if teacher_name is None:
            teacher_name = getattr(self.config, "teacher_model", "unknown")
        dataset_len = int(len(self.dataloader.dataset)) if hasattr(self.dataloader, "dataset") else -1
        selection_count = getattr(self.config, "offline_cache_selection_count", None)
        if selection_count is None:
            selection_count = getattr(self.config, "_offline_cache_selection_count", None)
        if selection_count is not None and bool(getattr(self.config, "offline_cache_selected_only", False)):
            try:
                dataset_len = int(selection_count)
            except (TypeError, ValueError):
                pass
        return {
            "teacher_name": teacher_name,
            "tokenizer_name": getattr(self.tok, "name_or_path", "unknown"),
            "max_seq_len": int(self.config.max_seq_len),
            "entropy_approx_m": int(getattr(self.config, "entropy_approx_m", 12)),
            "rs_vocab_samples": int(getattr(self.config, "rs_vocab_samples", 64)),
            "rs_vocab_beta": float(getattr(self.config, "rs_vocab_beta", 1.0)),
            "entropy_approx_temperature": float(
                getattr(self.config, "entropy_approx_temperature", getattr(self.config, "cache_temperature", 1.0))
            ),
            "cache_mode": getattr(self.config, "offline_cache_mode", "entropy"),
            "dataset_len": dataset_len,
            "seed": int(getattr(self.config, "seed", 0) or 0),
        } | self._selection_signature()

    def _selection_signature(self) -> Dict[str, Any]:
        signature: Dict[str, Any] = {}
        selection_hash = getattr(self.config, "offline_cache_selection_hash", None)
        if selection_hash is None:
            selection_hash = getattr(self.config, "_offline_cache_selection_hash", None)
        if selection_hash:
            signature["sample_selection_hash"] = str(selection_hash)
        selection_count = getattr(self.config, "offline_cache_selection_count", None)
        if selection_count is None:
            selection_count = getattr(self.config, "_offline_cache_selection_count", None)
        if selection_count is not None:
            try:
                signature["sample_selection_count"] = int(selection_count)
            except (TypeError, ValueError):
                pass
        return signature
