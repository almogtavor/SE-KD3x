from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch


class TokenEntropyLogger:
    """Helper that writes RS-KD entropy sampling distributions for later graphing."""

    def __init__(
        self,
        *,
        enabled: bool,
        dump_path: Path,
        limit: int,
        tokenizer=None,
        ddp_rank: int = 0,
    ) -> None:
        self.enabled = bool(enabled) and limit > 0
        self.dump_path = dump_path
        self.limit = max(0, limit)
        self.tokenizer = tokenizer
        self.ddp_rank = ddp_rank
        self.logged_docs = 0
        if self.can_write:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            self.dump_path.write_text("", encoding="utf-8")

    @property
    def can_write(self) -> bool:
        return self.enabled and self.ddp_rank == 0

    def record(
        self,
        *,
        batch_index: int,
        valid_indices: torch.Tensor,
        entropies: torch.Tensor,
        probabilities: torch.Tensor,
        source: str,
        bucket_filtered: bool,
        bucket_bounds: Optional[Tuple[float, float]],
        selection_fraction: float,
        alpha: float,
        q_floor: float,
        quota: int,
        input_ids: torch.Tensor,
        valid_next: torch.Tensor,
        global_step: int,
        k_percent: float,
    ) -> None:
        if not self.can_write:
            return
        if self.logged_docs >= self.limit:
            self.enabled = False
            return
        if valid_indices.numel() == 0:
            return
        token_positions = valid_indices.detach().to("cpu", non_blocking=True).long()
        ent_vals = entropies.detach().to("cpu", non_blocking=True).float().tolist()
        prob_vals = probabilities.detach().to("cpu", non_blocking=True).float().tolist()
        if not ent_vals or not prob_vals:
            return
        batch_ids = input_ids[batch_index]
        if batch_ids.device.type != "cpu":
            batch_ids = batch_ids.to("cpu", non_blocking=True)
        next_positions = (token_positions + 1).tolist()
        token_ids_tensor = batch_ids[next_positions]
        token_ids = (
            token_ids_tensor.detach().cpu().tolist()
            if isinstance(token_ids_tensor, torch.Tensor)
            else list(token_ids_tensor)
        )
        token_strings = self._convert_tokens(token_ids)
        record = {
            "doc_id": int(self.logged_docs),
            "global_step": int(global_step),
            "batch_index": int(batch_index),
            "token_indices": [int(x) for x in token_positions.tolist()],
            "token_ids": token_ids,
            "token_strings": token_strings,
            "entropies": ent_vals,
            "probabilities": prob_vals,
            "candidate_source": source,
            "bucket_filtered": bool(bucket_filtered),
            "bucket_bounds": list(bucket_bounds) if bucket_bounds else None,
            "selection_fraction": float(selection_fraction),
            "selection_quota": int(quota),
            "candidate_count": len(ent_vals),
            "rs_alpha": float(alpha),
            "prob_floor": float(q_floor),
            "k_percent": float(k_percent),
        }
        doc_valid_mask = valid_next[batch_index]
        record["doc_valid_token_count"] = int(doc_valid_mask.sum().item())
        with self.dump_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")
        self.logged_docs += 1
        if self.logged_docs >= self.limit:
            self.enabled = False

    def record_selection(
        self,
        *,
        keep_mask: torch.Tensor,
        valid_next: torch.Tensor,
        input_ids: torch.Tensor,
        entropies: Optional[torch.Tensor],
        global_step: int,
        k_percent: float,
        distill_type: str,
        selection_policy: Optional[str] = None,
        save_full_document: bool = True,
    ) -> None:
        """Simplified logging for any distill type - logs selected vs valid tokens.
        
        Args:
            save_full_document: If True, also saves all token IDs and all entropies
                               for the entire document (not just selected tokens).
        """
        if not self.can_write:
            return
        if self.logged_docs >= self.limit:
            self.enabled = False
            return
        # Log one document per batch (first with selections)
        for i in range(keep_mask.size(0)):
            if self.logged_docs >= self.limit:
                break
            sel_idx = torch.where(keep_mask[i])[0]
            if sel_idx.numel() == 0:
                continue
            valid_idx = torch.where(valid_next[i])[0]
            sel_pos = sel_idx.cpu().tolist()
            next_pos = [p + 1 for p in sel_pos]
            batch_ids = input_ids[i].cpu()
            token_ids = batch_ids[next_pos].tolist()
            token_strings = self._convert_tokens(token_ids)
            ent_vals = None
            if entropies is not None:
                ent_vals = entropies[i, sel_idx].cpu().float().tolist()
            record = {
                "doc_id": int(self.logged_docs),
                "global_step": int(global_step),
                "batch_index": int(i),
                "distill_type": distill_type,
                "selection_policy": selection_policy,
                "selected_indices": sel_pos,
                "token_ids": token_ids,
                "token_strings": token_strings,
                "entropies": ent_vals,
                "selected_count": len(sel_pos),
                "valid_count": int(valid_idx.numel()),
                "k_percent": float(k_percent),
            }
            
            # Optionally save full document data for comprehensive visualization
            if save_full_document:
                # All input token IDs (L tokens) - the full sequence
                all_token_ids = batch_ids.tolist()
                record["all_token_ids"] = all_token_ids
                record["all_token_strings"] = self._convert_tokens(all_token_ids)
                
                # All valid positions (positions that have a next token to predict)
                record["all_valid_indices"] = valid_idx.cpu().tolist()
                
                # Full entropy array for all positions (L-1 values, or just valid positions)
                if entropies is not None:
                    # Save entropy for all valid positions
                    valid_ent = entropies[i, valid_idx].cpu().float().tolist()
                    record["all_valid_entropies"] = valid_ent
            
            with self.dump_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record))
                handle.write("\n")
            self.logged_docs += 1

    def _convert_tokens(self, token_ids: Sequence[int]) -> List[str]:
        tokenizer = self.tokenizer
        tokens: Optional[Sequence[str]] = None
        if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
            try:
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                if isinstance(tokens, str):
                    tokens = [tokens]
            except Exception:
                tokens = None
        if tokens is None:
            tokens = [str(tid) for tid in token_ids]
        cleaned: List[str] = []
        for tok in tokens:
            text = str(tok)
            text = text.replace("\n", "\\n").replace("\t", "\\t")
            cleaned.append(text)
        return cleaned
