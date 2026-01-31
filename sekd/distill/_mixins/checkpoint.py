from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, Tuple

import torch


class CheckpointMixin:
    def save_checkpoint(self, epoch: int, step: int, *, force: bool = False) -> Optional[Path]:
        """Save a training checkpoint."""
        if self.config.checkpoint_steps <= 0 and not force:
            return None
        checkpoint_name = f"checkpoint_epoch{epoch}_step{step}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.opt.state_dict(),
            'distill_type': self.config.distill_type,
            'k_percent': self.config.k_percent,
            # Record base/student/teacher identifiers to allow later export without extra CLI
            'base_model_dir': getattr(self.config, 'student_model', None),
            'student_model': getattr(self.config, 'student_model', None),
            'teacher_model': getattr(self.config, 'teacher_model', None),
        }
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        self._write_last_checkpoint(checkpoint_path)
        self._cleanup_old_checkpoints()
        return checkpoint_path

    def _write_last_checkpoint(self, checkpoint_path: Path) -> None:
        """Persist the most recent checkpoint path for resume helpers."""
        try:
            marker = self.checkpoint_dir / "last_checkpoint.txt"
            marker.write_text(str(checkpoint_path.resolve()), encoding="utf-8")
        except Exception as exc:
            print(f"[checkpoint] Warning: failed to write last_checkpoint.txt: {exc}")

    def _cleanup_old_checkpoints(self):
        """Keep only the most recent checkpoints."""
        if self.config.keep_checkpoints <= 0:
            return
        checkpoint_pattern = str(self.checkpoint_dir / "checkpoint_epoch*_step*.pt")
        checkpoint_files = glob.glob(checkpoint_pattern)
        checkpoint_files.sort(key=os.path.getmtime, reverse=True)  # Sort by modification time

        # Remove old checkpoints beyond keep_checkpoints limit
        for old_checkpoint in checkpoint_files[self.config.keep_checkpoints:]:
            os.remove(old_checkpoint)
            print(f"Removed old checkpoint: {old_checkpoint}")

    def load_checkpoint(self, checkpoint_path: str) -> Tuple[int, int]:
        """Load a training checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.student_device)

        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.opt.load_state_dict(checkpoint['optimizer_state_dict'])
        self.global_step = checkpoint['global_step']

        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"Resuming from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
        return checkpoint['epoch'], checkpoint['step']
