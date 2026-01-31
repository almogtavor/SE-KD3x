from __future__ import annotations

from torch.optim import AdamW
import torch


class OOMSetupHelper:
    @staticmethod
    def configure(self, config):
        # ===== OOM Reduction: 8-bit optimizer to save ~2-3x memory =====
        try:
            from bitsandbytes.optim import Adam8bit
            self.opt = Adam8bit(self.student.parameters(), lr=config.lr)
            if self.ddp_rank == 0:
                print("[OOM-opt] Using 8-bit Adam optimizer to reduce memory.")
        except Exception:
            self.opt = AdamW(self.student.parameters(), lr=config.lr)
            if self.ddp_rank == 0:
                print("[OOM-opt] bitsandbytes not available, using standard AdamW.")
        # ===== OOM Reduction: Enable gradient checkpointing on student =====
        student_base = self._student_base
        if hasattr(student_base, "gradient_checkpointing_enable"):
            try:
                student_base.gradient_checkpointing_enable()
                if self.ddp_rank == 0:
                    print("[OOM-opt] Enabled gradient checkpointing on student model.")
            except Exception as e:
                if self.ddp_rank == 0:
                    print(f"[OOM-opt] Could not enable gradient checkpointing: {e}")
        
        # ===== OOM Reduction: Enable memory-efficient attention =====
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_flash_sdp(True)
            if self.ddp_rank == 0:
                print("[OOM-opt] Enabled memory-efficient SDPA backends.")
        except Exception:
            pass
        
        # Try Flash Attention 2 on student if supported
        if hasattr(student_base, "config") and hasattr(student_base.config, "use_flash_attention_2"):
            try:
                student_base.config.use_flash_attention_2 = True
                if self.ddp_rank == 0:
                    print("[OOM-opt] Enabled Flash Attention 2 on student model.")
            except Exception:
                pass
        
        # ===== OOM Reduction: Mixed precision (AMP) setup =====
        # Initialize GradScaler for fp16 AMP (disabled for bfloat16 which doesn't need scaling).
        # PyTorch's GradScaler currently requires FP32 grads during unscale, so disable it when
        # the student parameters stay in lower precision (e.g. fp16) to avoid runtime errors.
        self._amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self._use_amp = True
        if self._amp_dtype == torch.float16:
            student_base = self._student_base
            trainable_params = [p for p in student_base.parameters() if p.requires_grad]
            has_low_precision_grads = any(p.dtype != torch.float32 for p in trainable_params)
            if not has_low_precision_grads:
                from torch.cuda.amp import GradScaler
                self._scaler = GradScaler(enabled=True)
                if self.ddp_rank == 0:
                    print(f"[OOM-opt] Using AMP with {self._amp_dtype} and GradScaler.")
            else:
                self._scaler = None
                if self.ddp_rank == 0:
                    dtypes = sorted({str(p.dtype) for p in trainable_params})
                    dtype_info = ", ".join(dtypes) if dtypes else "unknown dtypes"
                    print(
                        "[OOM-opt] Using AMP with torch.float16; GradScaler disabled because "
                        f"student grads remain in {dtype_info}, which PyTorch cannot unscale safely."
                    )
        else:
            self._scaler = None
            if self.ddp_rank == 0:
                print(f"[OOM-opt] Using AMP with {self._amp_dtype} (no scaler needed).")
