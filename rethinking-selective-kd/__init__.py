"""SE-KD3X: Rethinking Selective Knowledge Distillation for LLMs.

This package implements student-entropy-guided selective knowledge distillation
with support for position, class, and sample-level selection.
"""

from .config import TrainingConfig, TrainingMetrics
from .distill import Distiller, OnPolicyDistiller

__version__ = "0.1.0"
__all__ = [
    "TrainingConfig",
    "TrainingMetrics",
    "Distiller",
    "OnPolicyDistiller",
]
