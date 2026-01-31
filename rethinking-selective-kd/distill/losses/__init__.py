"""Loss functions for knowledge distillation."""

from .udkd import UDKDLoss, compute_udkd_loss

__all__ = ["UDKDLoss", "compute_udkd_loss"]
