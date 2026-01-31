from typing import Tuple, Union

import torch
from torch import Tensor


class LinUCBBandit:
    """LinUCB contextual bandit with a shared linear model."""

    def __init__(
        self,
        feature_dim: int,
        alpha: float = 1.0,
        lambda_: float = 1.0,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        self.feature_dim = int(feature_dim)
        self.alpha = float(alpha)
        self.lambda_ = float(lambda_)
        self.device = torch.device(device)

        self.A = torch.eye(self.feature_dim, device=self.device, dtype=torch.float32) * self.lambda_
        self.b = torch.zeros(self.feature_dim, device=self.device, dtype=torch.float32)
        self._factorize()

    def _factorize(self) -> None:
        # Numerical guard before factorization; keeps A positive-definite.
        jitter = 1e-6 * torch.eye(self.feature_dim, device=self.device, dtype=torch.float32)
        self._L = torch.linalg.cholesky(self.A + jitter)

    @property
    def theta(self) -> Tensor:
        # Solve A θ = b via Cholesky (A = L L^T)
        y = torch.cholesky_solve(self.b.unsqueeze(1), self._L)
        return y.squeeze(1)

    @torch.no_grad()
    def select(
        self,
        contexts: Tensor,
        threshold: float = 0.0,
        max_actions: int | None = None,
    ) -> Tuple[Tensor, Tensor]:
        if contexts.ndim != 2 or contexts.size(1) != self.feature_dim:
            raise ValueError("contexts must have shape [N, feature_dim]")
        if contexts.size(0) == 0:
            return torch.zeros(0, dtype=torch.bool), torch.zeros(0, dtype=torch.float32)

        # Keep track of the original device to map outputs back for downstream code
        orig_device = contexts.device
        ctx = contexts.to(self.device, dtype=torch.float32)
        self._factorize()
        theta = self.theta
        mean = ctx @ theta
        # Solve A x = ctx^T for each row (batched)
        x = torch.cholesky_solve(ctx.t(), self._L)   # [D,N]
        quad = (ctx * x.t()).sum(dim=1).clamp_min(1e-12)
        conf = self.alpha * torch.sqrt(quad)
        scores = mean + conf
        mask = scores > threshold

        if max_actions is not None and max_actions > 0 and mask.sum() > max_actions:
            topk_idx = torch.topk(scores, k=max_actions, largest=True, sorted=False).indices
            keep = torch.zeros_like(mask)
            keep[topk_idx] = True
            mask = keep

        # Return on the same device as the incoming contexts to avoid device mismatch downstream
        return mask.to(orig_device), scores.to(orig_device)

    @torch.no_grad()
    def update(self, contexts: Tensor, rewards: Tensor) -> None:
        if contexts.ndim != 2 or contexts.size(1) != self.feature_dim:
            raise ValueError("contexts must have shape [N, feature_dim]")
        if rewards.ndim != 1 or rewards.size(0) != contexts.size(0):
            raise ValueError("rewards must align with contexts")
        if contexts.size(0) == 0:
            return

        ctx = contexts.to(self.device, dtype=torch.float32)
        rew = rewards.to(self.device, dtype=torch.float32)
        self.A = self.A + ctx.t() @ ctx
        self.b = self.b + ctx.t() @ rew
        self._factorize()