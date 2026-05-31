"""PDE residual loss for PINN training."""

from __future__ import annotations

import torch
from torch import nn


class PDELoss(nn.Module):
    """Residual minimization for the term-life Thiele equation."""

    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, features: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        grads = torch.autograd.grad(
            outputs=predictions,
            inputs=features,
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True,
        )[0]
        dv_dt = grads[:, 0:1]
        interest_rate = features[:, 2:3]
        premium = features[:, 3:4]
        sum_assured = features[:, 4:5]
        mortality = features[:, 5:6]
        residual = dv_dt - interest_rate * predictions - premium + mortality * (sum_assured - predictions)
        loss = self.criterion(residual, torch.zeros_like(residual))
        return loss, residual
