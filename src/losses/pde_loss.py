"""PDE residual loss for PINN training.

Created: 2026-05-31
Purpose: Penalize violations of the Thiele reserve equation during training.
"""

from __future__ import annotations

import torch
from torch import nn


class PDELoss(nn.Module):
    """Residual minimization for the term-life Thiele equation.

    Scientific Context:
        This module enforces the physics-informed part of the PINN objective by
        penalizing deviations from the reserve differential equation.

    Business Interpretation:
        It discourages the model from producing reserve predictions that may fit
        sampled data but violate actuarial liability mechanics.
    """

    def __init__(self) -> None:
        """Initialize the PDE residual loss module."""
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, features: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the PDE residual loss and residual values.

        Args:
            features: Input features with gradients enabled.
            predictions: Model reserve predictions.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Scalar PDE loss and pointwise residuals.

        Scientific Context:
            Automatic differentiation is used to compute ``dV/dt`` directly from
            the network. The residual
            ``f = dV/dt - rV - P + μ(S - V)`` is then forced toward zero.

        Business Interpretation:
            This is the consistency check that asks whether the network's reserve
            forecast behaves like a real insurance reserve process.
        """
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
