"""Physics-informed PDE residual loss.

Created: 2026-06-03
Purpose: Penalize violations of the Thiele reserve differential equation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class PDEResidualLoss(BaseLoss):
    """Enforce the term-life reserve ODE inside the network.

    Scientific Context:
        The loss evaluates the Thiele residual
        ``f = dV/dt - rV - P + μ(S - V)`` and pushes it toward zero.

    Business Interpretation:
        This term keeps the model from behaving like a black-box curve fitter by
        requiring the reserves to evolve according to liability mechanics.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the PDE residual penalty.

        Args:
            model: Reserve model. Unused because predictions are precomputed.
            batch: Batch dictionary containing reserve inputs.
            predictions: Model reserve predictions.
            context: Shared execution context. Residuals are written here for
                downstream logging and diagnostics.

        Returns:
            torch.Tensor: Reduced PDE residual loss.
        """

        del model
        dv_dt = self.first_derivative(predictions=predictions, batch=batch, name="time")
        interest_rate = self.raw_feature(batch, "interest_rate")
        premium = self.raw_feature(batch, "premium")
        sum_assured = self.raw_feature(batch, "sum_assured")
        mortality = self.raw_feature(batch, "mortality")
        residual = dv_dt - interest_rate * predictions - premium + mortality * (sum_assured - predictions)
        context["pde_residual"] = residual
        return self.reduce(residual.pow(2))
