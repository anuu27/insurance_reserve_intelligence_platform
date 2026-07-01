"""Physics-informed PDE residual loss.

Created: 2026-06-03  Revised: 2026-06-11
Purpose: Penalize violations of the Thiele reserve differential equation.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class PDEResidualLoss(BaseLoss):
    """Enforce the term-life Thiele ODE in standardised z-space.

    The model now predicts  z = (V - μ_V) / σ_V  where μ_V and σ_V are
    the mean and std of raw £ reserves from the training set.

    So  V = z * σ_V + μ_V  and  dV/dt = dz/dt * σ_V.

    The Thiele ODE  dV/dt = r*V + P - μ_mort*(S - V)  in z-space:

        σ_V * dz/dt = r*(z*σ_V + μ_V) + P - μ_mort*(S - z*σ_V - μ_V)

    Dividing by σ_V:

        dz/dt = r*(z + μ_V/σ_V) + P/σ_V - μ_mort*(S/σ_V - z - μ_V/σ_V)

    Residual: f = dz/dt - r*(z + μ/σ) - P/σ + μ_mort*(S/σ - z - μ/σ)
    All terms are O(1) → stable loss from epoch 0.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        del model
        dz_dt   = self.first_derivative(predictions=predictions, batch=batch, name="time")
        r       = self.raw_feature(batch, "interest_rate")
        P       = self.raw_feature(batch, "premium")
        S       = self.raw_feature(batch, "sum_assured").clamp(min=1.0)
        mu_mort = self.raw_feature(batch, "mortality")

        t_mean = batch["target_mean"].to(predictions.device)   # μ_V
        t_std  = batch["target_std"].to(predictions.device)    # σ_V

        # z + μ_V/σ_V = (V - μ_V)/σ_V + μ_V/σ_V = V/σ_V
        z_plus_offset = predictions + t_mean / t_std

        residual = (
            dz_dt
            - r * z_plus_offset
            - P / t_std
            + mu_mort * (S / t_std - z_plus_offset)
        )
        context["pde_residual"] = residual
        return self.reduce(residual.pow(2))