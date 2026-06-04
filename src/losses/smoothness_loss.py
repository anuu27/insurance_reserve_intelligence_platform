"""Smoothness loss for reserve trajectories.

Created: 2026-06-03
Purpose: Penalize excessive time-axis curvature in the learned reserve surface.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class SmoothnessLoss(BaseLoss):
    """Penalize violent reserve oscillations over time.

    Scientific Context:
        The loss applies a curvature penalty ``mean((d²V/dt²)^2)`` so the model
        does not satisfy data and PDE constraints by creating unstable local
        wiggles.

    Business Interpretation:
        Smoother reserve curves are easier to explain, stress, and govern than
        noisy curves that imply unstable month-to-month reserve motion.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the smoothness penalty."""

        del model, context
        second_derivative = self.second_derivative(predictions=predictions, batch=batch, name="time")
        return self.reduce(second_derivative.pow(2))
