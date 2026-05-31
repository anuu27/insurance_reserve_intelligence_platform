"""Supervised reserve fitting loss.

Created: 2026-05-31
Purpose: Provide the supervised data loss between PINN and classical reserves.
"""

from __future__ import annotations

import torch
from torch import nn


class DataLoss(nn.Module):
    """Mean squared error against classical reserve targets."""

    def __init__(self) -> None:
        """Initialize the data loss module."""
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the supervised reserve-fitting loss.

        Args:
            predictions: Model reserve predictions.
            targets: Classical reserve targets.

        Returns:
            torch.Tensor: Scalar mean squared error.
        """
        return self.criterion(predictions, targets)
