"""Regularization loss.

Created: 2026-05-31
Purpose: Compute L2 regularization for reserve model parameters.
"""

from __future__ import annotations

import torch
from torch import nn


class RegularizationLoss(nn.Module):
    """Compute L2 weight regularization."""

    def __init__(self) -> None:
        """Initialize the regularization loss module."""
        super().__init__()

    def forward(self, model: nn.Module) -> torch.Tensor:
        """Compute an L2 penalty over model parameters.

        Args:
            model: Model whose parameters will be regularized.

        Returns:
            torch.Tensor: Scalar L2 penalty.
        """
        parameters = list(model.parameters())
        penalties = [parameter.pow(2.0).sum() for parameter in parameters]
        if not penalties:
            return torch.tensor(0.0)
        return torch.stack(penalties).sum()
