"""Boundary-condition loss.

Created: 2026-05-31
Purpose: Enforce the terminal reserve boundary condition for term-life contracts.
"""

from __future__ import annotations

import torch
from torch import nn


class BoundaryLoss(nn.Module):
    """Enforce V(T)=0 for term-life contracts.

    Scientific Context:
        Term-life contracts have no remaining liability after maturity, so the
        terminal reserve is constrained to zero.

    Business Interpretation:
        This makes the model respect the business fact that the contract should
        not carry reserve after coverage ends.
    """

    def __init__(self) -> None:
        """Initialize the boundary loss module."""
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, features: torch.Tensor, terms: torch.Tensor, model: nn.Module) -> torch.Tensor:
        """Compute the terminal boundary loss.

        Args:
            features: Input feature tensor.
            terms: Policy maturity values.
            model: Model used to predict boundary reserves.

        Returns:
            torch.Tensor: Scalar boundary-condition loss.

        Business Interpretation:
            This protects against unrealistic end-of-term reserve values that
            would otherwise distort forecasting and stress results.
        """
        boundary_features = features.clone()
        boundary_features[:, 0:1] = terms
        boundary_predictions = model(boundary_features)
        return self.criterion(boundary_predictions, torch.zeros_like(boundary_predictions))
