"""Boundary-condition loss for term insurance.

Created: 2026-06-03
Purpose: Enforce the terminal reserve condition that term-life liabilities vanish at maturity.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.losses.base_loss import BaseLoss


class BoundaryLoss(BaseLoss):
    """Enforce the term-insurance boundary condition ``V(T)=0``.

    Scientific Context:
        Once coverage expires, a term-life contract no longer has future death
        benefits to fund, so the reserve should collapse to zero at maturity.

    Business Interpretation:
        This prevents the model from carrying phantom liabilities after contract
        expiry, which would distort capital and profitability views.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the terminal boundary penalty.

        Args:
            model: Reserve model used to score the boundary inputs.
            batch: Batch dictionary containing normalized features and ``term``.
            predictions: In-batch reserve predictions. Unused here.
            context: Shared execution context. Boundary predictions are written
                here for diagnostics.

        Returns:
            torch.Tensor: Reduced ``L_boundary`` scalar.
        """

        del predictions
        features = self.require_batch_tensor(batch, "features")
        terms = self.require_batch_tensor(batch, "term")
        boundary_features = features.clone()
        boundary_features[:, FEATURE_INDEX["time"] : FEATURE_INDEX["time"] + 1] = terms / FEATURE_SCALES["time"]
        boundary_predictions = model(boundary_features)
        context["boundary_predictions"] = boundary_predictions
        return self.reduce(boundary_predictions.pow(2))
