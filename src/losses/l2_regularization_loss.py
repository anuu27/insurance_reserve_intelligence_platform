"""L2 parameter regularization loss.

Created: 2026-06-03
Purpose: Penalize large network weights to reduce overfitting and numerical instability.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class L2RegularizationLoss(BaseLoss):
    """Compute an L2 penalty over all model parameters.

    Scientific Context:
        This is the standard weight-decay style penalty ``sum(||θ||²)``.

    Business Interpretation:
        It is a generic modeling hygiene term that reduces the chance of a
        brittle reserve surface with large and unstable parameter magnitudes.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the L2 regularization penalty."""

        del batch, context
        penalties = [parameter.pow(2).sum() for parameter in model.parameters()]
        if not penalties:
            return predictions.new_tensor(0.0)
        return torch.stack(penalties).sum()
