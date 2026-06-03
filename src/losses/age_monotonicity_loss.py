"""Age monotonicity loss.

Created: 2026-06-03
Purpose: Encourage reserves to rise with attained age when the product structure implies increasing risk.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class AgeMonotonicityLoss(BaseLoss):
    """Penalize violations of ``dV/dx >= 0``.

    Scientific Context:
        For many term-life situations, older attained ages imply higher claim
        risk and therefore higher reserve needs, all else equal.

    Business Interpretation:
        This loss is useful when the portfolio design and pricing basis make age
        monotonicity a reasonable business expectation.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the age monotonicity penalty."""

        del model, context
        derivative = self.first_derivative(predictions=predictions, batch=batch, name="age")
        return self.reduce(torch.relu(-derivative))
