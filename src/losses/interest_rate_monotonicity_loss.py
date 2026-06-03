"""Interest-rate monotonicity loss.

Created: 2026-06-03
Purpose: Encourage reserves to decrease as discount rates rise.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class InterestRateMonotonicityLoss(BaseLoss):
    """Penalize violations of ``dV/dr <= 0``.

    Scientific Context:
        Higher discount rates typically reduce the present value of future
        liabilities, so reserves should not systematically increase with rates.

    Business Interpretation:
        This loss protects scenario and sensitivity outputs from producing
        counter-intuitive interest-rate behavior.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the interest-rate monotonicity penalty."""

        del model, context
        derivative = self.first_derivative(predictions=predictions, batch=batch, name="interest_rate")
        return self.reduce(torch.relu(derivative))
