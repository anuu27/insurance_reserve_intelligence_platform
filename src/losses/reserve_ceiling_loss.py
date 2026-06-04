"""Reserve ceiling loss.

Created: 2026-06-03
Purpose: Penalize reserves that exceed the contractual sum assured.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class ReserveCeilingLoss(BaseLoss):
    """Penalize violations of the heuristic bound ``V <= S``.

    Scientific Context:
        For term insurance, the reserve should typically stay below the maximum
        death benefit amount because the liability cannot exceed the contractual
        payout under the modeled state space.

    Business Interpretation:
        This acts as a pragmatic business bound that catches runaway reserve
        forecasts before they contaminate optimization and stress workflows.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the reserve ceiling penalty."""

        del model, context
        sum_assured = self.raw_feature(batch, "sum_assured")
        return self.reduce(torch.relu(predictions - sum_assured))
