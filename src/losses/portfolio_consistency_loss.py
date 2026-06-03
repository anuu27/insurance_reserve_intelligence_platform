"""Portfolio consistency loss.

Created: 2026-06-03
Purpose: Encourage batch-level reserve totals to align with an aggregate portfolio reserve target.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.losses.base_loss import BaseLoss


class PortfolioConsistencyLoss(BaseLoss):
    """Penalize mismatch between aggregate and summed individual reserves.

    Scientific Context:
        The current dataset is pointwise rather than explicitly hierarchical, so
        the aggregate proxy defaults to the sum of benchmark target reserves in
        the batch unless a portfolio target is supplied in ``context``.

    Business Interpretation:
        This loss helps preserve additivity, which matters when policy-level
        models are rolled up into portfolio, capital, or planning views.
    """

    def forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        predictions: torch.Tensor,
        context: dict[str, Any],
    ) -> torch.Tensor:
        """Compute the aggregate consistency penalty."""

        del model
        predicted_portfolio_reserve = predictions.sum(dim=0, keepdim=True)
        portfolio_target = context.get("portfolio_reserve_target")
        if portfolio_target is None:
            targets = self.require_batch_tensor(batch, "target")
            portfolio_target = targets.sum(dim=0, keepdim=True)
        penalty = (predicted_portfolio_reserve - portfolio_target).pow(2)
        return self.reduce(penalty)
