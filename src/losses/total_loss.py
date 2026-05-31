"""Composite training loss.

Created: 2026-05-31
Purpose: Assemble the weighted PINN objective from all constituent loss terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.losses.boundary_loss import BoundaryLoss
from src.losses.data_loss import DataLoss
from src.losses.pde_loss import PDELoss
from src.losses.regularization_loss import RegularizationLoss
from src.utils.config import LossConfig


@dataclass(slots=True)
class LossBreakdown:
    """Named loss components.

    Attributes:
        total: Weighted total loss.
        data: Supervised reserve-fitting loss.
        pde: PDE residual loss.
        boundary: Boundary-condition loss.
        regularization: L2 regularization penalty.
        residual: Pointwise PDE residual values.

    Business Interpretation:
        This breakdown lets researchers and model-risk reviewers see whether model
        improvement is coming from better data fit, stronger physics compliance,
        or better boundary adherence.
    """

    total: torch.Tensor
    data: torch.Tensor
    pde: torch.Tensor
    boundary: torch.Tensor
    regularization: torch.Tensor
    residual: torch.Tensor


class TotalLoss(nn.Module):
    """Composite objective for PINN training.

    Scientific Context:
        The total objective combines supervised learning, PDE residual matching,
        terminal boundary enforcement, and optional parameter regularization.

    Business Interpretation:
        This is the training contract that balances empirical accuracy with
        actuarial plausibility, which is important for reserve governance.
    """

    def __init__(self, config: LossConfig) -> None:
        """Initialize the composite loss module.

        Args:
            config: Loss-weight configuration.
        """
        super().__init__()
        self.config = config
        self.data_loss = DataLoss()
        self.pde_loss = PDELoss()
        self.boundary_loss = BoundaryLoss()
        self.regularization_loss = RegularizationLoss()

    def forward(
        self,
        model: nn.Module,
        features: torch.Tensor,
        targets: torch.Tensor,
        terms: torch.Tensor,
    ) -> LossBreakdown:
        """Compute the full PINN loss breakdown.

        Args:
            model: Reserve prediction model.
            features: Input features with gradients enabled.
            targets: Supervised reserve targets.
            terms: Policy maturity values for the boundary condition.

        Returns:
            LossBreakdown: Structured loss components and residuals.

        Business Interpretation:
            The output explains not just whether the model is wrong, but how it is
            wrong from an actuarial and governance perspective.
        """
        predictions = model(features)
        data_component = self.data_loss(predictions, targets)
        pde_component, residual = self.pde_loss(features, predictions)
        boundary_component = self.boundary_loss(features, terms, model)
        reg_component = self.regularization_loss(model)
        total = (
            self.config.lambda_data * data_component
            + self.config.lambda_pde * pde_component
            + self.config.lambda_boundary * boundary_component
            + self.config.lambda_reg * reg_component
        )
        return LossBreakdown(
            total=total,
            data=data_component,
            pde=pde_component,
            boundary=boundary_component,
            regularization=reg_component,
            residual=residual,
        )
