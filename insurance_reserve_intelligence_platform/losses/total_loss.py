"""Composite training loss."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from insurance_reserve_intelligence_platform.losses.boundary_loss import BoundaryLoss
from insurance_reserve_intelligence_platform.losses.data_loss import DataLoss
from insurance_reserve_intelligence_platform.losses.pde_loss import PDELoss
from insurance_reserve_intelligence_platform.losses.regularization_loss import RegularizationLoss
from insurance_reserve_intelligence_platform.utils.config import LossConfig


@dataclass(slots=True)
class LossBreakdown:
    """Named loss components."""

    total: torch.Tensor
    data: torch.Tensor
    pde: torch.Tensor
    boundary: torch.Tensor
    regularization: torch.Tensor
    residual: torch.Tensor


class TotalLoss(nn.Module):
    """Composite objective for PINN training."""

    def __init__(self, config: LossConfig) -> None:
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
