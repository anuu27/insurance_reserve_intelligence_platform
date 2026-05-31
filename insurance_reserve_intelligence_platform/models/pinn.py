"""PINN reserve model."""

from __future__ import annotations

import torch

from insurance_reserve_intelligence_platform.models.base_model import BaseReserveModel
from insurance_reserve_intelligence_platform.models.mlp import MLP


class PINNReserveModel(BaseReserveModel):
    """Physics-informed reserve estimator implemented with an MLP backbone."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, activation: str, dropout: float) -> None:
        super().__init__()
        self.backbone = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation=activation,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict reserve values."""

        return self.backbone(features)
