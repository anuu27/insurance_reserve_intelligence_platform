"""Factory for reserve models."""

from __future__ import annotations

from insurance_reserve_intelligence_platform.models.base_model import BaseReserveModel
from insurance_reserve_intelligence_platform.models.pinn import PINNReserveModel
from insurance_reserve_intelligence_platform.utils.config import ModelConfig


class ModelFactory:
    """Create models from configuration."""

    @staticmethod
    def create_pinn(config: ModelConfig) -> BaseReserveModel:
        """Instantiate the default PINN model."""

        return PINNReserveModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            dropout=config.dropout,
        )
