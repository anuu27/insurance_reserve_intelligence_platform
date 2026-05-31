"""Factory for reserve models.

Created: 2026-05-31
Purpose: Construct reserve models from typed configuration objects.
"""

from __future__ import annotations

from src.models.base_model import BaseReserveModel
from src.models.pinn import PINNReserveModel
from src.utils.config import ModelConfig


class ModelFactory:
    """Create models from configuration."""

    @staticmethod
    def create_pinn(config: ModelConfig) -> BaseReserveModel:
        """Instantiate the default PINN model.

        Args:
            config: Model configuration describing the network architecture.

        Returns:
            BaseReserveModel: Configured PINN reserve model.
        """

        return PINNReserveModel(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            activation=config.activation,
            dropout=config.dropout,
            skip_connections=config.skip_connections,
        )
