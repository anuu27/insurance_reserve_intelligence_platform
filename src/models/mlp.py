"""Generic multilayer perceptron blocks.

Created: 2026-05-31
Purpose: Provide configurable feed-forward neural network building blocks.
"""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def activation_factory(name: str) -> Callable[[], nn.Module]:
    """Map configuration strings to activation layers.

    Args:
        name: Activation function name from configuration.

    Returns:
        Callable[[], nn.Module]: Activation layer constructor.

    Raises:
        ValueError: If the activation name is unsupported.
    """

    registry: dict[str, Callable[[], nn.Module]] = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    key = name.lower()
    if key not in registry:
        raise ValueError(f"Unsupported activation: {name}")
    return registry[key]


class MLP(nn.Module):
    """Configurable feed-forward network."""

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, activation: str, dropout: float) -> None:
        """Initialize the MLP backbone.

        Args:
            input_dim: Input feature dimension.
            hidden_dim: Width of hidden layers.
            num_layers: Total number of linear layers including the output layer.
            activation: Activation function name.
            dropout: Dropout probability between hidden layers.

        Raises:
            ValueError: If fewer than two layers are requested.
        """
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        act = activation_factory(activation)
        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(act())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            features: Input feature tensor.

        Returns:
            torch.Tensor: Network output tensor.
        """

        return self.network(features)
