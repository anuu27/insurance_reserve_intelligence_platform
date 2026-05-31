"""Generic multilayer perceptron blocks."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn


def activation_factory(name: str) -> Callable[[], nn.Module]:
    """Map configuration strings to activation layers."""

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
        """Run a forward pass."""

        return self.network(features)
