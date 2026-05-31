"""Base neural model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseReserveModel(nn.Module, ABC):
    """Abstract reserve prediction model."""

    input_features: tuple[str, ...] = ("t", "age", "interest_rate", "premium", "sum_assured", "mortality")

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict reserves for input features."""
