"""Model and loss tests.

Created: 2026-05-31
Purpose: Validate reserve model forward passes and composite loss behavior.
"""

from __future__ import annotations

import torch

from src.losses.total_loss import TotalLoss
from src.models.factory import ModelFactory
from src.utils.config import LossConfig, ModelConfig


def test_model_forward_and_total_loss() -> None:
    """Verify that the reserve model and composite loss produce valid tensor outputs."""
    model = ModelFactory.create_pinn(ModelConfig())
    features = torch.rand(8, 6, requires_grad=True)
    targets = torch.rand(8, 1)
    terms = torch.ones(8, 1) * 10.0
    loss_fn = TotalLoss(LossConfig())
    breakdown = loss_fn(model=model, features=features, targets=targets, terms=terms)
    assert breakdown.total.item() >= 0.0
    assert breakdown.residual.shape == (8, 1)
