"""Model and loss tests."""

from __future__ import annotations

import torch

from insurance_reserve_intelligence_platform.losses.total_loss import TotalLoss
from insurance_reserve_intelligence_platform.models.factory import ModelFactory
from insurance_reserve_intelligence_platform.utils.config import LossConfig, ModelConfig


def test_model_forward_and_total_loss() -> None:
    model = ModelFactory.create_pinn(ModelConfig())
    features = torch.rand(8, 6, requires_grad=True)
    targets = torch.rand(8, 1)
    terms = torch.ones(8, 1) * 10.0
    loss_fn = TotalLoss(LossConfig())
    breakdown = loss_fn(model=model, features=features, targets=targets, terms=terms)
    assert breakdown.total.item() >= 0.0
    assert breakdown.residual.shape == (8, 1)
