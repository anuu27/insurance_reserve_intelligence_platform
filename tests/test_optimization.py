"""Optimization tests."""

from __future__ import annotations

from insurance_reserve_intelligence_platform.data.simulator import PolicySimulator
from insurance_reserve_intelligence_platform.optimization.optimizer_engine import OptimizationEngine
from insurance_reserve_intelligence_platform.pipeline import build_model
from insurance_reserve_intelligence_platform.utils.config import ConfigLoader
from insurance_reserve_intelligence_platform.utils.device import DeviceManager


def test_optimization_engine_returns_result_objects() -> None:
    config = ConfigLoader.load("configs/config.yaml")
    simulator = PolicySimulator(
        age_range=(35, 35),
        term_range=(15, 15),
        premium_range=(1200.0, 1200.0),
        interest_rate_range=(0.04, 0.04),
        sum_assured_range=(150000.0, 150000.0),
        seed=11,
    )
    policy = simulator.generate_random_policies(1)[0]
    engine = OptimizationEngine(
        model=build_model(config),
        device=DeviceManager(prefer_mixed_precision=False).device,
        config=config.optimization,
    )
    result = engine.target_reserve_optimization(policy, target_reserve=1000.0)
    assert result.variable_name == "interest_rate"
