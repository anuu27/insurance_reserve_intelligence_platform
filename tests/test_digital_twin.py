"""Digital twin tests."""

from __future__ import annotations

from insurance_reserve_intelligence_platform.digital_twin.engine import DigitalTwinEngine
from insurance_reserve_intelligence_platform.pipeline import build_model
from insurance_reserve_intelligence_platform.utils.config import ConfigLoader
from insurance_reserve_intelligence_platform.utils.device import DeviceManager
from insurance_reserve_intelligence_platform.data.simulator import PolicySimulator


def test_digital_twin_forecast_has_rows() -> None:
    config = ConfigLoader.load("configs/config.yaml")
    simulator = PolicySimulator(
        age_range=(40, 40),
        term_range=(10, 10),
        premium_range=(1000.0, 1000.0),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        seed=3,
    )
    policy = simulator.generate_random_policies(1)[0]
    engine = DigitalTwinEngine(
        model=build_model(config),
        device=DeviceManager(prefer_mixed_precision=False).device,
        config=config.digital_twin,
    )
    frame = engine.reserve_forecast(policy, steps=5)
    assert len(frame) == 5
