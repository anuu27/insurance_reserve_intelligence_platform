"""Policy simulator tests."""

from __future__ import annotations

from insurance_reserve_intelligence_platform.data.simulator import PolicySimulator


def test_policy_simulator_generates_requested_count() -> None:
    simulator = PolicySimulator(
        age_range=(25, 60),
        term_range=(5, 20),
        premium_range=(500.0, 2000.0),
        interest_rate_range=(0.01, 0.05),
        sum_assured_range=(100000.0, 500000.0),
        seed=7,
    )
    policies = simulator.generate_random_policies(5)
    assert len(policies) == 5
    assert all(policy.term >= 5 for policy in policies)
