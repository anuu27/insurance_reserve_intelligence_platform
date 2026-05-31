"""Actuarial solver tests.

Created: 2026-05-31
Purpose: Validate numerical reserve-solver boundary behavior.
"""

from __future__ import annotations

from src.actuarial.actuarial_solver import ThieleSolver
from src.data.simulator import PolicySimulator


def test_thiele_solver_returns_boundary_zero() -> None:
    """Verify that the solved term-life reserve ends at the zero terminal boundary."""
    simulator = PolicySimulator(
        age_range=(30, 30),
        term_range=(10, 10),
        premium_range=(1000.0, 1000.0),
        interest_rate_range=(0.03, 0.03),
        sum_assured_range=(100000.0, 100000.0),
        seed=1,
    )
    policy = simulator.generate_random_policies(1)[0]
    solver = ThieleSolver(method="rk4")
    trajectory = solver.solve(policy=policy, num_steps=20)
    assert trajectory.times[0] == 0.0
    assert abs(trajectory.reserves[-1]) < 1e-6
