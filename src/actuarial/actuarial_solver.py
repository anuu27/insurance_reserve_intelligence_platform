"""Classical actuarial reserve solvers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from src.actuarial.policy import Policy


@dataclass(slots=True)
class ReserveTrajectory:
    """Numerical reserve solution across a policy term."""

    times: np.ndarray
    reserves: np.ndarray


class BaseActuarialSolver(ABC):
    """Abstract base class for classical reserve solvers."""

    @abstractmethod
    def solve(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        """Solve the reserve equation for a policy."""


class ThieleSolver(BaseActuarialSolver):
    """Numerically solve the term-life Thiele reserve equation."""

    def __init__(self, method: str = "solve_ivp", integration_step: float = 0.25, rtol: float = 1e-6, atol: float = 1e-8) -> None:
        self.method = method
        self.integration_step = integration_step
        self.rtol = rtol
        self.atol = atol

    @staticmethod
    def _rhs(time_point: float, reserve: np.ndarray, policy: Policy) -> np.ndarray:
        reserve_value = float(reserve[0])
        mortality = policy.mortality_profile.intensity_at(time_point)
        derivative = policy.interest_rate * reserve_value + policy.premium - mortality * (policy.sum_assured - reserve_value)
        return np.asarray([derivative], dtype=float)

    def _solve_rk4(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        grid = np.linspace(float(policy.term), 0.0, num_steps, dtype=float)
        reserves = np.zeros_like(grid)
        reserves[0] = 0.0

        for index in range(1, len(grid)):
            t_prev = grid[index - 1]
            y_prev = np.asarray([reserves[index - 1]], dtype=float)
            step = grid[index] - grid[index - 1]
            k1 = self._rhs(t_prev, y_prev, policy)
            k2 = self._rhs(t_prev + step / 2.0, y_prev + step * k1 / 2.0, policy)
            k3 = self._rhs(t_prev + step / 2.0, y_prev + step * k2 / 2.0, policy)
            k4 = self._rhs(grid[index], y_prev + step * k3, policy)
            reserves[index] = float(y_prev[0] + (step / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]))

        return ReserveTrajectory(times=grid[::-1], reserves=reserves[::-1])

    def _solve_ivp(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        evaluation_times = np.linspace(float(policy.term), 0.0, num_steps, dtype=float)
        solution = solve_ivp(
            fun=lambda t, y: self._rhs(t, y, policy),
            t_span=(float(policy.term), 0.0),
            y0=np.asarray([0.0], dtype=float),
            t_eval=evaluation_times,
            method="RK45",
            rtol=self.rtol,
            atol=self.atol,
            max_step=self.integration_step,
        )
        if not solution.success:
            raise RuntimeError(f"Actuarial solver failed: {solution.message}")
        return ReserveTrajectory(times=solution.t[::-1], reserves=solution.y[0][::-1])

    def solve(self, policy: Policy, num_steps: int) -> ReserveTrajectory:
        """Solve the reserve trajectory using the configured numerical routine."""

        if self.method.lower() == "rk4":
            return self._solve_rk4(policy=policy, num_steps=num_steps)
        return self._solve_ivp(policy=policy, num_steps=num_steps)
