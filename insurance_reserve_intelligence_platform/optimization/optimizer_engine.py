"""Optimization workflows for reserve management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize, minimize_scalar

from insurance_reserve_intelligence_platform.actuarial.policy import Policy
from insurance_reserve_intelligence_platform.utils.config import OptimizationConfig


@dataclass(slots=True)
class OptimizationResult:
    """Optimization summary."""

    variable_name: str
    optimal_value: float
    objective_value: float
    method: str
    success: bool


class OptimizationEngine:
    """Optimize premiums, interest rates, and reserve-related objectives."""

    def __init__(self, model: torch.nn.Module, device: torch.device, config: OptimizationConfig) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

    def _features_from_policy(self, policy: Policy) -> torch.Tensor:
        return torch.tensor(
            [[
                0.0,
                float(policy.age),
                policy.interest_rate,
                policy.premium,
                policy.sum_assured,
                policy.mortality_profile.intensity_at(0.0),
            ]],
            dtype=torch.float32,
            device=self.device,
        )

    def _predict(self, features: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        return self.model(features)

    def target_reserve_optimization(self, policy: Policy, target_reserve: float) -> OptimizationResult:
        """Find the interest rate that produces a target reserve."""

        def objective(interest_rate: float) -> float:
            features = self._features_from_policy(policy).clone()
            features[:, 2] = interest_rate
            reserve = float(self._predict(features).detach().cpu().item())
            return (reserve - target_reserve) ** 2

        result = minimize_scalar(objective, bounds=(-0.02, 0.15), method="bounded")
        return OptimizationResult(
            variable_name="interest_rate",
            optimal_value=float(result.x),
            objective_value=float(result.fun),
            method="scipy_minimize_scalar",
            success=bool(result.success),
        )

    def premium_optimization(self, policy: Policy) -> OptimizationResult:
        """Use gradient ascent to maximize a simple profitability proxy."""

        premium = torch.tensor([policy.premium], dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([premium], lr=self.config.learning_rate)

        for _ in range(self.config.steps):
            optimizer.zero_grad(set_to_none=True)
            features = self._features_from_policy(policy).clone()
            features[:, 3] = premium
            reserve = self._predict(features)
            profitability = premium - 0.01 * reserve
            loss = -profitability.mean()
            loss.backward()
            optimizer.step()

        optimized_features = self._features_from_policy(policy).clone()
        optimized_features[:, 3] = premium.detach()
        objective_value = float((premium - 0.01 * self._predict(optimized_features)).detach().cpu().item())
        return OptimizationResult(
            variable_name="premium",
            optimal_value=float(premium.detach().cpu().item()),
            objective_value=objective_value,
            method="gradient_based",
            success=True,
        )

    def constrained_premium_optimization(self, policy: Policy) -> OptimizationResult:
        """Maximize profitability subject to a reserve floor."""

        def objective(values: np.ndarray) -> float:
            premium_value = float(values[0])
            features = self._features_from_policy(policy).clone()
            features[:, 3] = premium_value
            reserve = float(self._predict(features).detach().cpu().item())
            profitability = premium_value - 0.01 * reserve
            penalty = max(0.0, self.config.solvency_threshold - reserve) ** 2
            return -(profitability - 1000.0 * penalty)

        result = minimize(
            objective,
            x0=np.asarray([policy.premium], dtype=float),
            bounds=[(policy.premium * 0.5, policy.premium * 2.0)],
            method="L-BFGS-B",
        )
        return OptimizationResult(
            variable_name="premium",
            optimal_value=float(result.x[0]),
            objective_value=float(-result.fun),
            method="scipy_constrained",
            success=bool(result.success),
        )

    def bayesian_optimization_hook(
        self,
        policy: Policy,
        optimizer_fn: Callable[[Callable[[float], float]], tuple[float, float]],
    ) -> OptimizationResult:
        """Allow external Bayesian optimization libraries to plug in cleanly."""

        def objective(premium_value: float) -> float:
            features = self._features_from_policy(policy).clone()
            features[:, 3] = premium_value
            reserve = float(self._predict(features).detach().cpu().item())
            return -(premium_value - 0.01 * reserve)

        optimal_value, objective_value = optimizer_fn(objective)
        return OptimizationResult(
            variable_name="premium",
            optimal_value=float(optimal_value),
            objective_value=float(objective_value),
            method="bayesian_hook",
            success=True,
        )
