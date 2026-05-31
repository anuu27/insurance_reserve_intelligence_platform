"""Insurance liability digital twin workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from src.actuarial.policy import Policy
from src.data.simulator import PolicySimulator, ScenarioDefinition
from src.utils.config import DigitalTwinConfig


@dataclass(slots=True)
class RegimeDefinition:
    """Macroeconomic regime multipliers."""

    name: str
    interest_rate_shift: float
    mortality_multiplier: float
    inflation_multiplier: float


class DigitalTwinEngine:
    """Scenario-aware digital twin for insurance liabilities."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        config: DigitalTwinConfig,
        simulator: PolicySimulator | None = None,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.simulator = simulator

    def _feature_tensor(self, policy: Policy, time_point: float) -> torch.Tensor:
        return torch.tensor(
            [[
                time_point,
                float(policy.age),
                policy.interest_rate,
                policy.premium,
                policy.sum_assured,
                policy.mortality_profile.intensity_at(time_point),
            ]],
            dtype=torch.float32,
            device=self.device,
        )

    def reserve_forecast(self, policy: Policy, steps: int | None = None) -> pd.DataFrame:
        """Forecast reserves over time for a policy."""

        self.model.eval()
        horizon = steps or self.config.forecast_horizon
        times = np.linspace(0.0, float(policy.term), horizon, dtype=float)
        reserves: list[float] = []
        with torch.no_grad():
            for time_point in times:
                reserves.append(float(self.model(self._feature_tensor(policy, float(time_point))).item()))
        return pd.DataFrame({"time": times, "reserve": reserves, "policy_id": policy.policy_id})

    def scenario_simulation(self, policies: list[Policy], scenario: ScenarioDefinition) -> pd.DataFrame:
        """Run a direct what-if scenario across a portfolio."""

        if self.simulator is None:
            raise ValueError("A PolicySimulator is required for scenario simulation.")
        scenario_policies = self.simulator.generate_scenario_policies(policies, scenario)
        baseline = self.portfolio_simulation(policies)
        stressed = self.portfolio_simulation(scenario_policies)
        return baseline.merge(stressed, on="policy_id", suffixes=("_base", "_scenario"))

    def regime_simulation(self, policies: list[Policy]) -> pd.DataFrame:
        """Simulate predefined macro regimes."""

        regimes = [
            RegimeDefinition("base", 0.0, 1.0, 1.0),
            RegimeDefinition("soft_recession", -0.01, 1.05, 1.01),
            RegimeDefinition("inflationary", 0.02, 1.02, 1.05),
            RegimeDefinition("mortality_crisis", -0.005, 1.20, 1.02),
        ]
        rows: list[dict[str, float | str]] = []
        for regime in regimes:
            for policy in policies:
                features = self._feature_tensor(policy, 0.0)
                features[:, 2] += regime.interest_rate_shift
                features[:, 4] *= regime.inflation_multiplier
                features[:, 5] *= regime.mortality_multiplier
                with torch.no_grad():
                    reserve = float(self.model(features).item())
                rows.append({"policy_id": policy.policy_id, "regime": regime.name, "reserve": reserve})
        return pd.DataFrame(rows)

    def portfolio_simulation(self, policies: list[Policy]) -> pd.DataFrame:
        """Predict portfolio-level reserves policy by policy."""

        rows: list[dict[str, float | str]] = []
        self.model.eval()
        with torch.no_grad():
            for policy in policies:
                reserve = float(self.model(self._feature_tensor(policy, 0.0)).item())
                rows.append({"policy_id": policy.policy_id, "reserve": reserve})
        frame = pd.DataFrame(rows)
        frame["portfolio_reserve"] = frame["reserve"].sum()
        return frame
