"""Stress testing workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.actuarial.policy import Policy
from src.utils.config import StressScenarioConfig
from src.visualization.stress_plots import plot_stress_comparison


@dataclass(slots=True)
class StressResult:
    """Stress test output summary."""

    policy_id: str
    scenario_name: str
    before_reserve: float
    after_reserve: float
    delta: float
    delta_pct: float


class StressTester:
    """Apply actuarial and macro shocks to reserve predictions."""

    def __init__(self, model: torch.nn.Module, device: torch.device, config: StressScenarioConfig) -> None:
        self.model = model.to(device)
        self.device = device
        self.config = config

    def _policy_features(self, policy: Policy, time_point: float = 0.0) -> torch.Tensor:
        mortality = policy.mortality_profile.intensity_at(time_point)
        features = torch.tensor(
            [[time_point, float(policy.age), policy.interest_rate, policy.premium, policy.sum_assured, mortality]],
            dtype=torch.float32,
            device=self.device,
        )
        return features

    def _predict(self, features: torch.Tensor) -> float:
        self.model.eval()
        with torch.no_grad():
            return float(self.model(features).item())

    def _apply_shock(self, policy: Policy, scenario_name: str) -> torch.Tensor:
        features = self._policy_features(policy)
        shocked = features.clone()
        if scenario_name == "mortality_shock":
            shocked[:, 5] *= 1.0 + self.config.mortality_shock
        elif scenario_name == "interest_rate_shock":
            shocked[:, 2] += self.config.interest_rate_shock
        elif scenario_name == "inflation_shock":
            shocked[:, 4] *= 1.0 + self.config.inflation_shock
            shocked[:, 3] *= 1.0 + 0.5 * self.config.inflation_shock
        elif scenario_name == "longevity_shock":
            shocked[:, 5] *= 1.0 + self.config.longevity_shock
        elif scenario_name == "lapse_shock":
            shocked[:, 3] *= 1.0 - self.config.lapse_shock
        else:
            raise ValueError(f"Unsupported stress scenario: {scenario_name}")
        return shocked

    def run_scenario(self, policies: list[Policy], scenario_name: str) -> pd.DataFrame:
        """Run one stress scenario across a portfolio."""

        results: list[StressResult] = []
        for policy in policies:
            baseline_features = self._policy_features(policy)
            shocked_features = self._apply_shock(policy, scenario_name)
            before = self._predict(baseline_features)
            after = self._predict(shocked_features)
            delta = after - before
            delta_pct = delta / before if abs(before) > 1e-12 else np.nan
            results.append(
                StressResult(
                    policy_id=policy.policy_id,
                    scenario_name=scenario_name,
                    before_reserve=before,
                    after_reserve=after,
                    delta=delta,
                    delta_pct=delta_pct,
                )
            )
        frame = pd.DataFrame([asdict(result) for result in results])
        return frame

    def run_all(self, policies: list[Policy], output_dir: str) -> dict[str, pd.DataFrame]:
        """Run all configured scenarios, write CSVs, and generate plots."""

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        outputs: dict[str, pd.DataFrame] = {}
        for scenario_name in [
            "mortality_shock",
            "interest_rate_shock",
            "inflation_shock",
            "longevity_shock",
            "lapse_shock",
        ]:
            frame = self.run_scenario(policies, scenario_name)
            csv_path = Path(output_dir) / f"{scenario_name}.csv"
            frame.to_csv(csv_path, index=False)
            plot_stress_comparison(frame, str(Path(output_dir) / f"{scenario_name}.png"))
            outputs[scenario_name] = frame
        return outputs
