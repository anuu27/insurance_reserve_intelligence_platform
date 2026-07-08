"""Datasets for reserve learning.

Created: 2026-05-31
Purpose: Build training records by pairing policies with classical reserve trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.actuarial.actuarial_solver import BaseActuarialSolver
from src.actuarial.policy import Policy


FEATURE_INDEX: dict[str, int] = {
    "time": 0,
    "age": 1,
    "interest_rate": 2,
    "premium": 3,
    "sum_assured": 4,
    "mortality": 5,
}

FEATURE_SCALES: dict[str, float] = {
    "time": 30.0,
    "age": 45.0,
    "interest_rate": 0.07,
    "premium": 0.02,
    "sum_assured": 950_000.0,
    "mortality": 0.05,
}


@dataclass(slots=True)
class ReserveRecord:
    """Single reserve-learning record."""

    policy_id: str
    features: np.ndarray
    reserve: float
    term: float


class ReserveDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of normalized PINN inputs and z-scored reserve-ratio targets."""

    def __init__(
        self,
        policies: Iterable[Policy],
        solver: BaseActuarialSolver,
        time_steps: int,
        target_mean: float | None = None,
        target_std: float | None = None,
    ) -> None:
        self.records = self._build_records(
            list(policies),
            solver=solver,
            time_steps=time_steps,
        )

        if not self.records:
            raise ValueError("ReserveDataset cannot be built from zero records.")

        premium_values = np.asarray(
            [record.features[FEATURE_INDEX["premium"]] for record in self.records],
            dtype=np.float32,
        )
        self.premium_mean = float(premium_values.mean())
        self.premium_std = float(premium_values.std())
        if self.premium_std < 1e-8:
            self.premium_std = 1.0

        interest_values = np.asarray(
            [record.features[FEATURE_INDEX["interest_rate"]] for record in self.records],
            dtype=np.float32,
        )
        self.interest_mean = float(interest_values.mean())
        self.interest_std = float(interest_values.std())
        if self.interest_std < 1e-8:
            self.interest_std = 1.0

        reserves = np.asarray(
            [record.reserve for record in self.records],
            dtype=np.float64,
        )

        print("\n")
        print("=" * 60)
        print("DATASET RESERVE STATISTICS")
        print("=" * 60)
        print(f"Minimum Reserve : {reserves.min():,.2f}")
        print(f"Maximum Reserve : {reserves.max():,.2f}")
        print(f"Mean Reserve    : {reserves.mean():,.2f}")
        print(f"Std Reserve     : {reserves.std():,.2f}")
        print("=" * 60)

        reserve_ratios = np.asarray(
            [
                record.reserve / max(
                    float(record.features[FEATURE_INDEX["sum_assured"]]),
                    1.0,
                )
                for record in self.records
            ],
            dtype=np.float32,
        )

        self.target_mean = (
            float(reserve_ratios.mean())
            if target_mean is None
            else float(target_mean)
        )

        if target_std is None:
            std = float(reserve_ratios.std())
            self.target_std = std if std > 1e-8 else 1.0
        else:
            self.target_std = float(target_std)

        print("\nNormalization")
        print(f"interest_mean = {self.interest_mean:.8f}")
        print(f"interest_std  = {self.interest_std:.8f}")
        print(f"premium_mean  = {self.premium_mean:.8f}")
        print(f"premium_std   = {self.premium_std:.8f}")
        print(f"target_mean   = {self.target_mean:.8f}")
        print(f"target_std    = {self.target_std:.8f}")

        self.normalization = {
            "interest_mean": self.interest_mean,
            "interest_std": self.interest_std,
            "premium_mean": self.premium_mean,
            "premium_std": self.premium_std,
            "target_mean": self.target_mean,
            "target_std": self.target_std,
        }

    @staticmethod
    def _build_records(
        policies: list[Policy],
        solver: BaseActuarialSolver,
        time_steps: int,
    ) -> list[ReserveRecord]:
        records: list[ReserveRecord] = []

        for policy in policies:
            trajectory = solver.solve(policy=policy, num_steps=time_steps)

            for time_point, reserve in zip(trajectory.times, trajectory.reserves):
                mortality = policy.mortality_profile.intensity_at(time_point)
                premium_ratio = float(policy.premium) / max(float(policy.sum_assured), 1.0)

                features = np.asarray(
                    [
                        float(time_point),
                        float(policy.age),
                        float(policy.interest_rate),
                        premium_ratio,
                        float(policy.sum_assured),
                        float(mortality),
                    ],
                    dtype=np.float32,
                )

                records.append(
                    ReserveRecord(
                        policy_id=policy.policy_id,
                        features=features,
                        reserve=float(reserve),
                        term=float(policy.term),
                    )
                )

        return records

    def __len__(self) -> int:
        return len(self.records)

    def normalize_features(self, raw_features: torch.Tensor) -> torch.Tensor:
        """Normalize raw PINN features using the exact training contract."""

        features = raw_features.clone()

        features[..., FEATURE_INDEX["time"]] /= FEATURE_SCALES["time"]
        features[..., FEATURE_INDEX["age"]] /= FEATURE_SCALES["age"]

        features[..., FEATURE_INDEX["interest_rate"]] = (
            features[..., FEATURE_INDEX["interest_rate"]] - self.interest_mean
        ) / self.interest_std

        features[..., FEATURE_INDEX["premium"]] = (
            features[..., FEATURE_INDEX["premium"]] - self.premium_mean
        ) / self.premium_std

        features[..., FEATURE_INDEX["sum_assured"]] /= FEATURE_SCALES["sum_assured"]
        features[..., FEATURE_INDEX["mortality"]] /= FEATURE_SCALES["mortality"]

        return features

    def denormalize_target(
        self,
        z_value: torch.Tensor | float,
        sum_assured: torch.Tensor | float,
    ) -> torch.Tensor | float:
        """Convert model z-output back into reserve currency units."""

        if isinstance(z_value, torch.Tensor):
            scale = torch.clamp(
                torch.as_tensor(
                    sum_assured,
                    dtype=z_value.dtype,
                    device=z_value.device,
                ),
                min=1.0,
            )
            return (z_value * self.target_std + self.target_mean) * scale

        scale_value = max(float(sum_assured), 1.0)
        return (float(z_value) * self.target_std + self.target_mean) * scale_value

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]

        raw_features = torch.tensor(record.features, dtype=torch.float32)
        features = self.normalize_features(raw_features)

        sum_assured_value = float(raw_features[FEATURE_INDEX["sum_assured"]].item())
        sum_assured_scale = max(sum_assured_value, 1.0)

        reserve_ratio = float(record.reserve) / sum_assured_scale
        z_target = (reserve_ratio - self.target_mean) / self.target_std

        return {
            "features": features,
            "raw_features": raw_features,
            "target": torch.tensor([z_target], dtype=torch.float32),
            "sum_assured_scale": torch.tensor([sum_assured_scale], dtype=torch.float32),
            "target_mean": torch.tensor([self.target_mean], dtype=torch.float32),
            "target_std": torch.tensor([self.target_std], dtype=torch.float32),
            "term": torch.tensor([record.term], dtype=torch.float32),
            "interest_mean": torch.tensor([self.interest_mean], dtype=torch.float32),
            "interest_std": torch.tensor([self.interest_std], dtype=torch.float32),
            "premium_mean": torch.tensor([self.premium_mean], dtype=torch.float32),
            "premium_std": torch.tensor([self.premium_std], dtype=torch.float32),
        }