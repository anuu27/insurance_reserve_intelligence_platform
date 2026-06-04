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


FEATURE_INDEX = {
    "time": 0,
    "age": 1,
    "interest_rate": 2,
    "premium": 3,
    "sum_assured": 4,
    "mortality": 5,
}

FEATURE_SCALES = {
    "time": 30.0,
    "age": 100.0,
    "interest_rate": 0.1,
    "premium": 10_000.0,
    "sum_assured": 1_000_000.0,
    "mortality": 0.05,
}


@dataclass(slots=True)
class ReserveRecord:
    """Single collocation record used for training or evaluation.

    Attributes:
        policy_id: Policy identifier associated with the record.
        features: Feature vector used by the PINN.
        reserve: Classical reserve target value.
        term: Policy maturity used for the boundary condition.
    """

    policy_id: str
    features: np.ndarray
    reserve: float
    term: float


class ReserveDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of reserve trajectories derived from classical actuarial solutions."""

    def __init__(self, policies: Iterable[Policy], solver: BaseActuarialSolver, time_steps: int) -> None:
        """Initialize the dataset.

        Args:
            policies: Policies to expand into reserve records.
            solver: Classical solver used to generate target trajectories.
            time_steps: Number of time steps per policy.
        """
        self.records = self._build_records(list(policies), solver=solver, time_steps=time_steps)

    @staticmethod
    def _build_records(policies: list[Policy], solver: BaseActuarialSolver, time_steps: int) -> list[ReserveRecord]:
        """Convert policies into reserve-learning records.

        Args:
            policies: Policies to transform.
            solver: Classical solver used to generate reserve trajectories.
            time_steps: Number of time steps per policy.

        Returns:
            list[ReserveRecord]: Flattened learning records.
        """
        records: list[ReserveRecord] = []
        for policy in policies:
            trajectory = solver.solve(policy=policy, num_steps=time_steps)
            for time_point, reserve in zip(trajectory.times, trajectory.reserves):
                mortality = policy.mortality_profile.intensity_at(time_point)
                features = np.asarray(
                    [time_point, float(policy.age), policy.interest_rate, policy.premium, policy.sum_assured, mortality],
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
        """Return the number of records in the dataset.

        Returns:
            int: Dataset size.
        """
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Retrieve one record as tensors.

        Args:
            index: Record index to fetch.

        Returns:
            dict[str, torch.Tensor]: Feature, target, and term tensors.
        """
        record = self.records[index]
        raw_features = torch.tensor(record.features, dtype=torch.float32)
        features = raw_features.clone()

        features[FEATURE_INDEX["time"]] /= FEATURE_SCALES["time"]
        features[FEATURE_INDEX["age"]] /= FEATURE_SCALES["age"]
        features[FEATURE_INDEX["interest_rate"]] /= FEATURE_SCALES["interest_rate"]
        features[FEATURE_INDEX["premium"]] /= FEATURE_SCALES["premium"]
        features[FEATURE_INDEX["sum_assured"]] /= FEATURE_SCALES["sum_assured"]
        features[FEATURE_INDEX["mortality"]] /= FEATURE_SCALES["mortality"]
        return {
            "features": features,
            "raw_features": raw_features,
            "target": torch.tensor([record.reserve], dtype=torch.float32),
            "term": torch.tensor([record.term], dtype=torch.float32),
        }
