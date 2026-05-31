"""Datasets for reserve learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from src.actuarial.actuarial_solver import BaseActuarialSolver
from src.actuarial.policy import Policy


@dataclass(slots=True)
class ReserveRecord:
    """Single collocation record used for training or evaluation."""

    policy_id: str
    features: np.ndarray
    reserve: float
    term: float


class ReserveDataset(Dataset[dict[str, torch.Tensor]]):
    """Dataset of reserve trajectories derived from classical actuarial solutions."""

    def __init__(self, policies: Iterable[Policy], solver: BaseActuarialSolver, time_steps: int) -> None:
        self.records = self._build_records(list(policies), solver=solver, time_steps=time_steps)

    @staticmethod
    def _build_records(policies: list[Policy], solver: BaseActuarialSolver, time_steps: int) -> list[ReserveRecord]:
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
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        features = torch.tensor(record.features, dtype=torch.float32)
        return {
            "features": features,
            "target": torch.tensor([record.reserve], dtype=torch.float32),
            "term": torch.tensor([record.term], dtype=torch.float32),
        }
