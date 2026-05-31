"""Policy domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class MortalityProfile:
    """Discrete mortality profile for a policy."""

    times: np.ndarray
    intensities: np.ndarray
    source: str = "synthetic"

    def intensity_at(self, time_point: float) -> float:
        """Interpolate mortality intensity at an arbitrary time point."""

        return float(np.interp(time_point, self.times, self.intensities))


@dataclass(slots=True)
class Policy:
    """Term-life policy description."""

    policy_id: str
    age: int
    term: int
    premium: float
    interest_rate: float
    sum_assured: float
    mortality_profile: MortalityProfile
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def times(self, num_steps: int) -> np.ndarray:
        """Return evenly spaced time points over the policy term."""

        return np.linspace(0.0, float(self.term), num_steps, dtype=float)


def build_mortality_profile(times: Sequence[float], intensities: Sequence[float], source: str) -> MortalityProfile:
    """Construct a mortality profile from iterable inputs."""

    return MortalityProfile(times=np.asarray(times, dtype=float), intensities=np.asarray(intensities, dtype=float), source=source)
