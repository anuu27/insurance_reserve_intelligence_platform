"""Policy domain models.

Created: 2026-05-31
Purpose: Define policy and mortality containers used throughout the platform.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(slots=True)
class MortalityProfile:
    """Discrete mortality profile for a policy.

    Attributes:
        times: Time grid aligned to the mortality intensities.
        intensities: Mortality intensity values by time point.
        source: Origin of the mortality assumptions.

    Scientific Context:
        The mortality profile is a discrete representation of the force of
        mortality ``μ(t)`` used in the reserve equation.

    Business Interpretation:
        This is the death-risk curve for a policy. In plain terms, it tells the
        platform how claim risk changes over the contract lifetime.
    """

    times: np.ndarray
    intensities: np.ndarray
    source: str = "synthetic"

    def intensity_at(self, time_point: float) -> float:
        """Interpolate mortality intensity at an arbitrary time point.

        Args:
            time_point: Elapsed time at which to evaluate mortality.

        Returns:
            float: Interpolated mortality intensity.
        """

        return float(np.interp(time_point, self.times, self.intensities))


@dataclass(slots=True)
class Policy:
    """Term-life policy description.

    Attributes:
        policy_id: Unique policy identifier.
        age: Age at policy inception.
        term: Policy term in years.
        premium: Premium level used in valuation.
        interest_rate: Interest-rate assumption for reserve growth.
        sum_assured: Death benefit amount.
        mortality_profile: Mortality assumptions over the policy term.
        metadata: Optional auxiliary policy descriptors.

    Business Interpretation:
        This object is the digital representation of an insurance contract used by
        the reserve engine, stress tester, optimizer, and digital twin.
    """

    policy_id: str
    age: int
    term: int
    premium: float
    interest_rate: float
    sum_assured: float
    mortality_profile: MortalityProfile
    metadata: dict[str, float | int | str] = field(default_factory=dict)

    def times(self, num_steps: int) -> np.ndarray:
        """Return evenly spaced time points over the policy term.

        Args:
            num_steps: Number of time points to generate.

        Returns:
            np.ndarray: Evenly spaced time grid over the term.

        Business Interpretation:
            This grid is the schedule over which the system values and visualizes
            liability evolution.
        """

        return np.linspace(0.0, float(self.term), num_steps, dtype=float)


def build_mortality_profile(times: Sequence[float], intensities: Sequence[float], source: str) -> MortalityProfile:
    """Construct a mortality profile from iterable inputs.

    Args:
        times: Time points aligned to the intensities.
        intensities: Mortality intensity values.
        source: Origin label for the mortality assumptions.

    Returns:
        MortalityProfile: Structured mortality profile instance.

    Business Interpretation:
        This helper turns raw mortality inputs into a reusable contract-risk
        object that the rest of the platform can consume consistently.
    """

    return MortalityProfile(times=np.asarray(times, dtype=float), intensities=np.asarray(intensities, dtype=float), source=source)
