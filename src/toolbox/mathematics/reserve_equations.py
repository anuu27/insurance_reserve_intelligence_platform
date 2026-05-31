"""Closed-form helper expressions around the reserve PDE.

Created: 2026-05-31
Purpose: Provide scalar helper functions tied to the reserve differential equation.
"""

from __future__ import annotations


def thiele_residual(dv_dt: float, reserve: float, interest_rate: float, premium: float, mortality: float, sum_assured: float) -> float:
    """Evaluate the scalar Thiele residual.

    Args:
        dv_dt: Time derivative of the reserve.
        reserve: Reserve level ``V``.
        interest_rate: Interest-rate assumption ``r``.
        premium: Premium inflow ``P``.
        mortality: Mortality intensity ``μ``.
        sum_assured: Death benefit ``S``.

    Returns:
        float: Scalar Thiele residual value.

    Scientific Context:
        A zero residual indicates that the reserve state satisfies the governing
        differential equation exactly at the evaluation point.

    Business Interpretation:
        This is a local actuarial consistency score for a reserve estimate.
    """

    return dv_dt - interest_rate * reserve - premium + mortality * (sum_assured - reserve)
