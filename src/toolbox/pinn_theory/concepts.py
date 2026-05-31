"""PINN theory helpers.

Created: 2026-05-31
Purpose: Store core PINN terminology used by the platform and documentation.
"""

from __future__ import annotations


PINN_CONCEPTS: dict[str, str] = {
    "collocation_points": "Sampled points where the PDE residual is penalized.",
    "autodiff": "Automatic differentiation used to compute derivatives such as dV/dt.",
    "boundary_condition": "Constraint enforcing terminal reserve equal to zero for term life.",
}
