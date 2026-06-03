<!--
Created: 2026-06-03
Purpose: Describe the knowledge-informed reserve losses used beyond the core PINN objective.
-->

# KINN Losses

KINN losses encode domain knowledge that may not be written directly as the governing ODE.

## Monotonicity

- `mortality_monotonicity_loss`: enforce `dV/dμ >= 0`
- `age_monotonicity_loss`: enforce `dV/dx >= 0`
- `interest_rate_monotonicity_loss`: enforce `dV/dr <= 0`

These losses help reserve sensitivities move in directions actuaries expect.

## Business Bounds

- `solvency_loss`: enforce `V >= 0`
- `reserve_ceiling_loss`: enforce `V <= S`

These losses make reserve surfaces safer for reporting, pricing, and stress use.

## Shape and Aggregation

- `smoothness_loss`: enforce low curvature through `mean((d²V/dt²)^2)`
- `portfolio_consistency_loss`: enforce additivity through `mean((V_portfolio - sum(V_i))^2)`

These losses make the model easier to govern at both policy and portfolio level.

## Business Translation

In lay terms, KINN losses tell the model:

1. move reserves in the right direction when key drivers change
2. stay inside sensible business bounds
3. produce smooth curves and consistent rollups
