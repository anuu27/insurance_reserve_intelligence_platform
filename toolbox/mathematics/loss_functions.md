<!--
Created: 2026-06-03
Purpose: Summarize the mathematical reserve-loss family used by the platform.
-->

# Loss Functions

The platform models reserve learning as a weighted objective:

`L_total = Σ_i w_i L_i`

where `w_i` is configured in YAML and each `L_i` is a named loss.

## Core Variables

- `V`: predicted reserve
- `V_true`: reserve from the classical actuarial solver
- `t`: elapsed policy time
- `x`: attained age or age feature
- `r`: interest rate
- `P`: premium
- `μ`: mortality intensity
- `S`: sum assured
- `θ`: neural network parameters

## Loss Inventory

- `L_data = mean((V - V_true)^2)`
- `L_pde = mean((dV/dt - rV - P + μ(S - V))^2)`
- `L_boundary = mean(V(T)^2)`
- `L_mortality = mean(ReLU(-dV/dμ))`
- `L_age = mean(ReLU(-dV/dx))`
- `L_rate = mean(ReLU(dV/dr))`
- `L_solvency = mean(ReLU(-V))`
- `L_ceiling = mean(ReLU(V - S))`
- `L_smooth = mean((d²V/dt²)^2)`
- `L_portfolio = mean((V_portfolio - sum(V_i))^2)`
- `L_reg = sum(||θ||²)`

## Why Multiple Losses Exist

Each loss controls a different failure mode:

- data fit loss controls benchmark accuracy
- PDE loss controls actuarial dynamics
- boundary loss controls terminal contract behavior
- monotonicity losses control directional business logic
- solvency and ceiling losses control business-safe bounds
- smoothness loss controls curve stability
- portfolio consistency controls rollup coherence
- L2 regularization controls parameter complexity
