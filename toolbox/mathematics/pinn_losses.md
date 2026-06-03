<!--
Created: 2026-06-03
Purpose: Describe the pure PINN loss family for reserve learning.
-->

# PINN Losses

PINN losses encode data and governing-equation structure.

## `data_loss`

- Formula: `mean((V_pred - V_true)^2)`
- Role: benchmark fit
- Useful when: the classical solver is treated as the numerical source of truth
- Risk: can dominate structural learning if weighted too heavily

## `pde_loss`

- Formula: `mean((dV/dt - rV - P + μ(S - V))^2)`
- Role: physics or actuarial consistency
- Useful when: users want the learned reserve to satisfy Thiele dynamics
- Risk: derivative noise can make optimization harder

## `boundary_loss`

- Formula: `mean(V(T)^2)`
- Role: terminal condition enforcement
- Useful when: modeling term insurance with zero terminal reserve
- Risk: an excessively large weight can distort pre-maturity fit

## Business Translation

In lay terms, PINN losses tell the model:

1. match the reserve engine
2. obey the reserve equation
3. end the contract with no leftover liability
