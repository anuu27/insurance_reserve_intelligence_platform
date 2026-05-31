# Actuarial Background

## Mortality Tables

Mortality tables summarize age-dependent death rates used to value insurance liabilities. In this project, mortality is represented through intensity curves that can come from:

- Human Mortality Database extracts
- WHO mortality extracts
- Offline CSV files
- Synthetic Gompertz-style curves when needed for simulation

## Actuarial Reserves

An actuarial reserve is the present value of future outgo less future income under selected assumptions. For term life insurance, the main components are:

- Future death benefit payments
- Future premium income
- Investment earnings on held reserves

## Thiele Equation

The Thiele differential equation gives a dynamic reserve balance for contingent insurance liabilities. In the one-state term-life setup implemented here, reserve evolution depends on:

- interest accumulation
- premium inflow
- mortality-driven claim pressure

This project uses the Thiele equation as the actuarial truth model and as the governing physics constraint inside the PINN.

## Term Insurance

Term life provides a death benefit if the insured dies during a finite policy term. The boundary condition `V(T)=0` is natural because no liability remains after the term expires.

## Why Numerical Solvers Matter

Even when a closed form exists for simplified assumptions, a numerical reserve solver remains valuable because it:

- generalizes to richer decrement structures
- supports irregular mortality shapes
- provides benchmark trajectories for ML validation

The platform includes both Runge-Kutta and `scipy.solve_ivp` implementations for that reason.
