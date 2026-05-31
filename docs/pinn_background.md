# PINN Background

## What is a PINN

A Physics-Informed Neural Network learns a function while being penalized for violating governing equations and boundary conditions. Instead of relying only on observed labels, it also uses structural knowledge embedded in residual terms.

## Why PINNs Fit Insurance Liability Modelling

Insurance liabilities are not arbitrary regression targets. They are shaped by actuarial equations, terminal conditions, and scenario assumptions. A PINN is attractive because it:

- learns reserve values from actuarial targets
- respects reserve dynamics through PDE residual penalties
- provides differentiable outputs for sensitivity and optimization

## Collocation Points

Training does not only use labelled reserve values. The network is also evaluated at sampled policy-time points where the PDE residual is computed. These are the collocation points.

## Automatic Differentiation

PyTorch automatic differentiation provides:

- `dV/dt` for PDE enforcement
- `dV/dr`, `dV/dμ`, `dV/dP`, `dV/dS` for sensitivity analysis
- `d²V/dr²` for curvature analysis

This differentiability is one of the key reasons to use a neural surrogate rather than a black-box tabular approximation.

## Residual Minimization

If the network predicts a reserve surface that is locally accurate but violates the Thiele equation, the PDE loss penalizes that inconsistency. The PINN therefore balances:

- data fit
- physics consistency
- terminal boundary accuracy

## Practical Considerations

- Too much weight on data loss can weaken physics adherence.
- Too much weight on PDE loss can slow convergence if supervised targets are noisy.
- Boundary loss is especially important near maturity.
- Mixed precision can accelerate training on CUDA devices.

The platform exposes all loss weights and training hyperparameters in YAML for controlled experimentation.
