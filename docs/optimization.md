# Optimization

## Objective

The optimization module converts reserve predictions into actionable actuarial decisions. Because the PINN is differentiable, it can be used directly inside optimization loops.

## Implemented Workflows

### Target Reserve Optimization

This workflow searches for the interest rate that makes the predicted reserve as close as possible to a target value. It is useful for scenario calibration and inverse problem solving.

### Premium Optimization

A gradient-based optimizer adjusts premium to maximize a simple profitability proxy:

```text
profitability ≈ premium - 0.01 * reserve
```

This is intentionally simplified for research extensibility. The architecture allows richer profit and capital formulations later.

### Constrained Premium Optimization

SciPy optimization is used to maximize the same profitability proxy subject to a reserve-based solvency floor through a penalty formulation.

### Bayesian Optimization Hooks

The repository includes a generic hook so external Bayesian optimization packages can be plugged in without changing the engine contract.

## Practical Interpretation

- Inverse problem solving:
  recover assumptions consistent with a desired liability outcome

- Pricing support:
  explore premium adequacy while preserving solvency

- Capital efficiency:
  search for better economic or pricing positions under constraints
