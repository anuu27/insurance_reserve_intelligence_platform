<!--
Created: 2026-06-03
Purpose: Document the complete config-driven PINN and KINN loss framework.
-->

# Loss System

The platform uses a fully config-driven loss framework. Users enable or disable named losses in `configs/config.yaml` without editing Python code. This supports four experiment styles:

1. Pure PINN
2. PINN + KINN hybrid
3. KINN-style supervised model
4. Supervised-only reserve model

## Configuration Pattern

```yaml
losses:
  data_loss:
    enabled: true
    weight: 1.0
  pde_loss:
    enabled: true
    weight: 1.0
  boundary_loss:
    enabled: true
    weight: 10.0

loss_settings:
  reduction: mean
  use_adaptive_weights: false
```

## Loss Classes

PINN losses:

- `data_loss`
- `pde_loss`
- `boundary_loss`

KINN losses:

- `mortality_monotonicity_loss`
- `age_monotonicity_loss`
- `interest_rate_monotonicity_loss`
- `solvency_loss`
- `reserve_ceiling_loss`
- `smoothness_loss`
- `portfolio_consistency_loss`

Regularization loss:

- `l2_regularization_loss`

## Per-Loss Guidance

### `data_loss`

- Type: PINN or supervised anchor
- Formula: `L_data = mean((V_pred - V_true)^2)`
- Variables: `V_pred` is the neural reserve; `V_true` is the actuarial benchmark reserve
- What it does: Fits the model to classical solver outputs
- Why it is useful: Gives the model direct numerical grounding
- Business implication: Keeps the surrogate close to the incumbent reserve engine
- When to enable: Almost always
- When to disable: Rarely, mainly when studying pure constraint satisfaction
- Expected training impact: Stabilizes convergence and improves point accuracy
- Known risk: Over-weighting can cause the model to memorize solver noise and under-use physics

### `pde_loss`

- Type: PINN
- Formula: `f = dV/dt - rV - P + μ(S - V)`, `L_pde = mean(f^2)`
- Variables: `t` time, `r` interest rate, `P` premium, `μ` mortality intensity, `S` sum assured
- What it does: Penalizes violations of the Thiele reserve equation
- Why it is useful: Enforces actuarial dynamics rather than pure interpolation
- Business implication: Improves trust in scenario and sensitivity outputs
- When to enable: PINN and hybrid experiments
- When to disable: Supervised-only benchmarking or when PDE enforcement blocks early exploration
- Expected training impact: Adds structural discipline but can slow optimization
- Known risk: Bad feature scaling or noisy gradients can destabilize training

### `boundary_loss`

- Type: PINN
- Formula: `V(T)=0`, `L_boundary = mean(V(T)^2)`
- Variables: `T` policy term
- What it does: Forces the terminal term-life reserve to zero
- Why it is useful: Encodes a non-negotiable product boundary condition
- Business implication: Prevents phantom liabilities after coverage expiry
- When to enable: Term-life experiments
- When to disable: Only when modeling products with non-zero terminal value
- Expected training impact: Sharpens tail behavior near maturity
- Known risk: If over-weighted, it can distort earlier parts of the reserve curve

### `mortality_monotonicity_loss`

- Type: KINN
- Formula: `L_mortality = mean(ReLU(-dV/dμ))`
- Variables: `μ` mortality intensity
- What it does: Penalizes reserves that fall as mortality rises
- Why it is useful: Encodes a common actuarial direction-of-risk prior
- Business implication: Makes mortality stress outputs easier for business stakeholders to trust
- When to enable: Mortality-sensitive term-life experiments
- When to disable: If product structure or pricing basis breaks the monotone relationship
- Expected training impact: Improves monotone stress behavior
- Known risk: Can over-constrain edge cases with unusual premium patterns

### `age_monotonicity_loss`

- Type: KINN
- Formula: `L_age = mean(ReLU(-dV/dx))`
- Variables: `x` attained age
- What it does: Penalizes reserves that decrease as attained age increases
- Why it is useful: Encodes the idea that risk often rises with age
- Business implication: Supports more intuitive aging and cohort analytics
- When to enable: When age monotonicity is a reasonable product assumption
- When to disable: If premium structure, term effects, or underwriting create non-monotone age behavior
- Expected training impact: Improves age-gradient smoothness
- Known risk: May bias the model where age effects are product-specific

### `interest_rate_monotonicity_loss`

- Type: KINN
- Formula: `L_rate = mean(ReLU(dV/dr))`
- Variables: `r` interest rate
- What it does: Penalizes reserves that increase as discount rates increase
- Why it is useful: Enforces a standard present-value relationship
- Business implication: Makes rate stress testing more intuitive and defensible
- When to enable: Most liability valuation experiments
- When to disable: If the modeled economics imply a non-standard rate relationship
- Expected training impact: Improves rate sensitivity stability
- Known risk: May conflict with other constraints in unusual regions

### `solvency_loss`

- Type: KINN
- Formula: `L_solvency = mean(ReLU(-V))`
- Variables: `V` reserve
- What it does: Penalizes negative reserves
- Why it is useful: Adds a prudential floor
- Business implication: Reduces obviously problematic outputs for reporting and planning
- When to enable: Production-facing reserve models
- When to disable: Research settings where negative values are intentionally allowed
- Expected training impact: Clips implausible downside behavior
- Known risk: Can hide real modeling issues by flattening negatives rather than fixing them

### `reserve_ceiling_loss`

- Type: KINN
- Formula: `L_ceiling = mean(ReLU(V - S))`
- Variables: `S` sum assured
- What it does: Penalizes reserves above the contractual payout ceiling
- Why it is useful: Prevents runaway liability estimates
- Business implication: Protects stress, optimization, and pricing workflows from implausible spikes
- When to enable: Most term-life experiments
- When to disable: If the reserve basis legitimately allows values above face amount
- Expected training impact: Improves boundedness
- Known risk: Heuristic ceiling may be too strict for some accounting conventions

### `smoothness_loss`

- Type: KINN
- Formula: `L_smooth = mean((d²V/dt²)^2)`
- Variables: `t` time
- What it does: Penalizes excessive curvature in the reserve path
- Why it is useful: Discourages unstable oscillations
- Business implication: Produces reserve curves that are easier to explain and govern
- When to enable: When scenario stability matters
- When to disable: If the true reserve surface has sharp regime changes the model should preserve
- Expected training impact: Smooths trajectories and sensitivities
- Known risk: Can oversmooth legitimate structure

### `portfolio_consistency_loss`

- Type: KINN
- Formula: `L_portfolio = mean((V_portfolio - sum(V_i))^2)`
- Variables: `V_portfolio` aggregate reserve, `V_i` policy reserves
- What it does: Encourages portfolio additivity
- Why it is useful: Aligns policy-level predictions with portfolio rollups
- Business implication: Supports finance and capital aggregation workflows
- When to enable: Portfolio analytics and rollup experiments
- When to disable: If the batch does not represent a meaningful portfolio slice
- Expected training impact: Improves aggregate coherence
- Known risk: Weak batch design can make this signal noisy

### `l2_regularization_loss`

- Type: Regularization
- Formula: `L_reg = sum(||parameter||²)`
- Variables: `parameter` model weights
- What it does: Penalizes large parameter magnitudes
- Why it is useful: Reduces overfitting and stabilizes optimization
- Business implication: Improves robustness of reserve surfaces under new scenarios
- When to enable: Most experiments
- When to disable: Rarely, usually when isolating the impact of structural losses
- Expected training impact: Smaller, smoother parameterization
- Known risk: Excessive weight can underfit the reserve function

## Experiment Patterns

Pure PINN:

```yaml
losses:
  data_loss:
    enabled: true
    weight: 1.0
  pde_loss:
    enabled: true
    weight: 1.0
  boundary_loss:
    enabled: true
    weight: 10.0
  mortality_monotonicity_loss:
    enabled: false
    weight: 0.2
```

PINN + KINN:

```yaml
losses:
  data_loss:
    enabled: true
    weight: 1.0
  pde_loss:
    enabled: true
    weight: 1.0
  boundary_loss:
    enabled: true
    weight: 10.0
  mortality_monotonicity_loss:
    enabled: true
    weight: 0.2
  interest_rate_monotonicity_loss:
    enabled: true
    weight: 0.2
  solvency_loss:
    enabled: true
    weight: 1.0
  smoothness_loss:
    enabled: true
    weight: 0.05
```

Pure KINN-style supervised model:

```yaml
losses:
  data_loss:
    enabled: true
    weight: 1.0
  pde_loss:
    enabled: false
    weight: 1.0
  boundary_loss:
    enabled: false
    weight: 10.0
  mortality_monotonicity_loss:
    enabled: true
    weight: 0.2
  age_monotonicity_loss:
    enabled: true
    weight: 0.2
  solvency_loss:
    enabled: true
    weight: 1.0
```
