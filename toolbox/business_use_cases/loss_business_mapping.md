<!--
Created: 2026-06-03
Purpose: Map reserve-learning losses to business use cases and stakeholder value.
-->

# Loss Business Mapping

## Technique to Business Interpretation

- `data_loss`: keep the AI reserve close to the incumbent reserve engine
- `pde_loss`: make reserves behave like a real insurance liability process
- `boundary_loss`: avoid residual liabilities after the policy ends
- `mortality_monotonicity_loss`: ensure mortality stress raises reserve appropriately
- `age_monotonicity_loss`: make older-risk reserves trend upward when appropriate
- `interest_rate_monotonicity_loss`: ensure higher discount rates generally reduce reserve
- `solvency_loss`: prevent negative reserve outputs
- `reserve_ceiling_loss`: stop implausibly high reserve spikes
- `smoothness_loss`: deliver stable reserve paths and sensitivities
- `portfolio_consistency_loss`: align policy and portfolio reserve views
- `l2_regularization_loss`: reduce brittle overfit behavior

## Stakeholder Mapping

- Actuarial research: `data_loss`, `pde_loss`, `boundary_loss`
- Model risk and validation: monotonicity losses, solvency, ceiling, smoothness
- Finance and capital teams: portfolio consistency, solvency, ceiling
- Pricing and product: data loss plus monotonicity and rate sensitivity controls
- Executive users: smoothness and business-bound losses because they improve interpretability

## When to Use Which Family

- Pure PINN: when faithfulness to the reserve equation is the main research goal
- PINN + KINN: when the model must satisfy both mechanics and business priors
- KINN-style supervised: when data fit and actuarial intuition matter more than explicit PDE enforcement
- Supervised only: when building a benchmark ablation model
