# Stress Testing

## Purpose

Stress testing turns the reserve engine into a risk-management tool by quantifying how liability values move under adverse assumptions.

## Implemented Scenarios

### Mortality Shock

Applies an upward shock to mortality intensity. This is relevant for catastrophic mortality events.

### Interest-Rate Shock

Applies a downward or upward change to the interest-rate assumption. This supports asset-liability management analysis.

### Inflation Shock

Scales benefit size and partially scales premiums to represent inflationary cost pressure.

### Longevity Shock

Applies an improvement in mortality, useful as a robustness or assumption sensitivity test even though the initial product is term life.

### Lapse Shock

Reduces premium persistence to emulate adverse policyholder behavior.

## Outputs

- Before-versus-after reserve comparison
- Scenario CSV reports
- Stress comparison plots

## Business Interpretation

Stress results can be used to:

- explain reserve movement drivers
- prioritize assumption governance
- compare vulnerability across portfolios
- support executive capital discussions
