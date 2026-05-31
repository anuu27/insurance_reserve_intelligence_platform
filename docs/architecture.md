# Architecture

## Purpose

ActuaryTwin is structured as a layered insurance liability digital twin platform. The architecture deliberately separates domain simulation, actuarial ground truth generation, PINN learning, analysis workflows, and visualization so each concern can evolve independently.

## Layered Design

1. Data and assumption layer
   - Mortality data sources abstract access to Human Mortality Database extracts, WHO extracts, and generic CSV files.
   - The policy simulator generates random, stratified, and scenario-specific synthetic policies.

2. Classical actuarial layer
   - The Thiele solver produces reserve trajectories under the term-life reserve equation.
   - This layer acts as both a benchmark and a supervised target generator.

3. Machine learning layer
   - A configurable PyTorch PINN predicts reserve values from time and policy state variables.
   - PDE residuals and boundary conditions are enforced during training.

4. Orchestration layer
   - The trainer handles checkpointing, logging, early stopping, mixed precision, and resumption.
   - The evaluator, stress tester, optimization engine, and digital twin engine consume the trained model.

5. Reporting layer
   - Visualization modules export reserve, sensitivity, and stress charts.
   - CSV and TensorBoard outputs support research traceability.

## Core Components

- `utils/`
  - Typed config loading
  - Device selection across CUDA, MPS, and CPU
  - Reproducibility and checkpoint helpers

- `actuarial/`
  - Policy and mortality profile domain objects
  - Numerical Thiele reserve solver

- `data/`
  - Mortality ingestion
  - Synthetic portfolio generation
  - Dataset and dataloader creation

- `models/`
  - Base model contract
  - Configurable MLP backbone
  - PINN reserve model
  - Factory construction

- `losses/`
  - Data, PDE, boundary, and regularization losses
  - Composite weighted training objective

- `trainers/`
  - End-to-end training loop with experiment management

- `evaluators/`
  - Regression metrics
  - Autodiff-based sensitivity analytics

- `stress/`
  - Scenario shocks with report and chart generation

- `optimization/`
  - Gradient-based and SciPy optimization workflows

- `digital_twin/`
  - Forecasting, regime simulation, portfolio simulation, and scenario cloning

## Architectural Rationale

The classical solver and PINN coexist intentionally:

- The solver provides actuarial consistency and benchmark trajectories.
- The PINN enables fast differentiable inference.
- The digital twin layer converts reserve prediction into executive and operational analytics.

This structure keeps the platform useful for research, validation, and future productionization.
