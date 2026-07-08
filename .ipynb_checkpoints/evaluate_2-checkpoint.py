"""Evaluate the trained ActuaryTwin model.

Created: 2026-05-31  Revised: 2026-06-15
Purpose: Evaluate reserve accuracy and run the separated sensitivity pipeline.

Sensitivity pipeline (Option 3):
    One reference policy → fixed feature vector → perturb ONE feature at a time
    → PINN prediction → finite difference → true partial derivative

Premium is stored as a RATIO (premium / sum_assured) in training.
Interest rate and premium use z-score normalisation from the training set.
All other features use simple scaling by FEATURE_SCALES.
"""
from __future__ import annotations
from pathlib import Path
import torch
import copy
import numpy as np 
import pandas as pd
from src.actuarial.actuarial_solver import ThieleSolver
from src.visualization.reserve_plots import plot_reserve_trajectory
from src.evaluators.evaluator import ReserveEvaluator
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES


def build_reference_feature_vector(
    age: int          = 40,
    interest_rate: float = 0.04,
    premium_ratio: float = 0.0032,   # premium / sum_assured  (ratio, not £)
    sum_assured: float   = 500_000.0,
    mortality: float     = 0.0015,
    policy_time: float   = 5.0,
    # Training normalisation statistics (must match dataset)
    interest_mean: float = 0.0,
    interest_std: float  = 1.0,
    premium_mean: float  = 0.0,
    premium_std: float   = 1.0,
    n: int               = 1,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a fixed reference feature vector for the separated sensitivity pipeline.

    Raw feature order matches FEATURE_INDEX:
        [time, age, interest_rate, premium_ratio, sum_assured, mortality]

    Premium is stored as a RATIO (consistent with dataset._build_records).
    Interest rate and premium use z-score normalisation (consistent with
    dataset.__getitem__). All other features use simple scale division.

    Args:
        interest_mean / interest_std: from test_dataset.interest_mean/std
        premium_mean  / premium_std:  from test_dataset.premium_mean/std
        n: copies of the reference vector (sensitivity averaged across n)

    Returns:
        (normalised_features [n,6], raw_features [n,6])
    """
    # ── raw features in their stored units ────────────────────────────────
    # premium_ratio is already the ratio — NOT converted to £
    raw = torch.tensor([[
        policy_time,        # time in years
        float(age),         # age in years
        interest_rate,      # e.g. 0.04
        premium_ratio,      # ratio: premium/SA  e.g. 0.0032
        sum_assured,        # £ e.g. 500000
        mortality,          # intensity e.g. 0.0015/yr
    ]], dtype=torch.float32).repeat(n, 1)

    # ── normalise exactly as dataset.__getitem__ does ─────────────────────
    norm = raw.clone()

    # simple scale
    norm[:, FEATURE_INDEX["time"]]        /= FEATURE_SCALES["time"]
    norm[:, FEATURE_INDEX["age"]]         /= FEATURE_SCALES["age"]
    norm[:, FEATURE_INDEX["sum_assured"]] /= FEATURE_SCALES["sum_assured"]
    norm[:, FEATURE_INDEX["mortality"]]   /= FEATURE_SCALES["mortality"]

    # z-score (must use training-set mean/std, not FEATURE_SCALES)
    norm[:, FEATURE_INDEX["interest_rate"]] = (
        raw[:, FEATURE_INDEX["interest_rate"]] - interest_mean
    ) / interest_std

    norm[:, FEATURE_INDEX["premium"]] = (
        raw[:, FEATURE_INDEX["premium"]] - premium_mean
    ) / premium_std

    print("\n" + "=" * 60)
    print("REFERENCE POLICY  (sensitivity baseline)")
    print("=" * 60)
    print(f"  Age            : {age} yrs")
    print(f"  Interest rate  : {interest_rate*100:.2f}%")
    print(f"  Premium ratio  : {premium_ratio:.4f}  (£{premium_ratio*sum_assured:,.0f}/yr)")
    print(f"  Sum assured    : £{sum_assured:,.0f}")
    print(f"  Mortality      : {mortality:.4f}/yr")
    print(f"  Policy time    : {policy_time:.1f} yrs")
    print("-" * 60)
    print(f"  interest_mean  : {interest_mean:.6f}  interest_std : {interest_std:.6f}")
    print(f"  premium_mean   : {premium_mean:.6f}   premium_std  : {premium_std:.6f}")
    print("-" * 60)
    print("  Normalised vector (what the model sees):")
    print(f"    time     : {norm[0, FEATURE_INDEX['time']].item():.4f}")
    print(f"    age      : {norm[0, FEATURE_INDEX['age']].item():.4f}")
    print(f"    interest : {norm[0, FEATURE_INDEX['interest_rate']].item():.4f}  (z-score)")
    print(f"    premium  : {norm[0, FEATURE_INDEX['premium']].item():.4f}  (z-score)")
    print(f"    SA       : {norm[0, FEATURE_INDEX['sum_assured']].item():.4f}")
    print(f"    mortality: {norm[0, FEATURE_INDEX['mortality']].item():.4f}")
    print("=" * 60)
    print("Each sensitivity perturbs ONLY ONE raw value.")
    print("All others stay fixed at the reference above.")
    print("=" * 60)

    return norm.to(device), raw.to(device)

def compute_classical_sensitivities(
    config,
    reference_policy,
):
    """
    Compute finite-difference sensitivities directly from the
    classical Thiele solver.
    """

    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )

    deltas = {
        "interest_rate":0.0001,
        "mortality":1e-5,
        "premium":1e-5,
        "sum_assured":100.0,
    }

    results = {}

    for feature, delta in deltas.items():

        policy_hi = copy.deepcopy(reference_policy)
        policy_lo = copy.deepcopy(reference_policy)

        if feature == "interest_rate":
            policy_hi.interest_rate += delta
            policy_lo.interest_rate -= delta

        elif feature == "premium":

            ratio = policy_hi.premium / policy_hi.sum_assured

            ratio_hi = ratio + delta
            ratio_lo = ratio - delta

            policy_hi.premium = ratio_hi * policy_hi.sum_assured
            policy_lo.premium = ratio_lo * policy_lo.sum_assured

        elif feature == "sum_assured":

            policy_hi.sum_assured += delta
            policy_lo.sum_assured -= delta

        elif feature == "mortality":

            policy_hi.mortality_profile.intensities += delta
            policy_lo.mortality_profile.intensities -= delta

        traj_hi = solver.solve(
            policy_hi,
            num_steps=config.data.time_steps,
        )

        traj_lo = solver.solve(
            policy_lo,
            num_steps=config.data.time_steps,
        )

        valuation_time = 5.0

        idx = np.argmin(np.abs(traj_hi.times - valuation_time))
        
        reserve_hi = traj_hi.reserves[idx]
        reserve_lo = traj_lo.reserves[idx]

        derivative = (
            reserve_hi
            -
            reserve_lo
        ) / (2 * delta)

        results[feature] = derivative

    return results


def main() -> None:
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    _, _, test_loader, test_dataset, test_policies = build_dataloaders(config)

    device_manager = DeviceManager(
        preferred_device=config.trainer.device,
        prefer_mixed_precision=False,
    )
    device = device_manager.device

    model = build_model(config)
    checkpoint_path = Path(config.paths.checkpoints_dir) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = CheckpointManager(config.paths.checkpoints_dir).load(
            checkpoint_path, map_location=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = ReserveEvaluator(model=model, device=device)

    # ── 1. Regression accuracy on the test set ────────────────────────────
    result = evaluator.evaluate(test_loader)
    print(
        f"\nrun={config.trainer.run_name}  device={device_manager.summary()}"
        f"\nMSE={result.mse:.4f}  MAE={result.mae:.4f}  "
        f"RMSE={result.rmse:.4f}  R²={result.r2:.4f}"
    )

    # ── 2. Separated sensitivity pipeline ────────────────────────────────
    # Pass training-set normalisation statistics so the reference vector
    # is normalised IDENTICALLY to how the model was trained.
    ref_norm, ref_raw = build_reference_feature_vector(
        age=40,
        interest_rate=0.04,
        premium_ratio=0.0032,       # ratio, not raw £ — consistent with dataset
        sum_assured=500_000.0,
        mortality=0.0015,
        policy_time=5.0,
        interest_mean=test_dataset.interest_mean,
        interest_std=test_dataset.interest_std,
        premium_mean=test_dataset.premium_mean,
        premium_std=test_dataset.premium_std,
        n=1,
        device=device,
    )

    output_csv = Path(config.paths.reports_dir) / "sensitivity_report.csv"
    evaluator.generate_sensitivity_report(
        ref_norm,
        str(output_csv),
        ref_raw_features=ref_raw,
        target_mean=test_dataset.target_mean,
        target_std=test_dataset.target_std,
        interest_mean=test_dataset.interest_mean,
        interest_std=test_dataset.interest_std,
        premium_mean=test_dataset.premium_mean,
        premium_std=test_dataset.premium_std,
    )
    print(f"\nSensitivity report written to {output_csv}")
    reference_policy = copy.deepcopy(test_policies[0])

    reference_policy.age = 40
    reference_policy.interest_rate = 0.04
    reference_policy.sum_assured = 500000
    
    reference_policy.premium = (
        0.0032
        * reference_policy.sum_assured
    )
    
    reference_policy.mortality_profile.intensities[:] = 0.0015
    
    classical = compute_classical_sensitivities(
        config,
        reference_policy,
    )

    sens_df = pd.read_csv(output_csv)
    
    pinn = {
    
        "interest_rate":
            sens_df["dV_dr"].iloc[0],
    
        "mortality":
            sens_df["dV_dmu"].iloc[0],
    
        "premium":
            sens_df["dV_dP"].iloc[0],
    
        "sum_assured":
            sens_df["dV_dS"].iloc[0],
    
    }

    print()
    
    print("="*70)
    
    print("PINN vs CLASSICAL SENSITIVITIES")
    
    print("="*70)
    
    for key in pinn:
    
        print(
            f"{key:15s}"
            f"PINN={pinn[key]:14.6f}"
            f"   Classical={classical[key]:14.6f}"
        )
    
    print("="*70)

    # ── 3. Reserve trajectory plot ────────────────────────────────────────
    policy = test_policies[0]
    solver = ThieleSolver(
        method=config.solver.method,
        integration_step=config.solver.integration_step,
        rtol=config.solver.rtol,
        atol=config.solver.atol,
    )
    trajectory = solver.solve(policy=policy, num_steps=config.data.time_steps)
    trajectory_df = pd.DataFrame({
        "time":    trajectory.times,
        "reserve": trajectory.reserves,
    })
    reserve_plot_path = Path(config.paths.reports_dir) / "reserve_trajectory.png"
    plot_reserve_trajectory(trajectory_df, str(reserve_plot_path))
    print(f"Reserve plot written to {reserve_plot_path}")


if __name__ == "__main__":
    main()