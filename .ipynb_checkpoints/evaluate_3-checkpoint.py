"""Evaluate the trained ActuaryTwin model."""
from __future__ import annotations
from pathlib import Path
import torch
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
    age: int             = 40,
    interest_rate: float = 0.04,
    premium_ratio: float = 0.0032,
    sum_assured: float   = 500_000.0,
    mortality: float     = 0.0015,
    policy_time: float   = 5.0,
    interest_mean: float = 0.0,
    interest_std: float  = 1.0,
    premium_mean: float  = 0.0,
    premium_std: float   = 1.0,
    n: int               = 100,          # ← 100 copies, NOT 1
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed reference feature vector for the separated sensitivity pipeline."""

    raw = torch.tensor([[
        policy_time,
        float(age),
        interest_rate,
        premium_ratio,   # ratio, not £
        sum_assured,
        mortality,
    ]], dtype=torch.float32).repeat(n, 1)

    norm = raw.clone()

    # simple scale
    norm[:, FEATURE_INDEX["time"]]        /= FEATURE_SCALES["time"]
    norm[:, FEATURE_INDEX["age"]]         /= FEATURE_SCALES["age"]
    norm[:, FEATURE_INDEX["sum_assured"]] /= FEATURE_SCALES["sum_assured"]
    norm[:, FEATURE_INDEX["mortality"]]   /= FEATURE_SCALES["mortality"]

    # z-score — must use training-set statistics
    norm[:, FEATURE_INDEX["interest_rate"]] = (
        raw[:, FEATURE_INDEX["interest_rate"]] - interest_mean
    ) / interest_std

    norm[:, FEATURE_INDEX["premium"]] = (
        raw[:, FEATURE_INDEX["premium"]] - premium_mean
    ) / premium_std

    print("\n" + "=" * 60)
    print("REFERENCE POLICY  (sensitivity baseline)")
    print("=" * 60)
    print(f"  N copies       : {n}")
    print(f"  Age            : {age} yrs")
    print(f"  Interest rate  : {interest_rate*100:.2f}%")
    print(f"  Premium ratio  : {premium_ratio:.4f}  (£{premium_ratio*sum_assured:,.0f}/yr)")
    print(f"  Sum assured    : £{sum_assured:,.0f}")
    print(f"  Mortality      : {mortality:.4f}/yr")
    print(f"  Policy time    : {policy_time:.1f} yrs")
    print("-" * 60)
    print(f"  interest_mean  : {interest_mean:.6f}   interest_std : {interest_std:.6f}")
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

    # 1. Regression accuracy
    result = evaluator.evaluate(test_loader)
    print(
        f"\nrun={config.trainer.run_name}  device={device_manager.summary()}"
        f"\nMSE={result.mse:.4f}  MAE={result.mae:.4f}  "
        f"RMSE={result.rmse:.4f}  R²={result.r2:.4f}"
    )

    # 2. Separated sensitivity pipeline
    ref_norm, ref_raw = build_reference_feature_vector(
        age=40,
        interest_rate=0.04,
        premium_ratio=0.0032,
        sum_assured=500_000.0,
        mortality=0.0015,
        policy_time=5.0,
        interest_mean=test_dataset.interest_mean,
        interest_std=test_dataset.interest_std,
        premium_mean=test_dataset.premium_mean,
        premium_std=test_dataset.premium_std,
        n=100,                               # ← 100 copies
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

    # 3. Reserve trajectory plot
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