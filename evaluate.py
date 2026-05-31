"""Evaluate the trained ActuaryTwin model.

Created: 2026-05-31
Purpose: Evaluate reserve accuracy and export sensitivity analytics.
"""

from __future__ import annotations

from pathlib import Path

import torch

from src.evaluators.evaluator import ReserveEvaluator
from src.pipeline import build_dataloaders, build_model
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed


def main() -> None:
    """Run the evaluation entrypoint.

    Business Interpretation:
        This script validates the trained reserve engine before it is used in
        downstream analysis such as optimization or stress testing.
    """
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    _, _, test_loader, test_dataset, _ = build_dataloaders(config)
    device_manager = DeviceManager(prefer_mixed_precision=False)
    model = build_model(config)

    checkpoint_path = Path(config.paths.checkpoints_dir) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = CheckpointManager(config.paths.checkpoints_dir).load(checkpoint_path, map_location=device_manager.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    evaluator = ReserveEvaluator(model=model, device=device_manager.device)
    result = evaluator.evaluate(test_loader)
    print(f"MSE={result.mse:.6f} MAE={result.mae:.6f} RMSE={result.rmse:.6f} R2={result.r2:.6f}")

    sample_features = torch.stack([test_dataset[0]["features"], test_dataset[1]["features"]])
    output_csv = Path(config.paths.reports_dir) / "sensitivity_report.csv"
    evaluator.generate_sensitivity_report(sample_features, str(output_csv))
    print(f"Sensitivity report written to {output_csv}")


if __name__ == "__main__":
    main()
