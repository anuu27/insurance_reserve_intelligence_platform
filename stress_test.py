"""Run stress testing for the ActuaryTwin platform."""

from __future__ import annotations

from pathlib import Path

from src.pipeline import build_dataloaders, build_model
from src.stress.stress_tester import StressTester
from src.utils.checkpoint import CheckpointManager
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.device import DeviceManager
from src.utils.seed import set_seed


def main() -> None:
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    _, _, _, _, test_policies = build_dataloaders(config)
    device_manager = DeviceManager(prefer_mixed_precision=False)
    model = build_model(config)
    checkpoint_path = Path(config.paths.checkpoints_dir) / "best_model.pt"
    if checkpoint_path.exists():
        checkpoint = CheckpointManager(config.paths.checkpoints_dir).load(checkpoint_path, map_location=device_manager.device)
        model.load_state_dict(checkpoint["model_state_dict"])

    tester = StressTester(model=model, device=device_manager.device, config=config.stress)
    results = tester.run_all(test_policies[:10], output_dir=config.paths.reports_dir)
    print(f"Generated {len(results)} scenario reports in {config.paths.reports_dir}")


if __name__ == "__main__":
    main()
