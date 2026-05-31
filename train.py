"""Train the ActuaryTwin PINN model.

Created: 2026-05-31
Purpose: Launch an end-to-end training run for the reserve PINN.
"""

from __future__ import annotations

from pathlib import Path

from src.pipeline import build_dataloaders, build_model
from src.trainers.trainer import PINNTrainer
from src.utils.config import ConfigLoader, ensure_directories
from src.utils.seed import set_seed


def main() -> None:
    """Run the training entrypoint.

    Business Interpretation:
        This script is the operational starting point for fitting a new reserve
        surrogate from configured actuarial assumptions.
    """
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    train_loader, validation_loader, _, _, _ = build_dataloaders(config)
    model = build_model(config)
    trainer = PINNTrainer(model=model, config=config)
    trainer.fit(train_loader=train_loader, validation_loader=validation_loader)


if __name__ == "__main__":
    main()
