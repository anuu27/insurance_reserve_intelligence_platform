"""Train the ActuaryTwin PINN model."""

from __future__ import annotations

from pathlib import Path

from insurance_reserve_intelligence_platform.pipeline import build_dataloaders, build_model
from insurance_reserve_intelligence_platform.trainers.trainer import PINNTrainer
from insurance_reserve_intelligence_platform.utils.config import ConfigLoader, ensure_directories
from insurance_reserve_intelligence_platform.utils.seed import set_seed


def main() -> None:
    config = ConfigLoader.load(Path("configs/config.yaml"))
    ensure_directories(config)
    set_seed(config.seed)

    train_loader, validation_loader, _, _, _ = build_dataloaders(config)
    model = build_model(config)
    trainer = PINNTrainer(model=model, config=config)
    trainer.fit(train_loader=train_loader, validation_loader=validation_loader)


if __name__ == "__main__":
    main()
