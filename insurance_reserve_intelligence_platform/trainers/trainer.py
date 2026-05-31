"""Training loop implementations."""

from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    SummaryWriter = None  # type: ignore[assignment]

from insurance_reserve_intelligence_platform.losses.total_loss import LossBreakdown, TotalLoss
from insurance_reserve_intelligence_platform.utils.checkpoint import CheckpointManager
from insurance_reserve_intelligence_platform.utils.config import ExperimentConfig
from insurance_reserve_intelligence_platform.utils.device import DeviceManager
from insurance_reserve_intelligence_platform.utils.logger import configure_logger


@dataclass(slots=True)
class EpochMetrics:
    """Aggregate metrics for a training or validation epoch."""

    total_loss: float
    data_loss: float
    pde_loss: float
    boundary_loss: float
    regularization_loss: float


class BaseTrainer(ABC):
    """Abstract model trainer."""

    @abstractmethod
    def fit(self, train_loader: DataLoader, validation_loader: DataLoader) -> dict[str, list[float]]:
        """Train a model and return history."""


class PINNTrainer(BaseTrainer):
    """Full-featured trainer for the reserve PINN."""

    def __init__(self, model: nn.Module, config: ExperimentConfig, device_manager: DeviceManager | None = None) -> None:
        self.config = config
        self.device_manager = device_manager or DeviceManager(prefer_mixed_precision=config.trainer.mixed_precision)
        self.model = model.to(self.device_manager.device)
        self.loss_fn = TotalLoss(config.losses)
        self.optimizer = Adam(
            self.model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.trainer.scheduler_factor,
            patience=config.trainer.scheduler_patience,
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.device_manager.mixed_precision)
        self.checkpoints = CheckpointManager(config.paths.checkpoints_dir)
        self.writer = self._create_writer(config.paths.tensorboard_dir)
        self.logger = configure_logger(
            name="PINNTrainer",
            log_file=str(Path(config.paths.logs_dir) / "training.log"),
        )
        self.csv_log_path = Path(config.paths.logs_dir) / "training_metrics.csv"
        self.best_validation_loss = float("inf")
        self.epochs_without_improvement = 0
        self.start_epoch = 0
        self.history: dict[str, list[float]] = {"train_total_loss": [], "validation_total_loss": []}
        self._initialize_csv_log()
        if config.trainer.resume_from:
            self.resume(config.trainer.resume_from)

    @staticmethod
    def _create_writer(log_dir: str):
        if SummaryWriter is None:
            return _NullSummaryWriter()
        return SummaryWriter(log_dir=log_dir)

    def _initialize_csv_log(self) -> None:
        self.csv_log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_log_path.exists():
            with self.csv_log_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "epoch",
                        "split",
                        "total_loss",
                        "data_loss",
                        "pde_loss",
                        "boundary_loss",
                        "regularization_loss",
                    ],
                )
                writer.writeheader()

    def _log_csv(self, epoch: int, split: str, metrics: EpochMetrics) -> None:
        with self.csv_log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "split", *asdict(metrics).keys()])
            writer.writerow({"epoch": epoch, "split": split, **asdict(metrics)})

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        features = batch["features"].to(self.device_manager.device)
        return {
            "features": features.requires_grad_(True),
            "target": batch["target"].to(self.device_manager.device),
            "term": batch["term"].to(self.device_manager.device),
        }

    @staticmethod
    def _empty_loss_breakdown(device: torch.device) -> LossBreakdown:
        zero = torch.tensor(0.0, device=device)
        return LossBreakdown(total=zero, data=zero, pde=zero, boundary=zero, regularization=zero, residual=zero.unsqueeze(0))

    def _aggregate_epoch(self, loss_values: list[LossBreakdown]) -> EpochMetrics:
        if not loss_values:
            raise ValueError("No losses collected for the epoch.")
        return EpochMetrics(
            total_loss=float(torch.stack([item.total.detach() for item in loss_values]).mean().item()),
            data_loss=float(torch.stack([item.data.detach() for item in loss_values]).mean().item()),
            pde_loss=float(torch.stack([item.pde.detach() for item in loss_values]).mean().item()),
            boundary_loss=float(torch.stack([item.boundary.detach() for item in loss_values]).mean().item()),
            regularization_loss=float(torch.stack([item.regularization.detach() for item in loss_values]).mean().item()),
        )

    def _step(self, batch: dict[str, torch.Tensor], training: bool) -> LossBreakdown:
        prepared = self._move_batch(batch)
        if training:
            self.optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            with self.device_manager.autocast_context():
                loss_breakdown = self.loss_fn(
                    model=self.model,
                    features=prepared["features"],
                    targets=prepared["target"],
                    terms=prepared["term"],
                )
            if training:
                self.scaler.scale(loss_breakdown.total).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        return loss_breakdown

    def _run_epoch(self, loader: DataLoader, training: bool) -> EpochMetrics:
        self.model.train(mode=training)
        losses: list[LossBreakdown] = []
        for batch in loader:
            losses.append(self._step(batch, training=training))
        return self._aggregate_epoch(losses)

    def _save_checkpoint(self, epoch: int, validation_metrics: EpochMetrics) -> None:
        state: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "best_validation_loss": self.best_validation_loss,
            "validation_metrics": asdict(validation_metrics),
            "config": self.config,
        }
        self.checkpoints.save(f"epoch_{epoch:03d}.pt", state)
        if validation_metrics.total_loss <= self.best_validation_loss:
            self.checkpoints.save("best_model.pt", state)

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""

        checkpoint = self.checkpoints.load(checkpoint_path, map_location=self.device_manager.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        self.best_validation_loss = float(checkpoint["best_validation_loss"])
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.logger.info("Resumed from checkpoint %s at epoch %s", checkpoint_path, self.start_epoch)

    def fit(self, train_loader: DataLoader, validation_loader: DataLoader) -> dict[str, list[float]]:
        """Train the model and return the loss history."""

        for epoch in range(self.start_epoch, self.config.trainer.epochs):
            train_metrics = self._run_epoch(train_loader, training=True)
            validation_metrics = self._run_epoch(validation_loader, training=False)
            self.scheduler.step(validation_metrics.total_loss)

            self.history["train_total_loss"].append(train_metrics.total_loss)
            self.history["validation_total_loss"].append(validation_metrics.total_loss)
            self.writer.add_scalar("loss/train_total", train_metrics.total_loss, epoch)
            self.writer.add_scalar("loss/validation_total", validation_metrics.total_loss, epoch)
            self.writer.add_scalar("loss/train_pde", train_metrics.pde_loss, epoch)
            self.writer.add_scalar("loss/validation_pde", validation_metrics.pde_loss, epoch)
            self._log_csv(epoch, "train", train_metrics)
            self._log_csv(epoch, "validation", validation_metrics)

            self.logger.info(
                "Epoch %s | train %.6f | validation %.6f",
                epoch,
                train_metrics.total_loss,
                validation_metrics.total_loss,
            )

            if epoch % self.config.trainer.checkpoint_every == 0:
                self._save_checkpoint(epoch, validation_metrics)

            if validation_metrics.total_loss < self.best_validation_loss:
                self.best_validation_loss = validation_metrics.total_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, validation_metrics)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.config.trainer.early_stopping_patience:
                    self.logger.info("Early stopping triggered at epoch %s", epoch)
                    break

        self.writer.flush()
        self.writer.close()
        return self.history


class _NullSummaryWriter:
    """Fallback writer used when tensorboard is unavailable."""

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        del tag, scalar_value, global_step

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None
