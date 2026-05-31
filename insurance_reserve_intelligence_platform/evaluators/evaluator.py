"""Model evaluation and sensitivity analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from insurance_reserve_intelligence_platform.visualization.sensitivity_plots import plot_sensitivities


@dataclass(slots=True)
class EvaluationResult:
    """Summary of reserve model accuracy."""

    mse: float
    mae: float
    rmse: float
    r2: float


class ReserveEvaluator:
    """Evaluate reserve predictions and sensitivities."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        self.model = model.to(device)
        self.device = device

    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        """Compute standard regression metrics."""

        self.model.eval()
        targets: list[np.ndarray] = []
        predictions: list[np.ndarray] = []
        for batch in dataloader:
            features = batch["features"].to(self.device)
            target = batch["target"].to(self.device)
            with torch.no_grad():
                prediction = self.model(features)
            targets.append(target.cpu().numpy())
            predictions.append(prediction.cpu().numpy())

        y_true = np.vstack(targets)
        y_pred = np.vstack(predictions)
        mse = float(np.mean((y_true - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        variance = float(np.var(y_true))
        r2 = 1.0 - mse / variance if variance > 0.0 else 0.0
        return EvaluationResult(mse=mse, mae=mae, rmse=rmse, r2=r2)

    def compute_sensitivities(self, features: torch.Tensor) -> pd.DataFrame:
        """Compute first- and second-order sensitivities with autodiff."""

        self.model.eval()
        inputs = features.to(self.device).clone().detach().requires_grad_(True)
        reserves = self.model(inputs)
        first_order = torch.autograd.grad(
            outputs=reserves,
            inputs=inputs,
            grad_outputs=torch.ones_like(reserves),
            create_graph=True,
        )[0]
        second_order = torch.autograd.grad(
            outputs=first_order[:, 2:3],
            inputs=inputs,
            grad_outputs=torch.ones_like(first_order[:, 2:3]),
            create_graph=False,
        )[0]

        frame = pd.DataFrame(
            {
                "dV_dr": first_order[:, 2].detach().cpu().numpy(),
                "dV_dmu": first_order[:, 5].detach().cpu().numpy(),
                "dV_dP": first_order[:, 3].detach().cpu().numpy(),
                "dV_dS": first_order[:, 4].detach().cpu().numpy(),
                "d2V_dr2": second_order[:, 2].detach().cpu().numpy(),
            }
        )
        return frame

    def generate_sensitivity_report(self, features: torch.Tensor, output_path: str) -> pd.DataFrame:
        """Create a CSV sensitivity report and companion plot."""

        report = self.compute_sensitivities(features)
        report.to_csv(output_path, index=False)
        plot_sensitivities(report, output_path.replace(".csv", ".png"))
        return report
