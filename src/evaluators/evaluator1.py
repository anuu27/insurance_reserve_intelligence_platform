"""Model evaluation, sensitivity analysis, and elasticity computation."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from src.data.dataset import FEATURE_INDEX, FEATURE_SCALES
from src.visualization.sensitivity_plots import (
    plot_sensitivities,
    compute_and_plot_elasticities,
)


@dataclass(slots=True)
class EvaluationResult:
    mse: float
    mae: float
    rmse: float
    r2: float

_FD_DELTAS = {
    "sum_assured":5000.0,
}


class ReserveEvaluator:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    # ==========================================================
    # Utilities
    # ==========================================================

    def _denorm(
        self,
        z,
        target_mean,
        target_std,
        S,
    ):
        """
        Convert network output z back to reserve (£)

        V = (z * std + mean) * SumAssured
        """
        return (
            z * target_std
            + target_mean
        ) * S


    def _fd_V_high(
        self,
        features,
        feat_name,
        target_mean,
        target_std,
        S_base,
    ):
        """
        Finite-difference helper.

        Used ONLY for Sum Assured.
        """

        delta_norm = (
            _FD_DELTAS[feat_name]
            /
            FEATURE_SCALES[feat_name]
        )

        f_high = features.clone()

        f_high[:, FEATURE_INDEX[feat_name]] += delta_norm

        if feat_name == "sum_assured":

            S_use = (
                S_base
                +
                _FD_DELTAS[feat_name]
            )

        else:

            S_use = S_base

        with torch.no_grad():

            z_high = self.model(f_high)

        return self._denorm(
            z_high,
            target_mean,
            target_std,
            S_use,
        )


    def _autograd_gradient(
        self,
        features,
        target_mean,
        target_std,
        S,
    ):
        """
        Compute dV/d(all inputs)
        """

        inputs = (
            features
            .clone()
            .detach()
            .requires_grad_(True)
        )

        z = self.model(inputs)

        V = self._denorm(
            z,
            target_mean,
            target_std,
            S,
        )

        grads = torch.autograd.grad(
            outputs=V,
            inputs=inputs,
            grad_outputs=torch.ones_like(V),
            create_graph=True,
            retain_graph=True,
        )[0]

        return grads

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, dataloader: DataLoader) -> EvaluationResult:
        """Regression metrics in real £."""
        self.model.eval()
        y_true, y_pred = [], []
        for batch in dataloader:
            features = batch["features"].to(self.device)
            t_mean   = batch["target_mean"].to(self.device)
            t_std    = batch["target_std"].to(self.device)
            S        = batch["sum_assured_scale"].to(self.device)
            target_z = batch["target"].to(self.device)
            with torch.no_grad():
                pred_z = self.model(features)
            y_pred.append(self._denorm(pred_z,   t_mean, t_std, S).cpu().numpy())
            y_true.append(self._denorm(target_z, t_mean, t_std, S).cpu().numpy())

        y_true = np.vstack(y_true)
        y_pred = np.vstack(y_pred)
        mse  = float(np.mean((y_true - y_pred) ** 2))
        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(mse))
        var  = float(np.var(y_true))
        r2   = 1.0 - mse / var if var > 0 else 0.0
        return EvaluationResult(mse=mse, mae=mae, rmse=rmse, r2=r2)

    def compute_sensitivities(
        self,
        features: torch.Tensor,
        raw_features: torch.Tensor,
        target_mean: float,
        target_std: float,
        interest_std: float,
        premium_std: float,
    ):
        """
        Hybrid sensitivity computation.
    
        Autograd:
            dV/dr
            dV/dμ
            dV/dP
            d²V/dr²
    
        Finite Difference:
            dV/dS
        """
    
        self.model.eval()
    
        f = features.to(self.device).detach()
    
        S = (
            raw_features[:, FEATURE_INDEX["sum_assured"]]
            .to(self.device)
            .clamp(min=1.0)
            .unsqueeze(1)
        )
    
        # -----------------------------------------------------
        # Base prediction
        # -----------------------------------------------------
    
        with torch.no_grad():
    
            z_base = self.model(f)
    
            V_base = self._denorm(
                z_base,
                target_mean,
                target_std,
                S,
            )
    
        v_mean_real = float(V_base.mean().item())
    
        # -----------------------------------------------------
        # AUTOGRAD FIRST DERIVATIVES
        # -----------------------------------------------------
    
        grads = self._autograd_gradient(
            f,
            target_mean,
            target_std,
            S,
        )
        
        print("\nGradient diagnostics")

        print("Interest grad :", grads[:, FEATURE_INDEX["interest_rate"]].abs().mean().item())
        print("Premium grad  :", grads[:, FEATURE_INDEX["premium"]].abs().mean().item())
        print("Mortality grad:", grads[:, FEATURE_INDEX["mortality"]].abs().mean().item())
        print("SA grad       :", grads[:, FEATURE_INDEX["sum_assured"]].abs().mean().item())
        if interest_std < 1e-8:
            interest_std = 1.0
        
        if premium_std < 1e-8:
            premium_std = 1.0
        
        dV_dr = (
            grads[:, FEATURE_INDEX["interest_rate"]]
            / interest_std
        ).detach().cpu().numpy()
        
        dV_dmu = (
            grads[:, FEATURE_INDEX["mortality"]]
            / FEATURE_SCALES["mortality"]
        ).detach().cpu().numpy()
        
        dV_dP = (
            grads[:, FEATURE_INDEX["premium"]]
            / premium_std
        ).detach().cpu().numpy()
    
        # -----------------------------------------------------
        # SUM ASSURED
        # Keep finite difference
        # -----------------------------------------------------
    
        V_high = self._fd_V_high(
            f,
            "sum_assured",
            target_mean,
            target_std,
            S,
        )
    
        dV_dS = (
            (V_high - V_base)
            /
            _FD_DELTAS["sum_assured"]
        ).squeeze(1).cpu().numpy()
    
        # -----------------------------------------------------
        # SECOND DERIVATIVE wrt INTEREST RATE
        # -----------------------------------------------------
    
        inputs = (
            f.clone()
            .detach()
            .requires_grad_(True)
        )
    
        pred_z = self.model(inputs)
    
        grad_z = torch.autograd.grad(
            outputs=pred_z,
            inputs=inputs,
            grad_outputs=torch.ones_like(pred_z),
            create_graph=True,
        )[0]
    
        r_idx = FEATURE_INDEX["interest_rate"]
        
        grad2_z = torch.autograd.grad(
            outputs=grad_z[:, r_idx:r_idx + 1],
            inputs=inputs,
            grad_outputs=torch.ones_like(
                grad_z[:, r_idx:r_idx + 1]
            ),
            create_graph=False,
        )[0][:, r_idx].detach()
        
        d2V_dr2 = (
            grad2_z
            * target_std
            * S.squeeze(1).detach()
            / (interest_std ** 2)
        ).cpu().numpy()
            
        # -----------------------------------------------------
        # DataFrame
        # -----------------------------------------------------
        print("\nDEBUG")

        print("Mean reserve :", v_mean_real)
        
        print("Mean dV/dr  :", np.mean(dV_dr))
        print("Mean dV/dμ  :", np.mean(dV_dmu))
        print("Mean dV/dP  :", np.mean(dV_dP))
        print("Mean dV/dS  :", np.mean(dV_dS))
    
        sens_df = pd.DataFrame(
            {
                "dV_dr": dV_dr,
                "dV_dmu": dV_dmu,
                "dV_dP": dV_dP,
                "dV_dS": dV_dS,
                "d2V_dr2": d2V_dr2,
            }
        )
    
        return sens_df, v_mean_real

    def generate_sensitivity_report(
        self,
        features: torch.Tensor,
        output_path: str,
        raw_features: torch.Tensor | None = None,
        target_mean: float = 0.0,
        target_std: float = 1.0,
        interest_std: float = 1.0,
        premium_std: float = 1.0,
    ) -> pd.DataFrame:
    
        if raw_features is None:
            raw_features = features.clone()
            for name, idx in FEATURE_INDEX.items():
                raw_features[:, idx] *= FEATURE_SCALES[name]
    
        sens_df, v_mean_real = self.compute_sensitivities(
            features,
            raw_features,
            target_mean,
            target_std,
            interest_std,
            premium_std,
        )
    
        print("\n" + "=" * 80)
        print("SENSITIVITY DIAGNOSTICS")
        print("=" * 80)

        for col in ["dV_dr", "dV_dmu", "dV_dP", "dV_dS"]:

            vals = sens_df[col]

            print("\n" + "-" * 60)
            print(col)

            print(f"Mean      : {vals.mean():.6f}")
            print(f"Median    : {vals.median():.6f}")
            print(f"Min       : {vals.min():.6f}")
            print(f"Max       : {vals.max():.6f}")

            print(f"Positive% : {(vals > 0).mean()*100:.2f}")
            print(f"Negative% : {(vals < 0).mean()*100:.2f}")

        print("\n" + "=" * 80)
        print("ABSOLUTE MEAN MAGNITUDES")
        print("=" * 80)

        for col in ["dV_dr", "dV_dmu", "dV_dP", "dV_dS"]:
            print(
                f"{col:10s} : "
                f"{np.abs(sens_df[col]).mean():.6f}"
            )

        # ── 1. Sensitivity report (raw dV/dx, different units) ────────────────
        sens_df.to_csv(output_path, index=False)
        plot_sensitivities(sens_df, output_path.replace(".csv", ".png"))

        # ── 2. Elasticity report (dimensionless, all comparable) ─────────────
        elas_path = str(output_path).replace("sensitivity_report", "elasticity_report")
        elas_png  = elas_path.replace(".csv", ".png")
        premium_ratio_mean = float(
            raw_features[:, FEATURE_INDEX["premium"]]
            .mean()
            .item()
        )
        
        elas_summary = compute_and_plot_elasticities(
            sens_df,
            elas_png,
            v_mean_real,
            premium_ratio_mean,
        )
        elas_summary.to_csv(elas_path, index=False)
        print(f"  elasticity CSV   → {elas_path}")

        return sens_df