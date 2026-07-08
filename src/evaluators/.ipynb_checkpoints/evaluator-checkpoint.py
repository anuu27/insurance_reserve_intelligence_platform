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

'''
"""Model evaluation and sensitivity analysis — Option 3 separated pipeline."""
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
)'''


@dataclass(slots=True)
class EvaluationResult:
    mse: float
    mae: float
    rmse: float
    r2: float


class ReserveEvaluator:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    # ── utilities ─────────────────────────────────────────────────────────

    def _denorm(self, z, target_mean, target_std, S):
        """z → real £:  V = (z·σ + μ) · S"""
        return (z * target_std + target_mean) * S

    def _predict_V(self, features, S, target_mean, target_std):
        """Run model on normalised features, denormalise to real £."""
        with torch.no_grad():
            z = self.model(features)
        return self._denorm(z, target_mean, target_std, S)

    def _normalise_delta(
        self,
        feat_name: str,
        delta_real: float,
        interest_std: float,
        premium_std: float,
    ) -> float:
        """Convert a real-unit perturbation to normalised-space perturbation.

        Matches dataset.__getitem__ normalisation exactly:
          - interest_rate, premium  → z-score  → delta / std
          - all others              → scale     → delta / FEATURE_SCALES
        """
        if feat_name == "interest_rate":
            return delta_real / interest_std
        elif feat_name == "premium":
            return delta_real / premium_std
        else:
            return delta_real / FEATURE_SCALES[feat_name]

    # ── evaluation ────────────────────────────────────────────────────────

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

    # ── separated sensitivity pipeline ────────────────────────────────────

    def compute_sensitivities(
        self,
        ref_features: torch.Tensor,       # already normalised [n, 6]
        ref_raw_features: torch.Tensor,   # raw units [n, 6]
        target_mean: float,
        target_std: float,
        interest_mean: float,
        interest_std: float,
        premium_mean: float,
        premium_std: float,
    ) -> tuple[pd.DataFrame, float]:
        """Compute true partial derivatives via the separated sensitivity pipeline.

        For each variable, ONE feature is perturbed while ALL others stay
        exactly at their reference values. This gives:

            dV/dr  = ΔV when only r changes      (premium_ratio fixed)
            dV/dμ  = ΔV when only μ changes
            dV/dP  = ΔV when only premium_ratio changes  (r fixed)
            dV/dS  = ΔV when only S changes

        Perturbation normalisation:
            interest_rate, premium  → delta / std       (z-score features)
            time, age, SA, mortality → delta / SCALE    (scaled features)

        This matches dataset.__getitem__ exactly so the model receives
        inputs in the same space it was trained on.
        """
        self.model.eval()

        f   = ref_features.to(self.device)
        raw = ref_raw_features.to(self.device)
        S   = raw[:, FEATURE_INDEX["sum_assured"]].clamp(min=1.0).unsqueeze(1)

        # ── base prediction ───────────────────────────────────────────────
        V_base      = self._predict_V(f, S, target_mean, target_std)
        v_mean_real = float(V_base.mean().item())

        print(f"\nReference portfolio: N={f.shape[0]}")
        print(f"Mean base reserve:   £{v_mean_real:,.2f}")

        # ── perturbation config ───────────────────────────────────────────
        # delta_real: shift in the RAW (real-unit) space
        # The normalised delta is computed by _normalise_delta() which
        # applies the same transform used during training.
        perturbations = {
            "interest_rate":1e-4,
            "mortality":1e-5,
            "premium":1e-5,
            "sum_assured":100.0,
        }

        results = {}

        for feat_name, delta_real in perturbations.items():
            idx = FEATURE_INDEX[feat_name]

            # Convert real-unit delta → normalised-space delta
            delta_norm = self._normalise_delta(
                feat_name, delta_real, interest_std, premium_std
            )

            # Perturb ONLY this feature
            f_high = f.clone()
            f_low  = f.clone()
            
            f_high[:, idx] += delta_norm
            f_low[:, idx]  -= delta_norm
            
            if feat_name == "sum_assured":
                S_high = S + delta_real
                S_low  = S - delta_real
            else:
                S_high = S
                S_low  = S
            
            V_high = self._predict_V(f_high, S_high, target_mean, target_std)
            V_low  = self._predict_V(f_low, S_low, target_mean, target_std)
            
            dV = (
                (V_high - V_low)
                / (2.0 * delta_real)
            ).squeeze(1).cpu().numpy()
            results[feat_name] = dV

            sign = "✓" if (
                (feat_name == "interest_rate" and np.mean(dV) < 0) or
                (feat_name != "interest_rate" and np.mean(dV) > 0)
            ) else "✗"
            print(
                f"  {sign} d(V)/d({feat_name:17s})  "
                f"Δ={delta_real:+.6f}  "
                f"mean={np.mean(dV):+.4f}  "
                f"pos={( dV>0).mean()*100:.0f}%"
            )

        # ── d²V/dr² via central FD ────────────────────────────────────────
        # Central FD is more accurate than autograd in z-space
        dr_real = perturbations["interest_rate"]
        dr_norm = self._normalise_delta("interest_rate", dr_real, interest_std, premium_std)

        f_r_hi = f.clone(); f_r_hi[:, FEATURE_INDEX["interest_rate"]] += dr_norm
        f_r_lo = f.clone(); f_r_lo[:, FEATURE_INDEX["interest_rate"]] -= dr_norm

        V_r_hi = self._predict_V(f_r_hi, S, target_mean, target_std)
        V_r_lo = self._predict_V(f_r_lo, S, target_mean, target_std)

        d2V_dr2 = (
            (V_r_hi - 2 * V_base + V_r_lo) / (dr_real ** 2)
        ).squeeze(1).cpu().numpy()

        # ── diagnostics ───────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print("SENSITIVITY SUMMARY")
        print("=" * 60)
        col_map = {
            "interest_rate": ("dV_dr",  "neg"),
            "mortality":     ("dV_dmu", "pos"),
            "premium":       ("dV_dP",  "pos"),
            "sum_assured":   ("dV_dS",  "pos"),
        }
        for feat, (col, expected) in col_map.items():
            vals = results[feat]
            sign_pct = (vals < 0).mean()*100 if expected == "neg" else (vals > 0).mean()*100
            print(
                f"  {col:10s}  mean={vals.mean():+.4f}  "
                f"correct_sign={sign_pct:.0f}%"
            )
        print("=" * 60)

        sens_df = pd.DataFrame({
            "dV_dr":   results["interest_rate"],
            "dV_dmu":  results["mortality"],
            "dV_dP":   results["premium"],
            "dV_dS":   results["sum_assured"],
            "d2V_dr2": d2V_dr2,
        })

        return sens_df, v_mean_real

    # ── report ────────────────────────────────────────────────────────────

    def generate_sensitivity_report(
        self,
        ref_features: torch.Tensor,
        output_path: str,
        ref_raw_features: torch.Tensor | None = None,
        target_mean: float = 0.0,
        target_std: float  = 1.0,
        interest_mean: float = 0.0,
        interest_std: float  = 1.0,
        premium_mean: float  = 0.0,
        premium_std: float   = 1.0,
    ) -> pd.DataFrame:
        """Run the separated sensitivity pipeline and write CSV + plots."""
        if ref_raw_features is None:
            raise ValueError(
                "ref_raw_features is required. "
                "Pass the raw (un-normalised) reference feature tensor."
            )

        sens_df, v_mean_real = self.compute_sensitivities(
            ref_features,
            ref_raw_features,
            target_mean,
            target_std,
            interest_mean,
            interest_std,
            premium_mean,
            premium_std,
        )

        # Sensitivity plot (raw units, different per variable)
        sens_df.to_csv(output_path, index=False)
        plot_sensitivities(sens_df, output_path.replace(".csv", ".png"))

        # Elasticity plot (dimensionless, all comparable)
        # x_typical for premium is the mean premium_ratio from the reference set
        premium_ratio_typical = float(
            ref_raw_features[0, FEATURE_INDEX["premium"]].item()
        )
        elas_path = str(output_path).replace("sensitivity_report", "elasticity_report")
        elas_png  = elas_path.replace(".csv", ".png")
        elas_summary = compute_and_plot_elasticities(
            sens_df, elas_png, v_mean_real, premium_ratio_typical,
        )
        elas_summary.to_csv(elas_path, index=False)
        print(f"  elasticity CSV → {elas_path}")

        return sens_df

"""Model evaluation and sensitivity analysis — Option 3 separated pipeline."""


'''
@dataclass(slots=True)
class EvaluationResult:
    mse: float
    mae: float
    rmse: float
    r2: float


# ── perturbation sizes in RAW (real-unit) space ───────────────────────────────
# These are meaningful economic shifts — large enough to get a clean signal,
# small enough to stay local.
# DO NOT change these to tiny values like 1e-4 or 1e-5 — that causes
# numerical noise to dominate and produces nonsensical magnitudes.
_PERTURBATIONS = {
    "interest_rate": 0.01,      # 4% → 5%  (1 percentage point)
    "mortality":     0.0005,    # 0.0015 → 0.0020  (meaningful mortality shift)
    "premium":       0.0003,    # ratio: 0.0032 → 0.0035
    "sum_assured":   5_000.0,   # £500k → £505k
}


class ReserveEvaluator:

    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

    # ── utilities ─────────────────────────────────────────────────────────

    def _denorm(self, z, target_mean, target_std, S):
        """z → real £:  V = (z·σ + μ) · S"""
        return (z * target_std + target_mean) * S

    def _predict_V(self, features, S, target_mean, target_std):
        """Run model on normalised features, denormalise to real £."""
        with torch.no_grad():
            z = self.model(features)
        return self._denorm(z, target_mean, target_std, S)

    def _normalise_delta(
        self,
        feat_name: str,
        delta_real: float,
        interest_std: float,
        premium_std: float,
    ) -> float:
        """Convert real-unit perturbation → normalised-space perturbation.

        Matches dataset.__getitem__ exactly:
          interest_rate, premium  → z-score  → delta / std
          everything else         → scale     → delta / FEATURE_SCALES
        """
        if feat_name == "interest_rate":
            return delta_real / interest_std
        elif feat_name == "premium":
            return delta_real / premium_std
        else:
            return delta_real / FEATURE_SCALES[feat_name]

    # ── evaluation ────────────────────────────────────────────────────────

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

    # ── separated sensitivity pipeline ────────────────────────────────────

    def compute_sensitivities(
        self,
        ref_features: torch.Tensor,
        ref_raw_features: torch.Tensor,
        target_mean: float,
        target_std: float,
        interest_mean: float,
        interest_std: float,
        premium_mean: float,
        premium_std: float,
    ) -> tuple[pd.DataFrame, float]:
        """True partial derivatives via the separated sensitivity pipeline.

        For each variable, ONE feature is perturbed in normalised space
        while ALL others stay at their reference values.

        Perturbation sizes from _PERTURBATIONS (real units).
        Normalisation via _normalise_delta() matches training exactly.
        """
        self.model.eval()

        f   = ref_features.to(self.device)
        raw = ref_raw_features.to(self.device)
        S   = raw[:, FEATURE_INDEX["sum_assured"]].clamp(min=1.0).unsqueeze(1)

        # base prediction
        V_base      = self._predict_V(f, S, target_mean, target_std)
        v_mean_real = float(V_base.mean().item())

        print(f"\nReference portfolio: N={f.shape[0]}")
        print(f"Mean base reserve:   £{v_mean_real:,.2f}")

        results = {}

        for feat_name, delta_real in _PERTURBATIONS.items():

            delta_norm = self._normalise_delta(
                feat_name, delta_real, interest_std, premium_std
            )

            # perturb ONLY this feature
            f_high = f.clone()
            f_high[:, FEATURE_INDEX[feat_name]] += delta_norm

            # S changes only when SA is perturbed
            S_high = (S + delta_real) if feat_name == "sum_assured" else S

            V_high = self._predict_V(f_high, S_high, target_mean, target_std)

            # forward finite difference: dV/dx ≈ ΔV / Δx
            dV = ((V_high - V_base) / delta_real).squeeze(1).cpu().numpy()
            results[feat_name] = dV

            correct = (
                (feat_name == "interest_rate" and np.mean(dV) < 0) or
                (feat_name != "interest_rate" and np.mean(dV) > 0)
            )
            sign = "✓" if correct else "✗"
            print(
                f"  {sign} dV/d({feat_name:17s})  "
                f"Δ_real={delta_real:+.6f}  "
                f"Δ_norm={delta_norm:+.6f}  "
                f"mean dV/dx={np.mean(dV):+.4f}  "
                f"pos={( dV > 0).mean()*100:.0f}%"
            )

        # d²V/dr² via central FD
        dr_real = _PERTURBATIONS["interest_rate"]
        dr_norm = self._normalise_delta(
            "interest_rate", dr_real, interest_std, premium_std
        )
        f_r_hi = f.clone()
        f_r_lo = f.clone()
        f_r_hi[:, FEATURE_INDEX["interest_rate"]] += dr_norm
        f_r_lo[:, FEATURE_INDEX["interest_rate"]] -= dr_norm
        V_r_hi = self._predict_V(f_r_hi, S, target_mean, target_std)
        V_r_lo = self._predict_V(f_r_lo, S, target_mean, target_std)
        d2V_dr2 = (
            (V_r_hi - 2 * V_base + V_r_lo) / (dr_real ** 2)
        ).squeeze(1).cpu().numpy()

        # summary
        print("\n" + "=" * 60)
        print("SENSITIVITY SUMMARY  (true partial derivatives)")
        print("=" * 60)
        col_map = {
            "interest_rate": ("dV_dr",  "neg"),
            "mortality":     ("dV_dmu", "pos"),
            "premium":       ("dV_dP",  "pos"),
            "sum_assured":   ("dV_dS",  "pos"),
        }
        for feat, (col, expected) in col_map.items():
            vals    = results[feat]
            sign_pct = (
                (vals < 0).mean() * 100 if expected == "neg"
                else (vals > 0).mean() * 100
            )
            print(
                f"  {col:10s}  mean={vals.mean():+.4f}  "
                f"correct_sign={sign_pct:.0f}%"
            )
        print("=" * 60)

        sens_df = pd.DataFrame({
            "dV_dr":   results["interest_rate"],
            "dV_dmu":  results["mortality"],
            "dV_dP":   results["premium"],
            "dV_dS":   results["sum_assured"],
            "d2V_dr2": d2V_dr2,
        })

        return sens_df, v_mean_real

    # ── report ────────────────────────────────────────────────────────────

    def generate_sensitivity_report(
        self,
        ref_features: torch.Tensor,
        output_path: str,
        ref_raw_features: torch.Tensor | None = None,
        target_mean: float   = 0.0,
        target_std: float    = 1.0,
        interest_mean: float = 0.0,
        interest_std: float  = 1.0,
        premium_mean: float  = 0.0,
        premium_std: float   = 1.0,
    ) -> pd.DataFrame:
        """Run the separated sensitivity pipeline and write CSV + plots."""
        if ref_raw_features is None:
            raise ValueError("ref_raw_features is required.")

        sens_df, v_mean_real = self.compute_sensitivities(
            ref_features,
            ref_raw_features,
            target_mean,
            target_std,
            interest_mean,
            interest_std,
            premium_mean,
            premium_std,
        )

        # sensitivity plot (raw units)
        sens_df.to_csv(output_path, index=False)
        plot_sensitivities(sens_df, output_path.replace(".csv", ".png"))

        # elasticity plot (dimensionless)
        premium_ratio_typical = float(
            ref_raw_features[0, FEATURE_INDEX["premium"]].item()
        )
        elas_path = str(output_path).replace("sensitivity_report", "elasticity_report")
        elas_png  = elas_path.replace(".csv", ".png")
        elas_summary = compute_and_plot_elasticities(
            sens_df, elas_png, v_mean_real, premium_ratio_typical,
        )
        elas_summary.to_csv(elas_path, index=False)
        print(f"  elasticity CSV → {elas_path}")

        return sens_df'''