"""
Headless entry point that reproduces the survival and calibration workflow.

Usage:
    python demo_quickstart.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from matplotlib import pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split


@dataclass
class SurvivalResults:
    median_ai_guided: float
    median_standard: float
    logrank_p_value: float
    hazard_ratio: float
    hazard_ratio_ci: Tuple[float, float]
    c_index: float


def build_survival_dataset(n_patients: int = 240, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    half = n_patients // 2
    group = np.array(["Standard"] * half + ["AI-guided"] * half)
    ai_indicator = (group == "AI-guided").astype(int)

    tumor_stage = rng.integers(1, 4, size=n_patients)  # I-III
    lymph_positive = rng.binomial(1, 0.35, size=n_patients)
    nerve_invasion = rng.binomial(1, 0.25, size=n_patients)

    linear_predictor = (
        0.55 * (tumor_stage - 1)
        + 0.90 * lymph_positive
        + 0.65 * nerve_invasion
        - 0.85 * ai_indicator
        + rng.normal(0.0, 0.25, size=n_patients)
    )

    base_hazard = 0.018  # monthly baseline hazard
    survival_time_months = -np.log(rng.uniform(size=n_patients)) / (
        base_hazard * np.exp(linear_predictor)
    )

    censoring_time = -np.log(rng.uniform(size=n_patients)) / 0.012
    observed_time = np.minimum(survival_time_months, censoring_time)
    event_observed = (survival_time_months <= censoring_time).astype(int)

    return pd.DataFrame(
        {
            "patient_id": range(1, n_patients + 1),
            "group": group,
            "time_months": observed_time,
            "event": event_observed,
            "tumor_stage": tumor_stage,
            "lymph_positive": lymph_positive,
            "nerve_invasion": nerve_invasion,
        }
    )


def analyse_survival(data: pd.DataFrame, output_path: Path) -> SurvivalResults:
    sns.set_context("talk")
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))

    kmf = KaplanMeierFitter()
    medians: Dict[str, float] = {}

    for group, color in [("Standard", "#2A9D8F"), ("AI-guided", "#E76F51")]:
        subset = data[data["group"] == group]
        kmf.fit(subset["time_months"], event_observed=subset["event"], label=group)
        kmf.plot_survival_function(ci_show=True, ci_alpha=0.15, lw=2, color=color)
        medians[group] = kmf.median_survival_time_

    plt.title("Kaplan-Meier Survival Curves with 95% CI", fontsize=16, fontweight="bold")
    plt.xlabel("Time since gastrectomy (months)")
    plt.ylabel("Disease-free survival probability")
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    standard = data[data["group"] == "Standard"]
    ai_guided = data[data["group"] == "AI-guided"]
    logrank = logrank_test(
        standard["time_months"],
        ai_guided["time_months"],
        standard["event"],
        ai_guided["event"],
    )

    covariate_cols = ["tumor_stage", "lymph_positive", "nerve_invasion"]
    cox_df = data[["time_months", "event"] + covariate_cols].copy()
    cox_df["ai_guided"] = (data["group"] == "AI-guided").astype(int)

    cph = CoxPHFitter()
    cph.fit(cox_df, duration_col="time_months", event_col="event")

    hazard_ratio = float(np.exp(cph.params_["ai_guided"]))
    ci_lower, ci_upper = np.exp(cph.confidence_intervals_.loc["ai_guided"])

    risk_scores = cph.predict_partial_hazard(cox_df).values.ravel()
    c_index = float(
        concordance_index(
            event_times=cox_df["time_months"],
            # negate partial hazards so higher scores = longer survival
            predicted_scores=-risk_scores,
            event_observed=cox_df["event"],
        )
    )

    return SurvivalResults(
        median_ai_guided=medians["AI-guided"],
        median_standard=medians["Standard"],
        logrank_p_value=float(logrank.p_value),
        hazard_ratio=hazard_ratio,
        hazard_ratio_ci=(float(ci_lower), float(ci_upper)),
        c_index=c_index,
    )


def simulate_metastasis_dataset(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    tumor_stage = rng.integers(1, 4, size=n_samples)
    lymph_nodes = rng.poisson(lam=6, size=n_samples)
    station_distance = rng.normal(loc=3.0, scale=1.0, size=n_samples)
    histology_score = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    bmi = rng.normal(loc=24, scale=4, size=n_samples)

    linear_pred = (
        0.6 * tumor_stage
        + 0.12 * lymph_nodes
        - 0.35 * station_distance
        - 0.25 * histology_score
        + 0.02 * (bmi - 25)
    )

    base_risk = 1 / (1 + np.exp(-(linear_pred - 2.5)))
    metastasis = rng.binomial(1, base_risk)

    return pd.DataFrame(
        {
            "tumor_stage": tumor_stage,
            "lymph_nodes": lymph_nodes,
            "station_distance": station_distance,
            "histology_score": histology_score,
            "bmi": bmi,
            "metastasis": metastasis,
        }
    )


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins) - 1
    ece = 0.0
    for idx in range(n_bins):
        mask = bin_ids == idx
        if np.any(mask):
            observed = y_true[mask].mean()
            predicted = y_prob[mask].mean()
            ece += abs(observed - predicted) * mask.mean()
    return float(ece)


def calibration_slope_intercept(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    eps = 1e-6
    clipped = np.clip(y_prob, eps, 1 - eps)
    logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(logits, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def evaluate_calibration_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    def make_estimator() -> GradientBoostingClassifier:
        return GradientBoostingClassifier(
            random_state=42,
            n_estimators=250,
            learning_rate=0.05,
        )

    base_model = make_estimator()
    base_model.fit(X_train, y_train)
    raw_probs = base_model.predict_proba(X_test)[:, 1]

    sigmoid_model = CalibratedClassifierCV(
        estimator=make_estimator(), method="sigmoid", cv=5
    )
    sigmoid_model.fit(X_train, y_train)
    sigmoid_probs = sigmoid_model.predict_proba(X_test)[:, 1]

    isotonic_model = CalibratedClassifierCV(
        estimator=make_estimator(), method="isotonic", cv=5
    )
    isotonic_model.fit(X_train, y_train)
    isotonic_probs = isotonic_model.predict_proba(X_test)[:, 1]

    labels = y_test.to_numpy()

    def summarise(label: str, probs: np.ndarray) -> Dict[str, float | str]:
        slope, intercept = calibration_slope_intercept(labels, probs)
        return {
            "Model": label,
            "Brier Score": float(brier_score_loss(labels, probs)),
            "ECE (10-bin)": expected_calibration_error(labels, probs),
            "Calibration Slope": slope,
            "Calibration Intercept": intercept,
        }

    summary = pd.DataFrame(
        [
            summarise("Gradient Boosting (uncalibrated)", raw_probs),
            summarise("Platt Scaling (sigmoid)", sigmoid_probs),
            summarise("Isotonic Regression", isotonic_probs),
        ]
    )

    return summary, {
        "Uncalibrated": raw_probs,
        "Platt scaling": sigmoid_probs,
        "Isotonic": isotonic_probs,
    }


def plot_calibration_curves(probabilities: Dict[str, np.ndarray], y_test: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for label, probs in probabilities.items():
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, markersize=7, label=label)

    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("Calibration Curves for Station-Level Metastasis Risk")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run() -> None:
    np.random.seed(42)
    output_km = Path("kaplan_meier_example.png")
    output_calibration = Path("calibration_curve_example.png")

    survival_df = build_survival_dataset()
    survival_results = analyse_survival(survival_df, output_km)

    metastasis_df = simulate_metastasis_dataset()
    X = metastasis_df.drop(columns="metastasis")
    y = metastasis_df["metastasis"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    calibration_summary, probs = evaluate_calibration_models(X_train, X_test, y_train, y_test)
    plot_calibration_curves(probs, y_test, output_calibration)

    print("=== Survival Analysis Summary ===")
    print(f"Median DFS (AI-guided): {survival_results.median_ai_guided:.1f} months")
    print(f"Median DFS (Standard):  {survival_results.median_standard:.1f} months")
    print(f"Log-rank p-value:      {survival_results.logrank_p_value:.4f}")
    print(
        "Hazard ratio (AI-guided vs Standard): "
        f"{survival_results.hazard_ratio:.2f} "
        f"(95% CI {survival_results.hazard_ratio_ci[0]:.2f}–{survival_results.hazard_ratio_ci[1]:.2f})"
    )
    print(f"C-index:                {survival_results.c_index:.3f}")

    print("\n=== Calibration Summary ===")
    print(calibration_summary.round(3).to_string(index=False))
    print(f"\nFigures saved to: {output_km.resolve()}, {output_calibration.resolve()}")


if __name__ == "__main__":
    run()
