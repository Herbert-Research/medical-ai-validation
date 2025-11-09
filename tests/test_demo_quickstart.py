from pathlib import Path

import numpy as np
import pytest

import demo_quickstart as dq


def test_survival_dataset_structure():
    df = dq.build_survival_dataset(n_patients=10, seed=123)
    expected_cols = {
        "patient_id",
        "group",
        "time_months",
        "event",
        "tumor_stage",
        "lymph_positive",
        "nerve_invasion",
    }
    assert set(df.columns) == expected_cols
    assert df["group"].value_counts().to_dict() == {"Standard": 5, "AI-guided": 5}
    assert df["event"].isin({0, 1}).all()


def test_survival_dataset_handles_odd_patient_counts():
    df = dq.build_survival_dataset(n_patients=11, seed=321)
    counts = df["group"].value_counts().to_dict()
    assert counts["Standard"] == 5
    assert counts["AI-guided"] == 6
    assert counts["Standard"] + counts["AI-guided"] == 11


def test_calibration_metrics_are_finite():
    df = dq.simulate_metastasis_dataset(n_samples=200, seed=99)
    X = df.drop(columns="metastasis")
    y = df["metastasis"]
    X_train, X_test, y_train, y_test = dq.train_test_split(
        X, y, test_size=0.5, stratify=y, random_state=99
    )

    summary, probs = dq.evaluate_calibration_models(X_train, X_test, y_train, y_test)
    assert {"Uncalibrated", "Platt scaling", "Isotonic"} <= set(probs.keys())
    assert np.isfinite(summary["Brier Score"]).all()
    assert np.isfinite(summary["ECE (10-bin)"]).all()

    raw_brier = summary.loc[
        summary["Model"] == "Gradient Boosting (uncalibrated)", "Brier Score"
    ].item()
    iso_brier = summary.loc[
        summary["Model"] == "Isotonic Regression", "Brier Score"
    ].item()
    assert iso_brier <= raw_brier + 1e-6


def test_survival_metrics_are_stable(tmp_path):
    df = dq.build_survival_dataset(n_patients=240, seed=123)
    results = dq.analyse_survival(df, tmp_path / "km.png")
    assert 0.4 < results.hazard_ratio < 0.8
    assert results.logrank_p_value < 0.01
    assert results.ph_global_p_value > 0.05


def test_tcga_cohort_preparation():
    data_path = Path("data/tcga_2018_clinical_data.tsv")
    if not data_path.exists():
        pytest.skip("TCGA pilot cohort not available.")
    clinical = dq.load_tcga_clinical_table(data_path)
    cohort = dq.prepare_tcga_survival_cohort(clinical)
    assert not cohort.empty
    assert {"Node-negative", "Node-positive"} <= set(cohort["group"])
    assert (cohort["time_months"] > 0).all()
