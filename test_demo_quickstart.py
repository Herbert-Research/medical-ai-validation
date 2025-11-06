import numpy as np

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
