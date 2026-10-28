import joblib
import numpy as np

from churnguard.data.generate import generate_customers
from churnguard.monitoring import detect_drift, monitor, psi_categorical, psi_numeric

CFG = {"psi_warning": 0.1, "psi_alert": 0.2, "ks_pvalue": 0.01,
       "max_drifted_feature_share": 0.25, "score_psi_alert": 0.2}


def test_psi_is_zero_for_identical_samples():
    x = np.random.default_rng(0).normal(size=5000)
    assert psi_numeric(x, x) < 1e-9


def test_psi_grows_with_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    small, large = psi_numeric(ref, rng.normal(0.1, 1, 5000)), psi_numeric(ref, rng.normal(1.0, 1, 5000))
    assert small < 0.1 < large


def test_categorical_psi_detects_mix_change():
    import pandas as pd

    ref = pd.Series(["a"] * 700 + ["b"] * 300)
    cur = pd.Series(["a"] * 300 + ["b"] * 700)
    assert psi_categorical(ref, ref) < 1e-9
    assert psi_categorical(ref, cur) > 0.2


def test_no_retrain_on_same_distribution(project):
    bundle = joblib.load(project["root"] / "models" / "champion" / "model.joblib")
    ref = generate_customers(4000, seed=10).drop(columns=["churn"])
    cur = generate_customers(4000, seed=11).drop(columns=["churn"])
    report = detect_drift(ref, cur, CFG, bundle)
    assert not report["retrain_required"], report["reasons"]


def test_drifted_batch_triggers_retraining(project):
    bundle = joblib.load(project["root"] / "models" / "champion" / "model.joblib")
    ref = generate_customers(4000, seed=10).drop(columns=["churn"])
    cur = generate_customers(4000, seed=11, drift=True).drop(columns=["churn"])
    report = detect_drift(ref, cur, CFG, bundle)
    assert report["retrain_required"]
    assert "monthly_charges" in report["drifted_features"]
    assert report["prediction_drift"]["cur_mean_score"] > report["prediction_drift"]["ref_mean_score"]


def test_monitor_writes_reports(project):
    params = project["params"]
    current = project["root"] / params["data"]["current_path"]
    current.parent.mkdir(parents=True, exist_ok=True)
    generate_customers(2000, seed=5, drift=True).drop(columns=["churn"]).to_csv(current, index=False)
    report = monitor(params)
    assert (project["root"] / "reports" / "drift_report.html").exists()
    assert isinstance(report["retrain_required"], bool)
