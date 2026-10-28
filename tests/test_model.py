import json

import joblib
import numpy as np
import pandas as pd

from churnguard.business import CampaignEconomics, campaign_net_value, optimal_threshold, risk_band
from churnguard.data.generate import generate_customers
from churnguard.data.validate import FEATURE_COLUMNS
from churnguard.explain import contributions
from churnguard.features import CATEGORICAL, FeatureEngineer, original_feature
from churnguard.registry import quality_gate
from churnguard.train import assign_split

ECON = CampaignEconomics(offer_cost=40, retention_success_rate=0.3, value_horizon_months=6)


def test_feature_engineering_adds_business_features(customers):
    out = FeatureEngineer().transform(customers)
    assert {"tickets_per_tenure_year", "is_new_customer", "nps_missing"} <= set(out.columns)
    assert out["avg_charge_per_month"].notna().all()


def test_original_feature_mapping():
    assert original_feature("num__tenure_months") == "tenure_months"
    assert original_feature("cat__contract_month-to-month") == "contract"
    assert original_feature("cat__payment_method_credit_card") == "payment_method"
    assert all(original_feature(f"cat__{c}_x") == c for c in CATEGORICAL)


def test_hash_split_is_stable_and_disjoint(customers):
    ids = customers["customer_id"]
    first = assign_split(ids, 0.2, 0.2)
    # Appending new customers must not move existing ones between splits.
    extra = generate_customers(500, seed=99)["customer_id"]
    second = assign_split(pd.concat([ids, extra], ignore_index=True), 0.2, 0.2)[: len(ids)]
    assert (first == second).all()
    shares = {s: (first == s).mean() for s in ("train", "valid", "test")}
    assert 0.5 < shares["train"] < 0.7 and 0.15 < shares["test"] < 0.25


def test_campaign_value_math():
    y = np.array([1, 0, 1, 0])
    contact = np.array([1, 1, 0, 0])
    mc = np.array([100, 100, 100, 100])
    # one churner saved: 0.3 * 100 * 6 = 180, minus two offers = 80
    assert campaign_net_value(y, contact, mc, ECON) == 180 - 80


def test_optimal_threshold_beats_extremes(customers):
    rng = np.random.default_rng(0)
    y = customers["churn"].to_numpy()
    proba = np.clip(y * 0.5 + rng.random(len(y)) * 0.5, 0, 1)
    mc = customers["monthly_charges"].to_numpy()
    t, best = optimal_threshold(y, proba, mc, ECON)
    assert best >= campaign_net_value(y, np.ones_like(y), mc, ECON)
    assert best >= campaign_net_value(y, proba >= 0.98, mc, ECON)
    assert 0 < t < 1


def test_risk_bands_are_ordered():
    bands = [risk_band(p, 0.3) for p in (0.05, 0.2, 0.35, 0.9)]
    assert bands == ["low", "medium", "high", "critical"]


def test_training_produces_versioned_artifacts(project):
    champion = project["root"] / "models" / "champion"
    meta = json.loads((champion / "metadata.json").read_text())
    assert meta["version"].startswith("v")
    assert meta["metrics"]["test"]["roc_auc"] > 0.7
    assert (project["root"] / "reports" / "model_card.html").exists()
    assert (project["root"] / "reports" / "figures" / "value_curve.png").exists()


def test_model_predictions_are_valid_probabilities(project, customers):
    bundle = joblib.load(project["root"] / "models" / "champion" / "model.joblib")
    proba = bundle["model"].predict_proba(customers[FEATURE_COLUMNS])[:, 1]
    assert ((proba >= 0) & (proba <= 1)).all()
    assert proba[customers["churn"] == 1].mean() > proba[customers["churn"] == 0].mean()


def test_shap_contributions_reconstruct_raw_score(project, customers):
    bundle = joblib.load(project["root"] / "models" / "champion" / "model.joblib")
    base = bundle["base"]
    X = customers[FEATURE_COLUMNS].head(50)
    contrib = contributions(base, X)
    matrix = base[:-1].transform(X).to_numpy(float)
    margin = base[-1].booster_.predict(matrix, raw_score=True)
    bias = base[-1].booster_.predict(matrix, pred_contrib=True)[:, -1]
    # Grouped SHAP values + bias must add up exactly to the model's raw score.
    np.testing.assert_allclose(contrib.sum(axis=1).to_numpy() + bias, margin, atol=1e-6)


def test_quality_gate_blocks_regressions():
    rules = {"min_roc_auc": 0.8, "min_pr_auc": 0.5, "max_brier": 0.2, "max_auc_drop_vs_champion": 0.005}

    def meta(v, auc, nv=1000.0):
        return {"version": v, "metrics": {"test": {"roc_auc": auc, "pr_auc": 0.6, "brier": 0.1}},
                "business": {"net_value_test": nv}}

    assert quality_gate(meta("new", 0.85), None, rules).promoted
    assert not quality_gate(meta("new", 0.75), None, rules).promoted
    assert not quality_gate(meta("new", 0.82), meta("old", 0.84), rules).promoted
    assert quality_gate(meta("new", 0.838), meta("old", 0.84), rules).promoted
    same = meta("new", 0.85, nv=500.0)
    same["metrics"]["champion_on_same_test"] = {"roc_auc": 0.84, "net_value": 1000.0}
    assert not quality_gate(same, meta("old", 0.84), rules).promoted  # better AUC, worse money
