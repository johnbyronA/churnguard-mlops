"""Build the interactive demo site (docs/index.html, served by GitHub Pages).

Exports the *real* champion model — every LightGBM tree, the imputation medians,
the one-hot layout and the calibration curve — plus the real held-out test set,
drift scenarios, the Optuna search history and a real quality-gate decision, and
inlines them into a single self-contained HTML page. The page then scores
customers in the browser with the same model the API serves.

    python scripts/build_site.py [--gate-log path/to/registry_log.jsonl]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from churnguard.business import CampaignEconomics
from churnguard.config import load_params, resolve
from churnguard.data.generate import ID_COL, TARGET, generate_customers
from churnguard.data.validate import FEATURE_COLUMNS
from churnguard.features import BASE_NUMERIC, CATEGORICAL
from churnguard.monitoring import detect_drift
from churnguard.train import assign_split

ROOT = Path(__file__).resolve().parents[1]


def compact_tree(node: dict, nodes: list) -> int:
    """Flatten a LightGBM tree into [feature, threshold, left, right, value] rows.

    Leaves are rows with feature = -1. Returns the row index of `node`.
    """
    idx = len(nodes)
    if "leaf_value" in node:
        nodes.append([-1, 0, 0, 0, round(node["leaf_value"], 7)])
        return idx
    assert node["decision_type"] == "<=", node["decision_type"]
    nodes.append(None)
    left = compact_tree(node["left_child"], nodes)
    right = compact_tree(node["right_child"], nodes)
    nodes[idx] = [node["split_feature"], round(node["threshold"], 7), left, right,
                  round(node["internal_value"], 7)]
    return idx


def export_model(bundle: dict) -> dict:
    base = bundle["base"]
    prep = base.named_steps["prep"]
    cal = bundle["model"].calibrated_classifiers_[0].calibrators[0]
    dump = base[-1].booster_.dump_model()
    trees = []
    for t in dump["tree_info"]:
        nodes: list = []
        compact_tree(t["tree_structure"], nodes)
        trees.append(nodes)
    enc = prep.named_transformers_["cat"]
    return {
        "columns": list(prep.get_feature_names_out()),
        "numeric": list(prep.transformers_[0][2]),
        "medians": [float(v) for v in prep.named_transformers_["num"].statistics_],
        "categorical": CATEGORICAL,
        "categories": [list(map(str, c)) for c in enc.categories_],
        "trees": trees,
        "calibration": {"a": float(cal.a_), "b": float(cal.b_)},
        "threshold": float(bundle["threshold"]),
    }


def histogram(ref: pd.Series, cur: pd.Series, numeric: bool, bins: int = 18) -> dict:
    if not numeric:
        cats = sorted(set(ref.astype(str)) | set(cur.astype(str)))
        r = ref.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        c = cur.astype(str).value_counts(normalize=True).reindex(cats, fill_value=0)
        return {"labels": cats, "ref": r.round(4).tolist(), "cur": c.round(4).tolist()}
    both = pd.concat([ref, cur]).dropna()
    lo, hi = np.quantile(both, 0.005), np.quantile(both, 0.995)
    if both.nunique() <= 12:
        edges = np.arange(both.min() - 0.5, both.max() + 1.5, 1.0)
    else:
        edges = np.linspace(lo, hi, bins + 1)
    r = np.histogram(ref.dropna().clip(edges[0], edges[-1]), edges)[0] / ref.notna().sum()
    c = np.histogram(cur.dropna().clip(edges[0], edges[-1]), edges)[0] / cur.notna().sum()
    return {"edges": np.round(edges, 3).tolist(), "ref": np.round(r, 4).tolist(), "cur": np.round(c, 4).tolist()}


def drift_scenario(reference: pd.DataFrame, current: pd.DataFrame, cfg: dict, bundle: dict) -> dict:
    report = detect_drift(reference, current, cfg, bundle)
    s_ref = bundle["model"].predict_proba(reference)[:, 1]
    s_cur = bundle["model"].predict_proba(current)[:, 1]
    hists = {col: histogram(reference[col], current[col], col in BASE_NUMERIC) for col in BASE_NUMERIC + CATEGORICAL}
    hists["churn_score"] = histogram(pd.Series(s_ref), pd.Series(s_cur), True, bins=20)
    return {
        "retrain_required": report["retrain_required"],
        "reasons": report["reasons"],
        "prediction": report["prediction_drift"],
        "features": [{k: f[k] for k in ("feature", "type", "psi", "status", "ref_mean", "cur_mean")}
                     for f in report["features"]],
        "hists": hists,
    }


def optuna_history() -> list[dict]:
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{resolve('mlflow.db')}")
        meta = json.loads((resolve("models/champion/metadata.json")).read_text())
        runs = mlflow.search_runs(
            experiment_names=["churnguard"],
            filter_string=f"tags.mlflow.parentRunId = '{meta['mlflow_run_id']}'",
        )
        out = []
        for _, r in runs.iterrows():
            out.append({
                "trial": int(r["tags.mlflow.runName"].split("-")[-1]),
                "auc": round(float(r["metrics.valid_roc_auc"]), 5),
                "num_leaves": int(float(r["params.num_leaves"])),
                "learning_rate": round(float(r["params.learning_rate"]), 4),
                "n_estimators": int(float(r["params.n_estimators"])),
            })
        return sorted(out, key=lambda d: d["trial"])
    except Exception as exc:
        print("Optuna history unavailable:", exc)
        return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate-log", type=Path, help="registry_log.jsonl from a continuous-training run")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "index.html")
    args = ap.parse_args()

    params = load_params()
    bundle = joblib.load(resolve("models/champion/model.joblib"))
    meta = json.loads(resolve("models/champion/metadata.json").read_text())
    econ = CampaignEconomics.from_params(params)

    # Rebuild the exact held-out test set the champion was evaluated on.
    df = generate_customers(params["data"]["n_customers"], seed=params["seed"])
    split = assign_split(df[ID_COL], params["train"]["test_size"], params["train"]["valid_size"])
    test = df[split == "test"].reset_index(drop=True)
    p_test = bundle["model"].predict_proba(test[FEATURE_COLUMNS])[:, 1]

    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(test), size=240, replace=False)
    samples = test.iloc[sample_idx][FEATURE_COLUMNS + [TARGET]].copy()
    samples["p"] = p_test[sample_idx]
    samples = samples.replace({np.nan: None})

    reference = pd.read_csv(resolve("models/champion/reference.csv.gz"))
    cfg = params["monitoring"]
    stable = generate_customers(5000, seed=101).drop(columns=[TARGET])
    drifted = generate_customers(5000, seed=102, drift=True).drop(columns=[TARGET])

    gate = None
    if args.gate_log and args.gate_log.exists():
        lines = [json.loads(x) for x in args.gate_log.read_text().strip().splitlines()]
        gate = next((x for x in reversed(lines) if x.get("champion")), lines[-1])

    data = {
        "model": export_model(bundle),
        "meta": {
            "version": meta["version"],
            "trained_at": meta["trained_at"],
            "metrics": meta["metrics"]["test"],
            "business": meta["business"],
            "hyperparameters": meta["hyperparameters"],
            "data": meta["data"],
            "optuna": meta["optuna"],
        },
        "economics": {"offer_cost": econ.offer_cost, "success": econ.retention_success_rate,
                      "horizon": econ.value_horizon_months},
        "test": {"y": test[TARGET].astype(int).tolist(), "p": np.round(p_test, 5).tolist(),
                 "mc": test["monthly_charges"].round(2).tolist()},
        "samples": samples.to_dict(orient="records"),
        "drift": {"stable": drift_scenario(reference, stable, cfg, bundle),
                  "drifted": drift_scenario(reference, drifted, cfg, bundle),
                  "policy": cfg},
        "optuna": optuna_history(),
        "gate": gate,
    }

    template = (ROOT / "site" / "index.template.html").read_text(encoding="utf-8")
    payload = json.dumps(data, separators=(",", ":"), allow_nan=False)
    scorer = (ROOT / "site" / "scorer.js").read_text(encoding="utf-8")
    html = template.replace("/*__CHURNGUARD_DATA__*/null", payload).replace("/*__SCORER__*/", scorer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")

    # A small fixture so the JS scorer can be verified against Python.
    fixture = {"rows": test[FEATURE_COLUMNS].head(300).replace({np.nan: None}).to_dict(orient="records"),
               "p": np.round(p_test[:300], 6).tolist()}
    (ROOT / "site" / "parity_fixture.json").write_text(json.dumps(fixture), encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
