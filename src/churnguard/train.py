"""Training stage.

validate data -> stratified split -> Optuna search (every trial tracked in MLflow)
-> refit -> probability calibration -> business-optimal threshold -> test evaluation
-> explainability + model card -> versioned "challenger" artifact + MLflow registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churnguard import reporting
from churnguard.business import CampaignEconomics, campaign_net_value, optimal_threshold, value_curve
from churnguard.config import load_params, project_root, resolve
from churnguard.data.generate import ID_COL, TARGET
from churnguard.data.validate import FEATURE_COLUMNS, validate_or_raise
from churnguard.explain import global_importance
from churnguard.features import INPUT_COLUMNS, build_pipeline

log = logging.getLogger("churnguard.train")


def _git_sha() -> str:
    sha = os.getenv("GITHUB_SHA")
    if sha:
        return sha[:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=project_root(),
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "local"


def _data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:12]


def assign_split(ids: pd.Series, test_size: float, valid_size: float) -> np.ndarray:
    """Deterministic split from a hash of the customer id.

    A customer lands in the same split on every retraining run, so new data can be
    appended without ever leaking a past holdout customer into training — which is
    what makes champion-vs-challenger comparisons on the same holdout fair.
    """
    buckets = pd.util.hash_pandas_object(ids.astype(str), index=False).to_numpy() % 10_000 / 10_000
    return np.where(buckets < test_size, "test", np.where(buckets < test_size + valid_size, "valid", "train"))


def lift_at(y: np.ndarray, proba: np.ndarray, top: float = 0.1) -> float:
    k = max(1, int(len(y) * top))
    idx = np.argsort(-proba)[:k]
    return float(np.mean(y[idx]) / np.mean(y))


def evaluate(y, proba, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "brier": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "lift_top_decile": lift_at(np.asarray(y), proba, 0.1),
        "contact_rate": float(pred.mean()),
    }


def _search_space(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 150, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 96),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def setup_mlflow(params: dict) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI") or params["paths"]["mlflow_tracking_uri"]
    if uri.startswith("sqlite:///") and not uri.startswith("sqlite:////"):
        uri = f"sqlite:///{resolve(uri.removeprefix('sqlite:///'))}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(params["train"]["experiment_name"])


def train(params: dict | None = None, n_trials: int | None = None, data_path: str | Path | None = None) -> dict:
    params = params or load_params()
    seed = params["seed"]
    n_trials = n_trials or params["train"]["n_trials"]
    econ = CampaignEconomics.from_params(params)

    df = pd.read_csv(data_path or resolve(params["data"]["raw_path"]))
    validation = validate_or_raise(df)
    log.info("Data contract passed: %s rows, %s warnings", validation.n_rows, len(validation.warnings))

    split = assign_split(df[ID_COL], params["train"]["test_size"], params["train"]["valid_size"])
    X, y = df[FEATURE_COLUMNS], df[TARGET].to_numpy()
    X_train, y_train = X[split == "train"], y[split == "train"]
    X_valid, y_valid = X[split == "valid"], y[split == "valid"]
    X_test, y_test = X[split == "test"], y[split == "test"]

    setup_mlflow(params)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")

    with mlflow.start_run(run_name=f"train-{version}") as run:
        mlflow.set_tags({"stage": "training", "git_sha": _git_sha(), "model_family": "lightgbm"})
        mlflow.log_params({"n_trials": n_trials, "seed": seed, "data_hash": _data_hash(df),
                           "n_rows": len(df), "churn_rate": round(float(y.mean()), 4)})

        def objective(trial: optuna.Trial) -> float:
            hp = _search_space(trial)
            pipe = build_pipeline(hp, seed=seed).fit(X_train, y_train)
            auc = roc_auc_score(y_valid, pipe.predict_proba(X_valid)[:, 1])
            with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
                mlflow.log_params(hp)
                mlflow.log_metric("valid_roc_auc", auc)
            return auc

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                    study_name=f"churnguard-{version}")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_hp = {**study.best_params, "subsample_freq": 1}
        log.info("Best trial #%s valid AUC=%.4f", study.best_trial.number, study.best_value)
        mlflow.log_params({f"best_{k}": v for k, v in best_hp.items()})
        mlflow.log_metric("best_valid_roc_auc", study.best_value)

        base = build_pipeline(best_hp, seed=seed).fit(X_train, y_train)
        calibrated = CalibratedClassifierCV(FrozenEstimator(base), method=params["train"]["calibration"])
        calibrated.fit(X_valid, y_valid)

        p_valid = calibrated.predict_proba(X_valid)[:, 1]
        threshold, _ = optimal_threshold(y_valid, p_valid, X_valid["monthly_charges"].to_numpy(), econ)

        p_test_raw = base.predict_proba(X_test)[:, 1]
        p_test = calibrated.predict_proba(X_test)[:, 1]
        test_metrics = evaluate(y_test, p_test, threshold)
        valid_metrics = evaluate(y_valid, p_valid, threshold)

        mc_test = X_test["monthly_charges"].to_numpy()
        business = {
            "threshold": threshold,
            "offer_cost": econ.offer_cost,
            "retention_success_rate": econ.retention_success_rate,
            "value_horizon_months": econ.value_horizon_months,
            "net_value_test": campaign_net_value(y_test, p_test >= threshold, mc_test, econ),
            "net_value_at_0_5": campaign_net_value(y_test, p_test >= 0.5, mc_test, econ),
            "net_value_contact_all": campaign_net_value(y_test, np.ones_like(y_test), mc_test, econ),
            "net_value_perfect": campaign_net_value(y_test, y_test, mc_test, econ),
        }

        # Fair comparison: score the current champion on *this* holdout, so the
        # promotion gate compares both models on identical data.
        champion_on_test = None
        champ_file = resolve(params["paths"]["models_dir"]) / "champion" / "model.joblib"
        if champ_file.exists():
            try:
                champ = joblib.load(champ_file)
                p_champ = champ["model"].predict_proba(X_test)[:, 1]
                champion_on_test = evaluate(y_test, p_champ, champ["threshold"])
                champion_on_test["net_value"] = campaign_net_value(
                    y_test, p_champ >= champ["threshold"], X_test["monthly_charges"].to_numpy(), econ)
                mlflow.log_metrics({f"champion_test_{k}": v for k, v in champion_on_test.items()})
            except Exception as exc:
                log.warning("Could not score champion on the new holdout: %s", exc)

        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        mlflow.log_metrics({f"valid_{k}": v for k, v in valid_metrics.items()})
        mlflow.log_metrics({k: v for k, v in business.items() if k.startswith("net_value")})
        mlflow.log_metric("decision_threshold", threshold)

        # ---------- reports ----------
        reports = resolve(params["paths"]["reports_dir"])
        figdir = reports / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        importance = global_importance(base, X_test.sample(min(3000, len(X_test)), random_state=seed))
        ts, vs = value_curve(y_test, p_test, mc_test, econ)
        figures = {
            "roc_pr": reporting.plot_roc_pr(y_test, p_test, figdir / "roc_pr.png",
                                            test_metrics["roc_auc"], test_metrics["pr_auc"]),
            "calibration": reporting.plot_calibration(y_test, p_test_raw, p_test, figdir / "calibration.png"),
            "value_curve": reporting.plot_value_curve(ts, vs, threshold, figdir / "value_curve.png"),
            "importance": reporting.plot_importance(importance, figdir / "feature_importance.png"),
        }

        metadata = {
            "version": version,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_sha": _git_sha(),
            "mlflow_run_id": run.info.run_id,
            "model_family": "LightGBM + Platt calibration" if params["train"]["calibration"] == "sigmoid"
            else "LightGBM + isotonic calibration",
            "input_columns": INPUT_COLUMNS,
            "hyperparameters": best_hp,
            "optuna": {"n_trials": n_trials, "best_trial": study.best_trial.number,
                       "best_valid_roc_auc": study.best_value},
            "data": {"hash": _data_hash(df), "n_rows": len(df), "n_train": len(X_train),
                     "n_valid": len(X_valid), "n_test": len(X_test), "churn_rate": float(y.mean())},
            "metrics": {"test": test_metrics, "valid": valid_metrics, "champion_on_same_test": champion_on_test},
            "business": business,
            "feature_importance": {k: round(float(v), 5) for k, v in importance.items()},
        }

        out = resolve(params["paths"]["models_dir"]) / "challenger"
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": calibrated, "base": base, "threshold": threshold}, out / "model.joblib", compress=3)
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        # The training snapshot is the drift baseline, versioned *with* the model:
        # whichever model is champion, monitoring compares against what it learned from.
        X_train.to_csv(out / "reference.csv.gz", index=False, compression="gzip")

        reporting.write_model_report(metadata, figures, reports / "model_card.html")
        (reports / "metrics.json").write_text(
            json.dumps({"version": version, "test": test_metrics, "business": business}, indent=2), encoding="utf-8"
        )

        mlflow.log_artifacts(str(figdir), artifact_path="figures")
        mlflow.log_artifact(str(reports / "model_card.html"))
        mlflow.log_dict(metadata, "metadata.json")
        # Integer columns are declared as float so missing values at inference don't break the schema.
        sample = X_test.head(50).astype({c: float for c in X_test.select_dtypes("integer").columns})
        try:
            mlflow.sklearn.log_model(
                calibrated,
                name="model",
                signature=infer_signature(sample, calibrated.predict_proba(sample)[:, 1]),
                input_example=sample.head(3),
                registered_model_name=params["train"]["registered_model_name"],
                serialization_format="cloudpickle",
            )
        except Exception as exc:  # registry is a nice-to-have; the file artifact is the source of truth
            log.warning("MLflow model logging skipped: %s", exc)

    log.info("Challenger %s | test AUC %.4f | PR-AUC %.4f | net value $%.0f",
             version, test_metrics["roc_auc"], test_metrics["pr_auc"], business["net_value_test"])
    return metadata
