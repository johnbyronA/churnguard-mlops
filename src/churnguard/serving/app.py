"""FastAPI service that serves the champion model.

Endpoints
  GET  /health            liveness + loaded model version
  GET  /model             model card metadata (metrics, threshold, hyperparameters)
  POST /predict           score one customer, with SHAP reasons and a next-best action
  POST /predict/batch     score up to 1,000 customers
  POST /admin/reload      hot-reload the champion after a promotion (no downtime)
  GET  /metrics           Prometheus metrics (latency, volume, score distribution)

Every prediction is appended to a JSONL log — the production feed that the
monitoring job compares against the training reference.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from churnguard.business import CampaignEconomics, recommended_action, risk_band
from churnguard.config import load_params, resolve
from churnguard.explain import contributions, top_reasons
from churnguard.features import INPUT_COLUMNS
from churnguard.serving.schemas import BatchRequest, BatchResponse, Customer, Prediction

log = logging.getLogger("churnguard.api")

REQUESTS = Counter("churnguard_requests_total", "Prediction requests", ["endpoint", "status"])
PREDICTIONS = Counter("churnguard_predictions_total", "Customers scored", ["risk_band"])
LATENCY = Histogram("churnguard_request_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5))
SCORES = Histogram("churnguard_churn_probability", "Distribution of predicted churn probability",
                   buckets=tuple(i / 10 for i in range(1, 11)))


class ModelService:
    def __init__(self, model_dir: Path, log_path: Path | None, econ: CampaignEconomics):
        self.model_dir = model_dir
        self.log_path = log_path
        self.econ = econ
        self._lock = threading.Lock()
        self.load()

    def load(self) -> None:
        bundle = joblib.load(self.model_dir / "model.joblib")
        metadata = json.loads((self.model_dir / "metadata.json").read_text(encoding="utf-8"))
        with self._lock:
            self.model, self.base = bundle["model"], bundle["base"]
            self.threshold, self.metadata = round(float(bundle["threshold"]), 4), metadata
        log.info("Loaded model %s (threshold %.2f)", metadata["version"], self.threshold)

    @property
    def version(self) -> str:
        return self.metadata["version"]

    def predict(self, customers: list[Customer]) -> list[Prediction]:
        df = pd.DataFrame([c.model_dump() for c in customers])
        ids = df.pop("customer_id")
        X = df[INPUT_COLUMNS].astype({"total_charges": float, "nps_score": float})
        with self._lock:
            proba = self.model.predict_proba(X)[:, 1]
            contrib = contributions(self.base, X)
            engineered = self.base[0].transform(X)  # raw + engineered values, for readable reasons
            threshold, version = self.threshold, self.version

        out = []
        for i, p in enumerate(proba):
            band = risk_band(float(p), threshold)
            mc = float(X.iloc[i]["monthly_charges"])
            ev = float(p) * float(self.econ.customer_value(mc)) - self.econ.offer_cost
            out.append(Prediction(
                customer_id=ids.iloc[i],
                churn_probability=round(float(p), 4),
                will_churn=bool(p >= threshold),
                risk_band=band,
                recommended_action=recommended_action(band, mc),
                expected_value_of_action=round(ev, 2),
                top_reasons=top_reasons(contrib.iloc[i], engineered.iloc[i]),
                model_version=version,
            ))
            SCORES.observe(float(p))
            PREDICTIONS.labels(band).inc()
        self._log(df.assign(customer_id=ids), proba, version)
        return out

    def _log(self, df: pd.DataFrame, proba, version: str) -> None:
        if not self.log_path:
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            now = datetime.now(UTC).isoformat(timespec="seconds")
            with self.log_path.open("a", encoding="utf-8") as fh:
                for rec, p in zip(df.to_dict(orient="records"), proba, strict=True):
                    fh.write(json.dumps({**rec, "churn_probability": float(p), "model_version": version,
                                         "ts": now}, default=str) + "\n")
        except OSError as exc:  # logging must never break serving
            log.warning("Prediction log write failed: %s", exc)


def create_app(model_dir: str | Path | None = None, log_path: str | Path | None = None) -> FastAPI:
    params = load_params()
    model_dir = Path(model_dir or os.getenv("MODEL_DIR") or resolve(params["paths"]["models_dir"]) / "champion")
    if log_path is None:
        log_path = os.getenv("PREDICTIONS_LOG") or resolve(params["paths"]["predictions_log"])
    econ = CampaignEconomics.from_params(params)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = ModelService(model_dir, Path(log_path) if log_path else None, econ)
        yield

    app = FastAPI(
        title="ChurnGuard API",
        version="1.0.0",
        description="Real-time churn risk scoring with explanations and next-best-action recommendations.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time-ms"] = f"{elapsed * 1000:.1f}"
        if request.url.path.startswith("/predict"):
            LATENCY.labels(request.url.path).observe(elapsed)
            REQUESTS.labels(request.url.path, str(response.status_code)).inc()
        return response

    @app.get("/health")
    def health(request: Request):
        svc: ModelService = request.app.state.service
        return {"status": "ok", "model_version": svc.version, "threshold": svc.threshold}

    @app.get("/model")
    def model_info(request: Request):
        m = request.app.state.service.metadata
        return {k: m[k] for k in ("version", "trained_at", "git_sha", "model_family", "metrics",
                                  "business", "hyperparameters", "feature_importance")}

    @app.post("/predict", response_model=Prediction)
    def predict(customer: Customer, request: Request):
        return request.app.state.service.predict([customer])[0]

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(batch: BatchRequest, request: Request):
        svc: ModelService = request.app.state.service
        preds = svc.predict(batch.customers)
        return BatchResponse(model_version=svc.version, n=len(preds),
                             flagged=sum(p.will_churn for p in preds), predictions=preds)

    @app.post("/admin/reload")
    def reload(request: Request):
        token = os.getenv("ADMIN_TOKEN")
        if token and request.headers.get("X-Admin-Token") != token:
            raise HTTPException(status_code=401, detail="invalid admin token")
        svc: ModelService = request.app.state.service
        before = svc.version
        svc.load()
        return {"previous": before, "current": svc.version}

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app() if os.getenv("CHURNGUARD_AUTOLOAD", "1") == "1" else None
