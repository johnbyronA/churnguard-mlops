import pytest
from fastapi.testclient import TestClient

from churnguard.serving.app import create_app

RISKY = {
    "customer_id": "CUST-TEST-1",
    "age": 34,
    "city": "Medellin",
    "tenure_months": 2,
    "contract": "month-to-month",
    "internet_service": "fiber",
    "num_products": 1,
    "monthly_charges": 120.0,
    "total_charges": 240.0,
    "payment_method": "electronic_check",
    "paperless_billing": 1,
    "support_tickets_90d": 5,
    "late_payments_12m": 3,
    "usage_change_pct": -35.0,
    "nps_score": 2,
}
LOYAL = {**RISKY, "customer_id": "CUST-TEST-2", "tenure_months": 64, "contract": "two-year",
         "internet_service": "dsl", "num_products": 4, "monthly_charges": 55.0, "total_charges": None,
         "payment_method": "bank_transfer", "support_tickets_90d": 0, "late_payments_12m": 0,
         "usage_change_pct": 6.0, "nps_score": None}


@pytest.fixture(scope="module")
def client(project):
    app = create_app(model_dir=project["root"] / "models" / "champion",
                     log_path=project["root"] / "logs" / "predictions.jsonl")
    with TestClient(app) as c:
        yield c


def test_health(client, project):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == project["metadata"]["version"]


def test_model_card_endpoint(client):
    body = client.get("/model").json()
    assert {"metrics", "business", "hyperparameters"} <= set(body)


def test_predict_ranks_risky_above_loyal(client):
    risky = client.post("/predict", json=RISKY).json()
    loyal = client.post("/predict", json=LOYAL).json()
    assert risky["churn_probability"] > loyal["churn_probability"]
    assert risky["risk_band"] in {"high", "critical"}
    assert loyal["risk_band"] in {"low", "medium"}
    assert len(risky["top_reasons"]) > 0
    assert all(r["impact"] > 0 for r in risky["top_reasons"])


def test_invalid_payload_is_rejected(client):
    r = client.post("/predict", json={**RISKY, "age": 7, "contract": "lifetime"})
    assert r.status_code == 422


def test_batch(client):
    body = client.post("/predict/batch", json={"customers": [RISKY, LOYAL, RISKY]}).json()
    assert body["n"] == 3
    assert body["flagged"] >= 2


def test_predictions_are_logged_for_monitoring(client, project):
    client.post("/predict", json=RISKY)
    log = project["root"] / "logs" / "predictions.jsonl"
    assert log.exists() and log.read_text().strip()


def test_prometheus_metrics(client):
    client.post("/predict", json=RISKY)
    text = client.get("/metrics").text
    assert "churnguard_predictions_total" in text
    assert "churnguard_request_latency_seconds" in text


def test_hot_reload(client, project):
    body = client.post("/admin/reload").json()
    assert body["current"] == project["metadata"]["version"]
