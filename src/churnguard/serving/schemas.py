"""Request/response contracts for the prediction API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "customer_id": "CUST-000123",
            "age": 34,
            "city": "Medellin",
            "tenure_months": 4,
            "contract": "month-to-month",
            "internet_service": "fiber",
            "num_products": 1,
            "monthly_charges": 112.5,
            "total_charges": 450.0,
            "payment_method": "electronic_check",
            "paperless_billing": 1,
            "support_tickets_90d": 4,
            "late_payments_12m": 2,
            "usage_change_pct": -28.0,
            "nps_score": 3,
        }
    })

    customer_id: str | None = Field(None, max_length=64)
    age: int = Field(..., ge=18, le=100)
    city: Literal["Medellin", "Bogota", "Cali", "Barranquilla", "Cartagena", "Bucaramanga"]
    tenure_months: int = Field(..., ge=0, le=120)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["fiber", "dsl", "none"]
    num_products: int = Field(..., ge=1, le=10)
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float | None = Field(None, ge=0)
    payment_method: Literal["electronic_check", "credit_card", "bank_transfer", "cash"]
    paperless_billing: int = Field(..., ge=0, le=1)
    support_tickets_90d: int = Field(..., ge=0, le=50)
    late_payments_12m: int = Field(..., ge=0, le=24)
    usage_change_pct: float = Field(..., ge=-100, le=100)
    nps_score: float | None = Field(None, ge=0, le=10)


class Reason(BaseModel):
    feature: str
    value: float | int | str | None
    impact: float = Field(..., description="SHAP contribution towards churn (log-odds)")


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float
    will_churn: bool
    risk_band: Literal["low", "medium", "high", "critical"]
    recommended_action: str
    expected_value_of_action: float = Field(..., description="Expected USD saved minus offer cost")
    top_reasons: list[Reason]
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    model_version: str
    n: int
    flagged: int
    predictions: list[Prediction]
