"""Turning probabilities into money.

A churn model is only useful if it drives a decision. We pick the threshold that
maximises the expected net value of a retention campaign, instead of a default 0.5:

    value(customer) = retention_success_rate * monthly_charges * horizon   if they would churn
    net(threshold)  = sum over contacted churners of value  -  offer_cost * contacted
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CampaignEconomics:
    offer_cost: float
    retention_success_rate: float
    value_horizon_months: int

    @classmethod
    def from_params(cls, params: dict) -> CampaignEconomics:
        b = params["business"]
        return cls(float(b["offer_cost"]), float(b["retention_success_rate"]), int(b["value_horizon_months"]))

    def customer_value(self, monthly_charges: np.ndarray) -> np.ndarray:
        return self.retention_success_rate * np.asarray(monthly_charges, float) * self.value_horizon_months


def campaign_net_value(
    y_true: np.ndarray, contact: np.ndarray, monthly_charges: np.ndarray, econ: CampaignEconomics
) -> float:
    y_true = np.asarray(y_true).astype(bool)
    contact = np.asarray(contact).astype(bool)
    saved = econ.customer_value(monthly_charges)[contact & y_true].sum()
    return float(saved - econ.offer_cost * contact.sum())


def value_curve(
    y_true: np.ndarray,
    proba: np.ndarray,
    monthly_charges: np.ndarray,
    econ: CampaignEconomics,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    thresholds = np.linspace(0.02, 0.98, 97) if thresholds is None else thresholds
    values = np.array([campaign_net_value(y_true, proba >= t, monthly_charges, econ) for t in thresholds])
    return thresholds, values


def optimal_threshold(
    y_true: np.ndarray, proba: np.ndarray, monthly_charges: np.ndarray, econ: CampaignEconomics
) -> tuple[float, float]:
    thresholds, values = value_curve(y_true, proba, monthly_charges, econ)
    best = int(np.argmax(values))
    return round(float(thresholds[best]), 2), float(values[best])


def risk_band(probability: float, threshold: float) -> str:
    if probability >= max(threshold * 1.8, 0.6):
        return "critical"
    if probability >= threshold:
        return "high"
    if probability >= threshold * 0.6:
        return "medium"
    return "low"


def recommended_action(band: str, monthly_charges: float) -> str:
    high_value = monthly_charges >= 90
    return {
        "critical": "Priority retention call within 24h" + (" + loyalty discount" if high_value else ""),
        "high": "Send personalised retention offer",
        "medium": "Add to proactive engagement nurture flow",
        "low": "No action — monitor",
    }[band]
