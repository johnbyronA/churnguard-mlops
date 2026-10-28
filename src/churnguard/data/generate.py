"""Synthetic customer data for a subscription (telecom / streaming) business.

The generator encodes realistic churn drivers — contract type, tenure, price,
support friction, payment behaviour, engagement trend and satisfaction — with
interactions and noise, so the model has real signal to learn but cannot be perfect.

`drift=True` simulates a later production period after a price increase and a
service incident: more month-to-month contracts, higher charges, more support
tickets and falling usage. Monitoring must detect it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = ["Medellin", "Bogota", "Cali", "Barranquilla", "Cartagena", "Bucaramanga"]
CONTRACTS = ["month-to-month", "one-year", "two-year"]
INTERNET = ["fiber", "dsl", "none"]
PAYMENTS = ["electronic_check", "credit_card", "bank_transfer", "cash"]

TARGET = "churn"
ID_COL = "customer_id"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_customers(n: int = 20_000, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract_p = [0.70, 0.18, 0.12] if drift else [0.55, 0.25, 0.20]
    contract = rng.choice(CONTRACTS, size=n, p=contract_p)
    m2m = contract == "month-to-month"
    one_year = contract == "one-year"

    # Longer contracts correlate with longer tenure.
    tenure_scale = np.where(m2m, 14, np.where(one_year, 28, 40))
    tenure = np.clip(rng.gamma(shape=1.6, scale=tenure_scale), 0, 72).round().astype(int)

    internet = rng.choice(INTERNET, size=n, p=[0.45, 0.40, 0.15])
    base_price = np.where(internet == "fiber", 85, np.where(internet == "dsl", 60, 28))
    price_shift = 12 if drift else 0
    monthly = np.clip(base_price + price_shift + rng.normal(0, 14, n), 15, 160).round(2)

    num_products = np.clip(rng.poisson(1.3, n) + 1, 1, 6)
    total = (monthly * tenure * rng.normal(1.0, 0.05, n)).round(2)

    support_tickets = rng.poisson(1.9 if drift else 1.1, n)
    late_payments = rng.poisson(0.6, n)
    payment = rng.choice(PAYMENTS, size=n, p=[0.30, 0.30, 0.25, 0.15])
    paperless = rng.random(n) < 0.6
    usage_change = rng.normal(-6 if drift else 0, 15, n).round(1)  # % change last 3 months
    nps = np.clip(np.round(rng.normal(7.2 - 0.45 * support_tickets, 1.8)), 0, 10)
    age = np.clip(rng.normal(41, 13, n), 18, 85).round().astype(int)
    city = rng.choice(CITIES, size=n, p=[0.24, 0.30, 0.16, 0.12, 0.10, 0.08])

    logit = (
        -1.95
        + 1.35 * m2m
        + 0.35 * one_year
        - 0.045 * tenure
        + 0.018 * (monthly - 70)
        + 0.012 * (monthly - 70) * m2m           # price sensitivity is worse without a contract
        + 0.32 * support_tickets
        + 0.42 * late_payments
        + 0.45 * (payment == "electronic_check")
        - 0.045 * usage_change
        - 0.20 * (nps - 7)
        - 0.28 * (num_products - 1)
        + 0.35 * (internet == "fiber")
        + 0.25 * (age > 65)
        - 0.15 * paperless
        + rng.normal(0, 0.55, n)                  # unobserved factors
    )
    churn = (rng.random(n) < _sigmoid(logit)).astype(int)

    df = pd.DataFrame(
        {
            ID_COL: [f"CUST-{seed:03d}-{i:06d}" for i in range(n)],
            "age": age,
            "city": city,
            "tenure_months": tenure,
            "contract": contract,
            "internet_service": internet,
            "num_products": num_products,
            "monthly_charges": monthly,
            "total_charges": total,
            "payment_method": payment,
            "paperless_billing": paperless.astype(int),
            "support_tickets_90d": support_tickets,
            "late_payments_12m": late_payments,
            "usage_change_pct": usage_change,
            "nps_score": nps,
            TARGET: churn,
        }
    )

    # Realistic missingness: not every customer answers the NPS survey,
    # and billing has not closed yet for brand-new customers.
    df.loc[rng.random(n) < 0.06, "nps_score"] = np.nan
    df.loc[(df["tenure_months"] == 0) & (rng.random(n) < 0.7), "total_charges"] = np.nan
    return df
