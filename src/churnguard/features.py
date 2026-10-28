"""Feature engineering and the model pipeline.

Feature engineering lives *inside* the scikit-learn pipeline, so the exact same
code runs in training, batch scoring and the online API — no training/serving skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CATEGORICAL = ["city", "contract", "internet_service", "payment_method"]
BASE_NUMERIC = [
    "age",
    "tenure_months",
    "num_products",
    "monthly_charges",
    "total_charges",
    "paperless_billing",
    "support_tickets_90d",
    "late_payments_12m",
    "usage_change_pct",
    "nps_score",
]
ENGINEERED = [
    "avg_charge_per_month",
    "tickets_per_tenure_year",
    "is_new_customer",
    "charge_per_product",
    "nps_missing",
    "engagement_drop",
]
NUMERIC = BASE_NUMERIC + ENGINEERED
INPUT_COLUMNS = CATEGORICAL + BASE_NUMERIC


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless business features derived from raw customer attributes."""

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X[INPUT_COLUMNS].copy()
        tenure = X["tenure_months"].clip(lower=1)
        total = X["total_charges"].fillna(X["monthly_charges"] * X["tenure_months"])
        X["avg_charge_per_month"] = total / tenure
        X["tickets_per_tenure_year"] = X["support_tickets_90d"] / (tenure / 12 + 1)
        X["is_new_customer"] = (X["tenure_months"] < 6).astype(int)
        X["charge_per_product"] = X["monthly_charges"] / X["num_products"].clip(lower=1)
        X["nps_missing"] = X["nps_score"].isna().astype(int)
        X["engagement_drop"] = (X["usage_change_pct"] < -15).astype(int)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(INPUT_COLUMNS + ENGINEERED)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ],
        verbose_feature_names_out=True,
    ).set_output(transform="pandas")


def build_pipeline(model_params: dict | None = None, seed: int = 42) -> Pipeline:
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": -1,
    }
    params.update(model_params or {})
    return Pipeline(
        [
            ("features", FeatureEngineer()),
            ("prep", build_preprocessor()),
            ("model", LGBMClassifier(**params)),
        ]
    )


def original_feature(transformed_name: str) -> str:
    """Map a transformed column back to the business feature it came from.

    'num__tenure_months' -> 'tenure_months'; 'cat__contract_one-year' -> 'contract'.
    """
    prefix, _, rest = transformed_name.partition("__")
    if prefix == "cat":
        for col in CATEGORICAL:
            if rest.startswith(col + "_"):
                return col
    return rest
