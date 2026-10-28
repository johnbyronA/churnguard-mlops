"""Model explanations using LightGBM's native TreeSHAP (`pred_contrib=True`).

Using the booster's built-in SHAP values keeps the serving image light (no `shap`
dependency) while giving exact per-prediction attributions. One-hot columns are
summed back into the business feature they came from, so explanations read as
"contract", not "cat__contract_month-to-month".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churnguard.features import original_feature


def contributions(base_pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """SHAP contributions (log-odds) per original feature, one row per customer."""
    transformed = base_pipeline[:-1].transform(X)
    booster = base_pipeline[-1].booster_
    raw = booster.predict(transformed.to_numpy(dtype=float), pred_contrib=True)
    contrib = pd.DataFrame(raw[:, :-1], columns=transformed.columns, index=X.index)
    grouped = contrib.T.groupby(original_feature).sum().T
    return grouped


def global_importance(base_pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    return contributions(base_pipeline, X).abs().mean().sort_values(ascending=False)


def top_reasons(contrib_row: pd.Series, raw_row: pd.Series, k: int = 3) -> list[dict]:
    """Top-k features pushing this customer *towards* churn."""
    positive = contrib_row[contrib_row > 0].sort_values(ascending=False).head(k)
    reasons = []
    for feature, impact in positive.items():
        value = raw_row.get(feature)
        if isinstance(value, float | np.floating) and np.isnan(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        reasons.append({"feature": feature, "value": value, "impact": round(float(impact), 4)})
    return reasons
