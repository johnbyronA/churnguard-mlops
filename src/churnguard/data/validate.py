"""Data contracts.

Every dataset that enters training or scoring is checked against an explicit
contract: required columns, types, value ranges, allowed categories, null
budgets, duplicate keys and target sanity. Bad data fails loudly *before*
it can silently degrade a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from churnguard.data.generate import CITIES, CONTRACTS, ID_COL, INTERNET, PAYMENTS, TARGET


@dataclass(frozen=True)
class ColumnRule:
    kind: str  # "numeric" | "categorical" | "binary" | "id"
    min: float | None = None
    max: float | None = None
    allowed: tuple[str, ...] | None = None
    max_null_rate: float = 0.0


CONTRACT: dict[str, ColumnRule] = {
    ID_COL: ColumnRule("id"),
    "age": ColumnRule("numeric", 18, 100),
    "city": ColumnRule("categorical", allowed=tuple(CITIES)),
    "tenure_months": ColumnRule("numeric", 0, 120),
    "contract": ColumnRule("categorical", allowed=tuple(CONTRACTS)),
    "internet_service": ColumnRule("categorical", allowed=tuple(INTERNET)),
    "num_products": ColumnRule("numeric", 1, 10),
    "monthly_charges": ColumnRule("numeric", 0, 500),
    "total_charges": ColumnRule("numeric", 0, 60_000, max_null_rate=0.05),
    "payment_method": ColumnRule("categorical", allowed=tuple(PAYMENTS)),
    "paperless_billing": ColumnRule("binary"),
    "support_tickets_90d": ColumnRule("numeric", 0, 50),
    "late_payments_12m": ColumnRule("numeric", 0, 24),
    "usage_change_pct": ColumnRule("numeric", -100, 100),
    "nps_score": ColumnRule("numeric", 0, 10, max_null_rate=0.15),
}

FEATURE_COLUMNS = [c for c in CONTRACT if c != ID_COL]


@dataclass
class ValidationReport:
    n_rows: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {"n_rows": self.n_rows, "passed": self.passed, "errors": self.errors, "warnings": self.warnings}


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__("Data contract violated:\n- " + "\n- ".join(report.errors))


def validate(df: pd.DataFrame, require_target: bool = True, min_rows: int = 100) -> ValidationReport:
    report = ValidationReport(n_rows=len(df))

    if len(df) < min_rows:
        report.errors.append(f"only {len(df)} rows (minimum {min_rows})")

    expected = set(CONTRACT) | ({TARGET} if require_target else set())
    missing = expected - set(df.columns)
    if missing:
        report.errors.append(f"missing columns: {sorted(missing)}")
    extra = set(df.columns) - expected - {TARGET}
    if extra:
        report.warnings.append(f"unexpected columns ignored: {sorted(extra)}")

    for col, rule in CONTRACT.items():
        if col not in df.columns:
            continue
        s = df[col]
        null_rate = float(s.isna().mean())
        if null_rate > rule.max_null_rate:
            report.errors.append(f"{col}: null rate {null_rate:.2%} exceeds budget {rule.max_null_rate:.0%}")
        s = s.dropna()
        if rule.kind == "id":
            dups = int(s.duplicated().sum())
            if dups:
                report.errors.append(f"{col}: {dups} duplicated keys")
        elif rule.kind in {"numeric", "binary"}:
            if not pd.api.types.is_numeric_dtype(s):
                report.errors.append(f"{col}: expected numeric dtype, got {s.dtype}")
                continue
            lo, hi = (0, 1) if rule.kind == "binary" else (rule.min, rule.max)
            bad = int(((s < lo) | (s > hi)).sum())
            if bad:
                report.errors.append(f"{col}: {bad} values outside [{lo}, {hi}]")
        elif rule.kind == "categorical":
            unknown = sorted(set(s.astype(str)) - set(rule.allowed or ()))
            if unknown:
                report.errors.append(f"{col}: unknown categories {unknown[:5]}")

    if require_target and TARGET in df.columns:
        values = set(df[TARGET].dropna().unique().tolist())
        if not values <= {0, 1}:
            report.errors.append(f"{TARGET}: non-binary values {sorted(values)}")
        rate = float(df[TARGET].mean())
        if not 0.02 <= rate <= 0.8:
            report.errors.append(f"{TARGET}: implausible churn rate {rate:.2%}")
        elif not 0.08 <= rate <= 0.5:
            report.warnings.append(f"{TARGET}: unusual churn rate {rate:.2%}")

    return report


def validate_or_raise(df: pd.DataFrame, require_target: bool = True, min_rows: int = 100) -> ValidationReport:
    report = validate(df, require_target=require_target, min_rows=min_rows)
    if not report.passed:
        raise DataValidationError(report)
    return report
