import numpy as np
import pandas as pd

from churnguard.data.generate import generate_customers
from churnguard.data.validate import DataValidationError, validate, validate_or_raise


def test_generator_is_deterministic():
    a, b = generate_customers(500, seed=1), generate_customers(500, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_generator_has_realistic_churn_rate(customers):
    assert 0.12 < customers["churn"].mean() < 0.4


def test_month_to_month_churns_more(customers):
    rates = customers.groupby("contract")["churn"].mean()
    assert rates["month-to-month"] > rates["one-year"] > rates["two-year"]


def test_drift_mode_shifts_distributions():
    base, drifted = generate_customers(3000, seed=1), generate_customers(3000, seed=1, drift=True)
    assert drifted["monthly_charges"].mean() > base["monthly_charges"].mean() + 8
    assert drifted["support_tickets_90d"].mean() > base["support_tickets_90d"].mean()


def test_clean_data_passes_contract(customers):
    report = validate(customers)
    assert report.passed, report.errors


def test_contract_catches_bad_data(customers):
    bad = customers.copy()
    bad.loc[:10, "age"] = 7
    bad.loc[:5, "contract"] = "lifetime"
    bad.loc[:, "monthly_charges"] = np.nan
    bad = pd.concat([bad, bad.head(3)])  # duplicated keys
    report = validate(bad)
    joined = " ".join(report.errors)
    assert not report.passed
    for expected in ("age", "contract", "monthly_charges", "duplicated"):
        assert expected in joined


def test_missing_columns_raise(customers):
    try:
        validate_or_raise(customers.drop(columns=["tenure_months"]))
    except DataValidationError as exc:
        assert "tenure_months" in str(exc)
    else:
        raise AssertionError("expected DataValidationError")


def test_scoring_batches_do_not_need_target(customers):
    assert validate(customers.drop(columns=["churn"]), require_target=False).passed
