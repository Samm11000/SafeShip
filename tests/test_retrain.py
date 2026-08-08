"""
Tests for the nightly retrain Lambda.

WHY THESE EXIST
    Retraining is the one component that can change a paying tenant's verdicts
    without anyone asking it to. Until now three things made that dangerous:

      1. MIN_BUILDS was 5 while the gate printed and logged "dataset_size >= 80".
      2. A 20% split of 5 rows is a ONE-ROW test set, on which `precision >= 0.75`
         is 0.0 or 1.0 — a coin flip promoting a model into production.
      3. .fillna(0) meant a missing test_pass_rate trained as 0.0 ("everything
         failed") while /score imputes the median. Train/serve skew.

    Now that safeship_ci sends real features and real labels can start arriving,
    these stop being theoretical.

The handler lives at lambda/retrain/handler.py and `lambda` is a Python keyword,
so it is loaded by path rather than imported.
"""
from __future__ import annotations

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "ml"))


def _load_handler():
    path = os.path.join(REPO, "lambda", "retrain", "handler.py")
    spec = importlib.util.spec_from_file_location("retrain_handler", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


handler = _load_handler()


# ── the isolated copies must not drift ───────────────────────────────────────

def test_training_medians_match_the_serving_path():
    """
    The handler carries its own copy because it is packaged alone (its Dockerfile
    copies only handler.py). If it drifts from app/imputation.py, training imputes
    one distribution while /score imputes another — which is the exact skew this
    change was made to remove.
    """
    from imputation import TRAINING_MEDIANS as SERVING

    assert handler.TRAINING_MEDIANS == dict(SERVING), (
        "lambda/retrain/handler.py::TRAINING_MEDIANS has drifted from "
        "app/imputation.py. Recompute both with tools/recompute_medians.py."
    )


def test_int_feature_set_matches_the_serving_path():
    from imputation import _INT_FEATURES as SERVING_INTS

    assert handler._INT_FEATURES == set(SERVING_INTS)


# ── imputation: never zero-fill ──────────────────────────────────────────────

def _frame(**overrides):
    base = {c: [1.0, 2.0, 3.0, 4.0] for c in handler.FEATURE_COLUMNS}
    base.update(overrides)
    return pd.DataFrame(base)


def test_a_missing_test_pass_rate_is_not_trained_as_zero():
    """
    THE REGRESSION. test_pass_rate=0.0 means "every single test failed" — the most
    alarming value in the range — and it carries 25% of the model's weight.
    """
    df = _frame(test_pass_rate=[0.9, np.nan, 0.8, 0.95])
    out = handler.impute_frame(df)

    filled = out["test_pass_rate"].iloc[1]
    assert filled != 0.0, "a missing test_pass_rate must not become 'all tests failed'"
    assert filled == pytest.approx(0.9), "expected the column median of 0.8/0.9/0.95"
    assert not out.isnull().any().any()


def test_a_missing_failure_rate_is_not_trained_as_flawless():
    df = _frame(recent_failure_rate=[0.4, 0.5, np.nan, 0.6])
    out = handler.impute_frame(df)
    assert out["recent_failure_rate"].iloc[2] != 0.0
    assert out["recent_failure_rate"].iloc[2] == pytest.approx(0.5)


def test_an_entirely_empty_column_falls_back_to_the_training_median():
    # A tenant that has never reported test_pass_rate has no median of its own.
    df = _frame(test_pass_rate=[np.nan] * 4)
    out = handler.impute_frame(df)
    assert out["test_pass_rate"].iloc[0] == pytest.approx(
        handler.TRAINING_MEDIANS["test_pass_rate"])


def test_integer_features_stay_integral_after_imputation():
    df = _frame(files_changed=[3, np.nan, 8, 10])
    out = handler.impute_frame(df)
    value = out["files_changed"].iloc[1]
    assert float(value) == int(value), f"{value} is not a whole number of files"


def test_imputation_returns_the_contract_columns_in_order():
    out = handler.impute_frame(_frame())
    assert list(out.columns) == handler.FEATURE_COLUMNS


def test_a_string_typed_column_from_dynamo_is_coerced_not_dropped():
    # DynamoDB/JSON round-trips can hand back numbers as strings.
    df = _frame(diff_size=["10", "20", None, "40"])
    out = handler.impute_frame(df)
    assert out["diff_size"].iloc[0] == 10
    assert not out["diff_size"].isnull().any()


# ── the promotion gate ───────────────────────────────────────────────────────

def _synthetic_history(n, risky_every=3, ascending_signal=True):
    """A tenant history with a learnable signal and ascending timestamps."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        risky = (i % risky_every == 0)
        rows.append({
            "timestamp": 1_700_000_000 + i * 3600,
            "diff_size": (i if ascending_signal else 100) + (800 if risky else 20),
            "files_changed": 30 if risky else 3,
            "hour_of_day": 3 if risky else 14,
            "day_of_week": 5 if risky else 2,
            "recent_failure_rate": 0.7 if risky else 0.05,
            "test_pass_rate": 0.5 if risky else 0.99,
            "is_hotfix": 1 if risky else 0,
            "deployer_exp": 2 if risky else 60,
            "days_since_deploy": 40.0 if risky else 2.0,
            "build_time_delta": 1.5 if risky else 0.0,
            "label": 1 if risky else 0,
            "sample_weight": 1.0,
        })
    return pd.DataFrame(rows)


def test_the_held_out_set_is_the_most_recent_builds_not_a_random_sample():
    """
    A random split trains on builds that happened after the ones it is scored on.
    In production the model only ever has the past, so a random split reports a
    number the deployed model cannot achieve — and the gate then trusts it.
    """
    df = _synthetic_history(100)
    _, X_test, _ = handler.train_model(df)

    # diff_size rises with time in this fixture, so the newest rows hold the
    # largest values. A random sample would mix early rows into the test set.
    assert len(X_test) == 20
    assert X_test["diff_size"].min() > df["diff_size"].iloc[:60].min()


def test_shuffled_input_is_still_split_by_timestamp():
    df = _synthetic_history(100).sample(frac=1.0, random_state=7)
    _, X_test, _ = handler.train_model(df)
    # Recovered from the timestamp column, not from arrival order.
    assert len(X_test) == 20


def test_a_tiny_dataset_cannot_promote_a_model():
    """
    The coin flip: 5 labelled builds gave a 1-row test set, where precision is
    0.0 or 1.0. This must now refuse to promote rather than gamble.
    """
    df = _synthetic_history(10)
    model, X_test, y_test = handler.train_model(df)

    passed, metrics = handler.validate_model(model, None, X_test, y_test, len(df))

    assert passed is False, "a 2-row test set must never promote a model"
    failed = [name for name, ok in metrics["checks"].items() if not ok]
    assert any("dataset_size" in f for f in failed)
    assert any("test_rows" in f for f in failed)


def test_a_test_window_without_enough_failures_cannot_promote():
    # Plenty of data, but the recent window happens to be all-green: precision
    # and AUC are then meaningless, so the honest move is to skip.
    df = _synthetic_history(200)
    df.loc[df.index[-40:], "label"] = 0          # last 40 builds all fine
    model, X_test, y_test = handler.train_model(df)

    passed, metrics = handler.validate_model(model, None, X_test, y_test, len(df))
    assert passed is False
    failed = [name for name, ok in metrics["checks"].items() if not ok]
    assert any("test_risky" in f for f in failed)


def test_check_names_state_the_thresholds_actually_enforced():
    """
    The bug that hid the other bugs: the gate printed "dataset_size >= 80" while
    comparing against MIN_BUILDS = 5, so its own logs disguised the problem.
    """
    df = _synthetic_history(200)
    model, X_test, y_test = handler.train_model(df)
    _, metrics = handler.validate_model(model, None, X_test, y_test, len(df))

    names = " ".join(metrics["checks"])
    assert f"dataset_size >= {handler.MIN_BUILDS}" in names
    assert f"test_rows >= {handler.MIN_TEST_ROWS}" in names
    assert f"test_risky >= {handler.MIN_TEST_RISKY}" in names
    # The precision bar is per-tenant now, so the check names the multiple and
    # the tenant's own failure rate rather than a number chosen for someone else.
    assert f"{handler.MIN_PRECISION_LIFT}x" in names
    assert "failure rate" in names


def test_the_precision_bar_scales_with_the_tenants_failure_rate():
    """
    Precision is bounded below by the base rate, so a fixed threshold means
    something different for every tenant. Under the old `precision >= 0.75`:

      a tenant failing 76% of the time -> a model that flags EVERY build scores
          0.760 and gets promoted
      a tenant at the DORA-elite rate under 5% -> can almost never clear the bar,
          however good the model is

    Backwards, since SafeShip is worth least to the first and most to the second.
    """
    import numpy as np

    class AlwaysRisky:
        def predict(self, X):
            return np.ones(len(X), dtype=int)

        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    import pandas as pd

    for base_rate in (0.76, 0.05):
        n = 400
        y = pd.Series([1] * int(n * base_rate) + [0] * (n - int(n * base_rate)))
        X = pd.DataFrame({c: np.random.default_rng(0).normal(size=n)
                          for c in handler.FEATURE_COLUMNS})
        passed, metrics = handler.validate_model(AlwaysRisky(), None, X, y, 500)

        assert passed is False, (
            f"a model that flags everything was promoted at a {base_rate:.0%} "
            "base rate"
        )
        failed = [k for k, ok in metrics["checks"].items() if not ok]
        assert any("failure rate" in f for f in failed), (
            f"the precision check did not catch it at {base_rate:.0%}; "
            f"failed checks were {failed}"
        )


def test_the_metrics_record_lift_so_promotions_stay_comparable():
    df = _synthetic_history(400)
    model, X_test, y_test = handler.train_model(df)
    _, metrics = handler.validate_model(model, None, X_test, y_test, len(df))
    assert "precision_lift" in metrics and "base_rate" in metrics
    assert metrics["precision_required"] >= handler.MIN_PRECISION_ABS


def test_the_gate_constants_are_mutually_reachable():
    """
    Getting these wrong fails silently — as "no tenant was ever promoted", which
    looks like "no tenant qualified yet". Both directions have to hold.
    """
    implied_test_rows = handler.MIN_BUILDS * 0.20
    assert implied_test_rows >= handler.MIN_TEST_ROWS, (
        f"MIN_BUILDS={handler.MIN_BUILDS} yields only {implied_test_rows:.0f} "
        f"test rows, below MIN_TEST_ROWS={handler.MIN_TEST_ROWS} — the data gate "
        "would pass and the test-set gate could never pass"
    )

    required_failure_rate = handler.MIN_TEST_RISKY / handler.MIN_TEST_ROWS
    assert required_failure_rate <= 0.20, (
        f"the floors demand a {required_failure_rate:.0%} deploy-failure rate in "
        "the recent window; real teams run nearer 5-15%, so no healthy tenant "
        "would ever be promoted"
    )


def test_a_healthy_history_still_promotes():
    """The floors must not have made promotion unreachable in practice."""
    df = _synthetic_history(400, risky_every=3)
    model, X_test, y_test = handler.train_model(df)
    passed, metrics = handler.validate_model(model, None, X_test, y_test, len(df))

    assert passed is True, f"failed: {[k for k, v in metrics['checks'].items() if not v]}"
    assert metrics["test_rows"] >= handler.MIN_TEST_ROWS
    assert metrics["split"] == "time-ordered tail"


def test_metrics_record_the_size_the_score_was_measured_on():
    # precision 1.0 on 30 rows and on 3 rows are not the same claim; a promotion
    # has to be auditable after the fact.
    df = _synthetic_history(200)
    model, X_test, y_test = handler.train_model(df)
    _, metrics = handler.validate_model(model, None, X_test, y_test, len(df))
    assert metrics["test_rows"] == len(y_test)
    assert metrics["test_risky_rows"] == int((y_test == 1).sum())
