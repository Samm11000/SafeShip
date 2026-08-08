"""
Tests for training the base model on real labelled commits.

The single most important test here is the label-leakage one. `recent_failure_rate`
is built from the outcomes of previous commits, and if the window includes the
current commit then its own label sits inside its own feature. The model then
learns to read the answer, benchmarks beautifully, and is worthless in
production — where the label does not exist yet. That failure is invisible in
every metric, which is exactly why it needs a test rather than a code review.

The dataset is ~16MB and downloaded on demand, so these skip when it is absent
rather than pulling it in CI.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "ml"))

from datasets import apachejit  # noqa: E402
from features import FEATURE_COLUMNS  # noqa: E402
from train_real_model import expected_calibration_error  # noqa: E402

HAS_DATA = os.path.isfile(apachejit.LOCAL_CSV)
needs_data = pytest.mark.skipif(
    not HAS_DATA,
    reason=f"ApacheJIT not downloaded — run: python {os.path.join('ml','datasets','apachejit.py')}")


# ── the mapping ──────────────────────────────────────────────────────────────

def _fake_raw():
    """A tiny ApacheJIT-shaped frame, so the mapping is testable without 16MB."""
    return pd.DataFrame({
        "commit_id": [f"c{i}" for i in range(6)],
        "project": ["apache/a"] * 3 + ["apache/b"] * 3,
        "buggy": [False, True, True, False, False, True],
        "fix": [False, False, True, False, True, False],
        "year": [2010, 2011, 2012, 2013, 2014, 2015],
        # 2010-01-01 09:00 UTC onwards, one day apart
        "author_date": [1262336400 + i * 86400 for i in range(6)],
        "la": [10, 200, 5, 60, 1, 400],
        "ld": [2, 50, 0, 6, 1, 100],
        "nf": [1, 12, 1, 3, 1, 25],
        "nd": [1, 4, 1, 2, 1, 8],
        "ns": [1, 2, 1, 1, 1, 3],
        "ent": [0.0, 2.5, 0.0, 1.2, 0.0, 3.1],
        "ndev": [1, 5, 2, 3, 1, 9],
        "age": [0.0, 12.5, 3.0, 40.0, 1.0, 88.0],
        "nuc": [1, 8, 2, 4, 1, 20],
        "aexp": [3, 150, 40, 7, 900, 12],
        "arexp": [3.0, 150.0, 40.0, 7.0, 900.0, 12.0],
        "asexp": [1.0, 90.0, 20.0, 5.0, 400.0, 6.0],
    })


def test_the_mapping_produces_exactly_the_feature_contract():
    X = apachejit.to_safeship_features(_fake_raw())
    assert list(X.columns) == list(FEATURE_COLUMNS)


def test_diff_size_is_churn_not_net_lines():
    # Same rule as safeship_ci: a 500-line deletion is not a small change.
    X = apachejit.to_safeship_features(_fake_raw())
    assert X["diff_size"].iloc[1] == 250      # la 200 + ld 50
    assert X["diff_size"].iloc[5] == 500


def test_clock_features_come_from_the_commit_timestamp():
    X = apachejit.to_safeship_features(_fake_raw())
    assert X["hour_of_day"].between(0, 23).all()
    assert X["day_of_week"].between(0, 6).all()


def test_the_two_ci_features_are_absent_rather_than_invented():
    """
    A commit dataset has no CI runs in it. Filling test_pass_rate with 1.0 to
    make the column non-null would be the original SafeShip bug — teaching the
    model that every historical build had perfect tests.
    """
    X = apachejit.to_safeship_features(_fake_raw())
    assert X["test_pass_rate"].isna().all()
    assert X["build_time_delta"].isna().all()


def test_a_fix_commit_maps_to_is_hotfix():
    X = apachejit.to_safeship_features(_fake_raw())
    assert list(X["is_hotfix"]) == [0, 0, 1, 0, 1, 0]


# ── THE LEAKAGE TEST ─────────────────────────────────────────────────────────

def test_recent_failure_rate_never_sees_its_own_commits_label():
    """
    The first commit of a project has no history, so its rate must be NaN — not
    its own label, and not the project average. If this ever returns a number
    for row 0, the window is including the present.
    """
    raw = _fake_raw()
    X = apachejit.to_safeship_features(raw)

    for project in raw["project"].unique():
        first = raw[raw["project"] == project].sort_values("author_date").index[0]
        assert pd.isna(X.loc[first, "recent_failure_rate"]), (
            f"{project}'s first commit has a failure rate before any history exists"
        )


def test_recent_failure_rate_is_built_only_from_earlier_commits():
    raw = _fake_raw()
    X = apachejit.to_safeship_features(raw)
    ordered = raw.sort_values("author_date")

    for pos, (idx, row) in enumerate(ordered.iterrows()):
        past = ordered[(ordered["project"] == row["project"])
                       & (ordered["author_date"] < row["author_date"])]
        value = X.loc[idx, "recent_failure_rate"]
        if past.empty:
            assert pd.isna(value)
        else:
            expected = past["buggy"].tail(apachejit.FAILURE_WINDOW).mean()
            assert value == pytest.approx(expected), (
                f"row {idx}: expected the mean of {len(past)} earlier commits"
            )


def test_history_does_not_bleed_between_projects():
    """One project's failures say nothing about another's."""
    raw = _fake_raw()
    X = apachejit.to_safeship_features(raw)
    # apache/b's first commit follows three buggy-ish apache/a commits.
    first_b = raw[raw["project"] == "apache/b"].index[0]
    assert pd.isna(X.loc[first_b, "recent_failure_rate"])


def test_a_perfect_leak_would_be_caught():
    """
    Guards the guard: if the implementation dropped shift() and the window
    included the current row, the tests above must fail rather than pass
    quietly. This builds that broken version and checks it disagrees.
    """
    raw = _fake_raw().sort_values("author_date")
    leaky = (raw.groupby("project")["buggy"]
             .transform(lambda s: s.rolling(apachejit.FAILURE_WINDOW,
                                            min_periods=1).mean()))
    correct = apachejit._rolling_failure_rate(raw)
    assert not leaky.equals(correct), (
        "the leaky and correct implementations produce identical output, so "
        "these tests could not tell them apart"
    )
    assert leaky.notna().all(), "the leaky version has no NaN — that is the tell"


# ── the split ────────────────────────────────────────────────────────────────

def test_the_split_is_by_time_with_no_future_in_training():
    X = pd.DataFrame({c: range(6) for c in FEATURE_COLUMNS})
    y = pd.Series([0, 1, 0, 1, 0, 1])
    year = pd.Series([2014, 2015, 2016, 2017, 2018, 2019])

    Xtr, Xte, ytr, yte = apachejit.time_split(X, y, year, through=2016)
    assert len(Xtr) == 3 and len(Xte) == 3
    assert year[Xtr.index].max() < year[Xte.index].min(), (
        "a training commit is newer than a test commit — that is the leak a "
        "random split introduces"
    )


# ── calibration maths ────────────────────────────────────────────────────────

def test_a_perfectly_calibrated_model_has_zero_error():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20_000)
    y = (rng.uniform(0, 1, 20_000) < p).astype(int)
    assert expected_calibration_error(p, y) < 0.02


def test_a_confidently_wrong_model_has_high_error():
    p = np.full(1000, 0.95)
    y = np.zeros(1000, dtype=int)      # claims 95%, never happens
    assert expected_calibration_error(p, y) > 0.9


def test_calibration_error_is_not_fooled_by_good_ranking():
    """
    A model can rank perfectly and still be badly calibrated. SafeShip shows the
    number, not the ranking, so AUC alone cannot tell us the number means
    anything.
    """
    from sklearn.metrics import roc_auc_score

    y = np.array([0] * 500 + [1] * 500)
    p = np.concatenate([np.linspace(0.90, 0.94, 500),
                        np.linspace(0.95, 0.99, 500)])
    assert roc_auc_score(y, p) == 1.0, "ranking is perfect"
    assert expected_calibration_error(p, y) > 0.4, "yet the probabilities are nonsense"


# ── against the real file, when present ──────────────────────────────────────

@needs_data
def test_the_real_dataset_matches_its_published_shape():
    info = apachejit.describe()
    assert info["rows"] == 106_674
    assert info["projects"] == 15
    assert 0.20 < info["base_rate"] < 0.30


@needs_data
def test_the_real_dataset_maps_cleanly():
    X, y, year = apachejit.load()
    assert list(X.columns) == list(FEATURE_COLUMNS)
    assert len(X) == len(y) == len(year)
    # Eight of ten are genuinely populated; the two CI features are not.
    populated = [c for c in FEATURE_COLUMNS if X[c].notna().any()]
    assert len(populated) == 8
    assert set(FEATURE_COLUMNS) - set(populated) == {"test_pass_rate", "build_time_delta"}


@needs_data
def test_the_real_split_holds_out_later_years_only():
    X, y, year = apachejit.load()
    Xtr, Xte, ytr, yte = apachejit.time_split(X, y, year)
    assert year[Xtr.index].max() <= apachejit.TRAIN_THROUGH_YEAR
    assert year[Xte.index].min() > apachejit.TRAIN_THROUGH_YEAR
    assert len(Xte) > 10_000, "the held-out period should be substantial"
