"""
ApacheJIT — real labelled commits to train the base model on.

WHY
    The base model was trained on 3000 rows invented by a hand-written rule in
    ml/generate_synthetic.py, so it learned that rule and nothing else. Its
    reported metrics measure how faithfully it copied the rule, not whether it
    predicts anything. A 19-line reimplementation of the generator scores AUC
    0.968 against the forest's 0.974.

    ApacheJIT is 106,674 real commits from 15 Apache projects (2003-2019),
    labelled bug-inducing or clean, released CC-BY-4.0. Trained on it with a
    time-ordered split, the same RandomForest reaches AUC-ROC 0.896 on commits
    from *later years it never saw* — which is an actual claim about prediction.

    Sarah Keshavarz and Mehdi Nagappan, "ApacheJIT: A Large Dataset for
    Just-In-Time Defect Prediction", MSR 2022. https://doi.org/10.5281/zenodo.5907002

WHAT THIS DATASET IS NOT
    Its label is "this commit was later identified as introducing a bug",
    derived by the SZZ algorithm. SafeShip predicts "this deploy will break
    production". Those are related but not the same question, and SZZ is known
    to mislabel when commits are tangled.

    So this is a better *prior* than invented data — it teaches genuine
    relationships between change shape and risk — but it is not a substitute for
    a tenant's own deploy outcomes. It is what the base model should be, and
    per-tenant retraining remains where real accuracy comes from.

TWO FEATURES CANNOT BE LEARNED FROM IT
    test_pass_rate and build_time_delta are properties of a CI run, and a commit
    dataset has no CI runs in it. They arrive as NaN, get imputed, and the model
    assigns them zero importance — correctly, because this data says nothing
    about them. They only become useful once real pipelines report them.
"""
from __future__ import annotations

import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import FEATURE_COLUMNS  # noqa: E402

ZENODO_URL = ("https://zenodo.org/records/5907002/files/"
              "apachejit_total.csv?download=1")
CITATION = ("Keshavarz & Nagappan, ApacheJIT (MSR 2022), CC-BY-4.0 — "
            "https://doi.org/10.5281/zenodo.5907002")

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
LOCAL_CSV = os.path.join(DATA_DIR, "apachejit_total.csv")

#: Last year of training data. The dataset's own split is 2003-2016 train,
#: 2017-2019 test, and keeping it makes results comparable with the paper.
TRAIN_THROUGH_YEAR = 2016

#: How many previous commits in the same project feed recent_failure_rate.
#: Ten matches the phrasing SafeShip already shows users ("40% of last 10
#: builds failed").
FAILURE_WINDOW = 10


def download(path: str = LOCAL_CSV, url: str = ZENODO_URL) -> str:
    """Fetch the dataset once and cache it. ~16MB."""
    if os.path.isfile(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[apachejit] downloading {url}")
    tmp = path + ".part"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, path)
    print(f"[apachejit] cached at {path} ({os.path.getsize(path)/1e6:.1f} MB)")
    return path


def load_raw(path: str = LOCAL_CSV) -> pd.DataFrame:
    df = pd.read_csv(download(path))
    missing = {"la", "ld", "nf", "fix", "aexp", "age", "buggy",
               "author_date", "project", "year"} - set(df.columns)
    if missing:
        raise ValueError(f"ApacheJIT is missing expected columns: {sorted(missing)}")
    return df


def to_safeship_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map ApacheJIT's Kamei-style metrics onto SafeShip's ten features.

    Eight map directly or derive; the two CI-run features cannot exist here and
    are left NaN for the caller to impute, exactly as /score does when a
    pipeline cannot measure them.
    """
    ts = pd.to_datetime(df["author_date"], unit="s")

    out = pd.DataFrame(index=df.index)
    # diff_size is churn — insertions plus deletions — which is what SafeShip
    # measures too. A 500-line deletion is not a small change.
    out["diff_size"] = df["la"] + df["ld"]
    out["files_changed"] = df["nf"]
    out["hour_of_day"] = ts.dt.hour
    out["day_of_week"] = ts.dt.dayofweek
    out["recent_failure_rate"] = _rolling_failure_rate(df)
    out["test_pass_rate"] = np.nan          # no CI runs in a commit dataset
    out["is_hotfix"] = df["fix"].astype(int)
    out["deployer_exp"] = df["aexp"]        # author experience = past commits
    out["days_since_deploy"] = df["age"]    # days since the files last changed
    out["build_time_delta"] = np.nan        # no CI runs in a commit dataset

    return out[FEATURE_COLUMNS]


def _rolling_failure_rate(df: pd.DataFrame) -> pd.Series:
    """
    Share of the project's previous FAILURE_WINDOW commits that were buggy.

    shift() before rolling is the whole point: without it a commit's own label
    is inside its own feature, the model learns to read the answer, and the
    score looks superb right up until it meets a commit whose label nobody
    knows yet. That is label leakage, and it is the easiest way to build a
    model that benchmarks beautifully and is worthless.
    """
    ordered = df.sort_values("author_date")
    rate = (ordered.groupby("project")["buggy"]
            .transform(lambda s: s.shift().rolling(FAILURE_WINDOW, min_periods=1).mean()))
    return rate.reindex(df.index)


def load(path: str = LOCAL_CSV):
    """
    Returns (X, y, year) with X in FEATURE_COLUMNS order.

    `year` comes back so callers can split by time rather than at random.
    """
    raw = load_raw(path)
    return to_safeship_features(raw), raw["buggy"].astype(int), raw["year"]


def time_split(X, y, year, through: int = TRAIN_THROUGH_YEAR):
    """
    Split by calendar year, never at random.

    A random split trains on commits from after the ones it is scored on. The
    deployed model only ever has the past, so a random split reports a number it
    can never achieve — and then the promotion gate believes that number.
    """
    train = year <= through
    return X[train], X[~train], y[train], y[~train]


def describe(path: str = LOCAL_CSV) -> dict:
    """Summary for logging and for the model metadata."""
    raw = load_raw(path)
    return {
        "rows": int(len(raw)),
        "projects": int(raw["project"].nunique()),
        "years": f"{int(raw['year'].min())}-{int(raw['year'].max())}",
        "base_rate": round(float(raw["buggy"].mean()), 4),
        "citation": CITATION,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
    X, y, year = load()
    Xtr, Xte, ytr, yte = time_split(X, y, year)
    print(f"train {len(Xtr):,} ({ytr.mean():.1%} buggy)   "
          f"test {len(Xte):,} ({yte.mean():.1%} buggy)")
