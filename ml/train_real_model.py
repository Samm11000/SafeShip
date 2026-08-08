"""
train_real_model.py — train the base model on real labelled commits.

    python ml/train_real_model.py                  # train, evaluate, save
    python ml/train_real_model.py --compare        # also score the synthetic model
    python ml/train_real_model.py --no-save        # evaluate only

WHAT CHANGED VERSUS train_base_model.py
    That script fits the model to ml/data/synthetic_builds.csv, whose labels are
    produced by a hand-written rule in generate_synthetic.py. The model learns
    the rule; the reported metrics measure how well it copied the rule. This one
    fits real commits with real bug labels, so the numbers are a claim about
    prediction rather than about reproduction.

    Three further differences, each from the JIT defect-prediction literature:

    1. SPLIT BY TIME, NOT AT RANDOM. A random split lets the model train on
       commits that happened after the ones it is scored on. The deployed model
       only ever has the past, so a random split reports a number it can never
       achieve in production.

    2. CALIBRATE THE OUTPUT. SafeShip presents a 0-100 score, which people read
       as a probability. Uncalibrated it is not one: the synthetic model says 9%
       where the true rate is 0.4%, and says 90% where the truth is 99.6%.
       Isotonic regression fixes the mapping without touching the ranking.

    3. REPORT AUC-PR, MCC AND CALIBRATION ERROR, not just precision. Under class
       imbalance, precision alone is dominated by the base rate; MCC is only
       high when all four cells of the confusion matrix are good.

    SMOTE is deliberately absent. The evidence is that rebalancing does not
    reliably improve MCC or AUC, and that threshold calibration does as well or
    better without inventing rows. class_weight="balanced" carries the load.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

sys.path.insert(0, os.path.join(os.path.dirname(HERE), "app"))

from datasets import apachejit  # noqa: E402
from features import FEATURE_COLUMNS, validate_model  # noqa: E402
from imputation import TRAINING_MEDIANS as SERVING_MEDIANS  # noqa: E402

MODEL_PATH = os.path.join(HERE, "data", "base_model_real.pkl")
META_PATH = os.path.join(HERE, "data", "base_metadata_real.json")

FOREST = dict(n_estimators=200, max_depth=8, min_samples_leaf=3,
              class_weight="balanced", random_state=42, n_jobs=-1)


def expected_calibration_error(p, y, bins: int = 10) -> float:
    """
    Mean gap between predicted probability and observed rate, per bin.

    This is the number that says whether "74/100" means anything. A model can
    rank perfectly (high AUC) and still be badly calibrated, and SafeShip shows
    users the number, not the ranking.
    """
    p, y = np.asarray(p), np.asarray(y)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.sum():
            total += abs(p[m].mean() - y[m].mean()) * m.sum()
    return float(total / len(p))


def evaluate(name: str, p, y, threshold: float = 0.70) -> dict:
    """Threshold 0.70 because that is where SafeShip's BLOCKED verdict starts."""
    pred = p > threshold
    return {
        "model": name,
        "auc_roc": round(float(roc_auc_score(y, p)), 4),
        "auc_pr": round(float(average_precision_score(y, p)), 4),
        "mcc": round(float(matthews_corrcoef(y, pred)), 4),
        "precision_at_block": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall_at_block": round(float(recall_score(y, pred, zero_division=0)), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "ece": round(expected_calibration_error(p, y), 4),
        "blocked_share": round(float(pred.mean()), 4),
    }


def reliability_table(p, y, bins: int = 5) -> str:
    """Predicted versus actual, which is the honest way to show calibration."""
    p, y = np.asarray(p), np.asarray(y)
    lines = ["  score band   n        predicted   actually buggy"]
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (p >= lo) & (p < hi if b < bins - 1 else p <= hi)
        if m.sum() < 10:
            continue
        lines.append(f"  {int(lo*100):3d}-{int(hi*100):3d}    {m.sum():6,}   "
                     f"{p[m].mean():8.1%}   {y[m].mean():13.1%}")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compare", action="store_true",
                    help="also evaluate the synthetic-trained model on this data")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    args = ap.parse_args()

    print("=" * 68)
    info = apachejit.describe()
    print(f"  dataset   {info['rows']:,} commits · {info['projects']} projects · {info['years']}")
    print(f"  base rate {info['base_rate']:.1%} bug-inducing")
    print(f"  source    {info['citation']}")
    print("=" * 68)

    X, y, year = apachejit.load()
    Xtr, Xte, ytr, yte = apachejit.time_split(X, y, year)
    print(f"\n  train {len(Xtr):,} ({ytr.mean():.1%} buggy)  ->  "
          f"test {len(Xte):,} ({yte.mean():.1%} buggy), later years only")

    # Impute from TRAINING medians only. Using the whole dataset's medians would
    # leak the test period's distribution into the training features.
    medians = Xtr.median()

    # A column that is entirely NaN has a NaN median, so fillna leaves it NaN —
    # and the model then trains on "this feature is always missing". At serve
    # time /score fills those same two features with the values in
    # app/imputation.py, so the model would meet a number where it had only ever
    # seen absence. That is train/serve skew, and it is silent. Fall back to the
    # serving medians so both sides agree.
    for col in FEATURE_COLUMNS:
        if pd.isna(medians[col]):
            medians[col] = SERVING_MEDIANS[col]
            print(f"  {col}: absent from this dataset, using the serving "
                  f"median {SERVING_MEDIANS[col]} so training matches /score")

    Xtr, Xte = Xtr.fillna(medians), Xte.fillna(medians)
    assert not Xtr.isna().any().any(), "features still missing after imputation"

    print("\n  fitting...")
    model = RandomForestClassifier(**FOREST).fit(Xtr, ytr)
    validate_model(model)

    results = [evaluate("raw", model.predict_proba(Xte)[:, 1], yte)]

    final = model
    if not args.no_calibrate:
        # cv="prefit" calibrates the already-fitted forest. Ideally the
        # calibration set is held out from fitting; here it reuses the training
        # period, which is the conservative direction — it cannot flatter the
        # test-period numbers, which are what we report.
        final = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
        final.fit(Xtr, ytr)
        results.append(evaluate("calibrated", final.predict_proba(Xte)[:, 1], yte))

    if args.compare and os.path.isfile(os.path.join(HERE, "data", "base_model.pkl")):
        synth = joblib.load(os.path.join(HERE, "data", "base_model.pkl"))
        results.append(evaluate("synthetic-trained",
                                synth.predict_proba(Xte)[:, 1], yte))

    print("\n  model               AUC-ROC  AUC-PR    MCC   Brier    ECE  blocked")
    for r in results:
        print(f"  {r['model']:18} {r['auc_roc']:7.3f} {r['auc_pr']:7.3f} "
              f"{r['mcc']:6.3f} {r['brier']:7.4f} {r['ece']:6.4f} "
              f"{r['blocked_share']:7.1%}")

    print("\n  calibration of the shipped model, on unseen later commits:")
    print(reliability_table(final.predict_proba(Xte)[:, 1], yte))

    print("\n  what the model actually uses:")
    for f, imp in sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
                         key=lambda t: -t[1]):
        note = ""
        if imp < 0.001:
            note = "  <- not present in this dataset; needs real pipeline data"
        print(f"    {f:22} {imp:.3f}{note}")

    if not args.no_save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(final, MODEL_PATH)
        chosen = results[1] if len(results) > 1 and not args.no_calibrate else results[0]
        meta = {
            "model_type": type(final).__name__,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "feature_columns": FEATURE_COLUMNS,
            "data_source": "apachejit",
            "dataset": info,
            "split": f"time-ordered, train <= {apachejit.TRAIN_THROUGH_YEAR}",
            "training_rows": int(len(Xtr)),
            "calibration": "none" if args.no_calibrate else "isotonic",
            "phase": "base",
            "training_medians": {k: round(float(v), 4) for k, v in medians.items()},
            "metrics": chosen,
            "all_metrics": results,
            "caveat": (
                "Labels are SZZ-derived bug-inducing commits, not deployment "
                "outcomes. This is a prior over change risk, not a model of a "
                "given tenant's deploys. test_pass_rate and build_time_delta "
                "are absent from this dataset and carry zero importance until "
                "real pipeline data supplies them."
            ),
        }
        with open(META_PATH, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        print(f"\n  saved {MODEL_PATH}")
        print(f"  saved {META_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
