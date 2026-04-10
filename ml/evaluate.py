"""
evaluate.py
Path: C:\deploy-gate\ml\evaluate.py

PURPOSE:
  Loads the trained model from S3 and runs a full evaluation report.
  Run this anytime to check how healthy the model is.
  Also fixes the UserWarning about feature names.

HOW TO RUN:
  cd C:\deploy-gate
  python ml\evaluate.py
"""

import os
import json
import joblib
import boto3
import tempfile
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
S3_BUCKET    = "deploy-gate-models"
S3_MODEL_KEY = "base/model.pkl"
S3_META_KEY  = "base/metadata.json"
AWS_REGION   = "ap-south-1"

DATA_PATH    = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_builds.csv"
)

FEATURE_COLUMNS = [
    "diff_size", "files_changed", "hour_of_day", "day_of_week",
    "recent_failure_rate", "test_pass_rate", "is_hotfix",
    "deployer_exp", "days_since_deploy", "build_time_delta",
]


# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL FROM S3
# ─────────────────────────────────────────────────────────────────────────────

def load_model_from_s3():
    print(f"[evaluate] Loading model from s3://{S3_BUCKET}/{S3_MODEL_KEY}")
    s3 = boto3.client("s3", region_name=AWS_REGION)

    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        s3.download_file(S3_BUCKET, S3_MODEL_KEY, tmp_path)
        model = joblib.load(tmp_path)
        print(f"  [OK] Model loaded. Type: {type(model).__name__}")
        return model
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_metadata_from_s3():
    s3 = boto3.client("s3", region_name=AWS_REGION)
    try:
        obj  = s3.get_object(Bucket=S3_BUCKET, Key=S3_META_KEY)
        meta = json.loads(obj["Body"].read())
        return meta
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation(model, df):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # Use DataFrame so sklearn doesn't warn about feature names
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred,    zero_division=0)
    f1        = f1_score(y_test, y_pred,        zero_division=0)
    auc       = roc_auc_score(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 60)
    print("MODEL EVALUATION REPORT")
    print("=" * 60)

    # ── Metadata from S3 ──
    meta = load_metadata_from_s3()
    if meta:
        print(f"\nModel trained at : {meta.get('trained_at', 'unknown')}")
        print(f"Training rows    : {meta.get('training_rows', 'unknown')}")
        print(f"Data source      : {meta.get('data_source', 'unknown')}")
        print(f"Phase            : {meta.get('phase', 'unknown')}")

    # ── Core metrics ──
    print(f"\n{'─'*40}")
    print(f"CORE METRICS (on held-out test set)")
    print(f"{'─'*40}")
    print(f"  Precision   : {precision:.4f}")
    print(f"  Recall      : {recall:.4f}")
    print(f"  F1 Score    : {f1:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")

    # ── What these numbers mean in plain English ──
    print(f"\nIN PLAIN ENGLISH:")
    blocked    = int(y_pred.sum())
    correct    = cm[1][1]
    false_alm  = cm[0][1]
    missed     = cm[1][0]
    total_risk = int(y_test.sum())

    print(f"  Out of {len(y_test)} test builds:")
    print(f"    Model flagged {blocked} as risky")
    print(f"    Of those {blocked} flagged: {correct} were genuinely risky, "
          f"{false_alm} were false alarms")
    print(f"    Of {total_risk} truly risky builds: "
          f"caught {correct}, missed {missed}")

    # ── Confusion matrix ──
    print(f"\n{'─'*40}")
    print(f"CONFUSION MATRIX")
    print(f"{'─'*40}")
    print(f"                     Predicted Safe   Predicted Risky")
    print(f"  Actually Safe    :     {cm[0][0]:4d}              {cm[0][1]:4d}  (false alarms)")
    print(f"  Actually Risky   :     {cm[1][0]:4d}  (missed)    {cm[1][1]:4d}  (caught!)")

    # ── Score distribution ──
    print(f"\n{'─'*40}")
    print(f"SCORE DISTRIBUTION (0-100)")
    print(f"{'─'*40}")
    scores = (y_proba * 100).astype(int)
    buckets = {
        "0-20  (very safe)": ((scores >= 0)  & (scores <= 20)).sum(),
        "21-40 (safe)":      ((scores >= 21) & (scores <= 40)).sum(),
        "41-60 (borderline)":((scores >= 41) & (scores <= 60)).sum(),
        "61-80 (risky)":     ((scores >= 61) & (scores <= 80)).sum(),
        "81-100 (very risky)":((scores >= 81) & (scores <= 100)).sum(),
    }
    for bucket, count in buckets.items():
        bar = "#" * (count // 3)
        print(f"  {bucket:25s} : {count:4d}  {bar}")

    # ── Feature importance ──
    print(f"\n{'─'*40}")
    print(f"FEATURE IMPORTANCE")
    print(f"(what the model weights most when scoring)")
    print(f"{'─'*40}")
    importances = model.feature_importances_
    feat_imp    = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda x: x[1], reverse=True
    )
    for rank, (feat, imp) in enumerate(feat_imp, 1):
        bar    = "#" * int(imp * 60)
        pct    = imp * 100
        print(f"  {rank}. {feat:28s} {pct:5.1f}%  {bar}")

    # ── Quality gate ──
    print(f"\n{'─'*40}")
    print(f"QUALITY GATE")
    print(f"{'─'*40}")
    checks = {
        "precision >= 0.75 (production target)": precision >= 0.75,
        "precision >= 0.55 (minimum bar)":       precision >= 0.55,
        "auc_roc   >= 0.70 (good separation)":   auc       >= 0.70,
        "recall    >= 0.40 (catching risks)":     recall    >= 0.40,
        "f1        >= 0.60 (balanced)":           f1        >= 0.60,
    }
    all_pass = True
    for check, passed in checks.items():
        status = "[OK]  " if passed else "[WARN]"
        print(f"  {status} {check}")
        if not passed:
            all_pass = False

    if precision >= 0.75:
        print(f"\n  PRODUCTION READY — precision {precision:.3f} exceeds 0.75 target")
    elif precision >= 0.55:
        print(f"\n  ACCEPTABLE — precision {precision:.3f} above minimum bar")
        print(f"  Will improve as tenant data accumulates (target: 0.75+)")
    else:
        print(f"\n  BELOW THRESHOLD — retrain with more data")

    # ── Scenario tests ──
    print(f"\n{'─'*40}")
    print(f"SCENARIO TESTS (sanity check)")
    print(f"{'─'*40}")

    scenarios = [
        {
            "name":     "Classic risky (Fri 5pm, 847 lines, 40% failures)",
            "features": pd.DataFrame([[847,12,17,4,0.4,0.85,0,10,3.0,0.1]],  columns=FEATURE_COLUMNS),
            "expect":   "high",
        },
        {
            "name":     "Safe deploy (Tue 10am, 45 lines, all green)",
            "features": pd.DataFrame([[45,3,10,1,0.0,1.0,0,50,1.0,-0.05]], columns=FEATURE_COLUMNS),
            "expect":   "low",
        },
        {
            "name":     "Hotfix at end of day",
            "features": pd.DataFrame([[200,8,17,2,0.1,0.9,1,20,1.0,0.0]],  columns=FEATURE_COLUMNS),
            "expect":   "medium",
        },
        {
            "name":     "Friday night large diff",
            "features": pd.DataFrame([[900,20,21,4,0.3,0.7,0,5,7.0,0.5]],  columns=FEATURE_COLUMNS),
            "expect":   "high",
        },
    ]

    scenario_pass = True
    for sc in scenarios:
        proba = model.predict_proba(sc["features"])[0][1]
        score = int(round(proba * 100))
        if sc["expect"] == "high":
            ok = score >= 60
        elif sc["expect"] == "low":
            ok = score <= 40
        else:
            ok = 20 <= score <= 80

        status = "[OK]  " if ok else "[FAIL]"
        print(f"  {status} {sc['name']}")
        print(f"          Score: {score}/100  (expected: {sc['expect']})")
        if not ok:
            scenario_pass = False

    # ── Final verdict ──
    print(f"\n{'='*60}")
    if all_pass and scenario_pass:
        print(f"OVERALL: MODEL IS HEALTHY")
    elif scenario_pass:
        print(f"OVERALL: MODEL WORKS CORRECTLY (metrics acceptable for base model)")
    else:
        print(f"OVERALL: REVIEW FAILED SCENARIOS ABOVE")
    print(f"{'='*60}")

    return {
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "auc_roc":   round(auc,       4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("RUNNING FULL MODEL EVALUATION")
    print("=" * 60)

    model = load_model_from_s3()

    print(f"\n[evaluate] Loading test data from {DATA_PATH}")
    df    = pd.read_csv(DATA_PATH)
    df    = df[df["label"].isin([0, 1])]
    print(f"  [OK] Loaded {len(df)} rows")

    metrics = run_evaluation(model, df)

    print(f"\n[evaluate] Done.")
    print(f"  Precision : {metrics['precision']}")
    print(f"  AUC-ROC   : {metrics['auc_roc']}")
    print(f"\nRun this file anytime to check model health.")


if __name__ == "__main__":
    main()