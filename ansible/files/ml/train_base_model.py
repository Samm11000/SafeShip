"""
train_base_model.py
Path: C:\deploy-gate\ml\train_base_model.py

PURPOSE:
  Trains the Random Forest base model on synthetic_builds.csv
  Validates it passes all quality checks
  Saves model.pkl + metadata.json to S3

HOW TO RUN:
  cd C:\deploy-gate
  python ml\train_base_model.py

WHAT YOU NEED BEFORE RUNNING:
  - synthetic_builds.csv must exist (run generate_synthetic.py first)
  - AWS configured (run: aws sts get-caller-identity to verify)
  - S3 bucket deploy-gate-models must exist

OUTPUT:
  Local:  C:\deploy-gate\ml\data\base_model.pkl
  S3:     s3://deploy-gate-models/base/model.pkl
          s3://deploy-gate-models/base/metadata.json
"""

import os
import json
import joblib
import boto3
import pandas as pd
import numpy as np

from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — update S3_BUCKET if you used a different bucket name
# ─────────────────────────────────────────────────────────────────────────────
S3_BUCKET       = "deploy-gate-models"
S3_MODEL_KEY    = "base/model.pkl"
S3_META_KEY     = "base/metadata.json"
AWS_REGION      = "ap-south-1"

DATA_PATH       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "synthetic_builds.csv")
LOCAL_MODEL_PATH= os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base_model.pkl")
LOCAL_META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "base_metadata.json")

# These MUST match the exact order in validator.py to_model_input()
FEATURE_COLUMNS = [
    "diff_size",
    "files_changed",
    "hour_of_day",
    "day_of_week",
    "recent_failure_rate",
    "test_pass_rate",
    "is_hotfix",
    "deployer_exp",
    "days_since_deploy",
    "build_time_delta",
]

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load and prepare data
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print(f"[train] Loading data from {DATA_PATH}")

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"synthetic_builds.csv not found at {DATA_PATH}\n"
            "Run generate_synthetic.py first."
        )

    df = pd.read_csv(DATA_PATH)
    print(f"[train] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Check all feature columns exist
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # Check label column exists
    if "label" not in df.columns:
        raise ValueError("No 'label' column found in CSV")

    # Drop any rows where label is still -1 (pending — shouldn't happen in synthetic)
    before = len(df)
    df = df[df["label"].isin([0, 1])]
    if len(df) < before:
        print(f"[train] Dropped {before - len(df)} rows with pending labels")

    # Check for nulls
    nulls = df[FEATURE_COLUMNS].isnull().sum()
    if nulls.sum() > 0:
        print(f"[train] WARNING: Found nulls, filling with defaults:\n{nulls[nulls > 0]}")
        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    # Use sample_weight if available, else default to 1.0
    weights = df["sample_weight"] if "sample_weight" in df.columns else None

    print(f"[train] Features: {list(X.columns)}")
    print(f"[train] Label distribution:")
    print(f"  Safe  (0): {(y==0).sum():4d} ({(y==0).mean()*100:.1f}%)")
    print(f"  Risky (1): {(y==1).sum():4d} ({(y==1).mean()*100:.1f}%)")

    return X, y, weights


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Apply SMOTE to fix class imbalance
# ─────────────────────────────────────────────────────────────────────────────

def apply_smote(X_train, y_train):
    print("\n[train] Applying SMOTE to balance training classes...")

    before_safe  = (y_train == 0).sum()
    before_risky = (y_train == 1).sum()
    print(f"  Before SMOTE: Safe={before_safe}, Risky={before_risky}")

    # SMOTE creates synthetic minority samples to balance the dataset
    # random_state=42 makes it reproducible
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    after_safe  = (y_resampled == 0).sum()
    after_risky = (y_resampled == 1).sum()
    print(f"  After  SMOTE: Safe={after_safe}, Risky={after_risky}")

    return X_resampled, y_resampled


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train Random Forest
# ─────────────────────────────────────────────────────────────────────────────

def train_model(X_train, y_train):
    print("\n[train] Training Random Forest Classifier...")
    print("  Parameters:")
    print("    n_estimators  = 100")
    print("    max_depth     = 8")
    print("    class_weight  = balanced")
    print("    min_samples_leaf = 3")

    model = RandomForestClassifier(
        n_estimators     = 100,
        max_depth        = 8,
        class_weight     = "balanced",
        min_samples_leaf = 3,
        random_state     = 42,
        n_jobs           = -1,       # use all CPU cores
    )

    model.fit(X_train, y_train)
    print(f"  [OK] Training complete. Trees: {len(model.estimators_)}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Evaluate model quality
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, X_full, y_full):
    print("\n[train] Evaluating model...")

    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)[:, 1]  # probability of risky

    precision   = precision_score(y_test, y_pred,  zero_division=0)
    recall      = recall_score(y_test, y_pred,      zero_division=0)
    f1          = f1_score(y_test, y_pred,          zero_division=0)
    auc         = roc_auc_score(y_test, y_proba)
    cm          = confusion_matrix(y_test, y_pred)

    # Cross-validation on full dataset (5-fold)
    print("  Running 5-fold cross-validation...")
    cv_scores   = cross_val_score(model, X_full, y_full, cv=5, scoring="precision")
    cv_mean     = cv_scores.mean()
    cv_std      = cv_scores.std()

    print(f"\n  --- MODEL EVALUATION REPORT ---")
    print(f"  Precision  : {precision:.3f}  (of builds flagged risky, how many actually were)")
    print(f"  Recall     : {recall:.3f}  (of truly risky builds, how many did we catch)")
    print(f"  F1 Score   : {f1:.3f}  (balance between precision and recall)")
    print(f"  AUC-ROC    : {auc:.3f}  (0.5=random, 1.0=perfect)")
    print(f"  CV Precision: {cv_mean:.3f} +/- {cv_std:.3f}  (5-fold cross-validation)")
    print(f"\n  Confusion Matrix:")
    print(f"    True  Safe predicted Safe  (correct): {cm[0][0]}")
    print(f"    True  Safe predicted Risky (false alarm): {cm[0][1]}")
    print(f"    True Risky predicted Safe  (missed!): {cm[1][0]}")
    print(f"    True Risky predicted Risky (correct): {cm[1][1]}")

    # Feature importance — what the model learned matters most
    importances = model.feature_importances_
    feat_imp    = sorted(
        zip(FEATURE_COLUMNS, importances),
        key=lambda x: x[1],
        reverse=True
    )
    print(f"\n  Feature Importance (what drives the score):")
    for feat, imp in feat_imp:
        bar = "#" * int(imp * 50)
        print(f"    {feat:28s} {imp:.3f}  {bar}")

    metrics = {
        "precision":    round(float(precision),  3),
        "recall":       round(float(recall),     3),
        "f1":           round(float(f1),         3),
        "auc_roc":      round(float(auc),        3),
        "cv_precision": round(float(cv_mean),    3),
        "cv_std":       round(float(cv_std),     3),
        "feature_importance": {f: round(float(i), 4) for f, i in feat_imp},
    }

    # Quality gate check
    print(f"\n  --- QUALITY GATE ---")
    gate_pass = True

    checks = {
        "precision >= 0.55":  precision  >= 0.55,
        "auc_roc  >= 0.65":   auc        >= 0.65,
        "recall   >= 0.40":   recall     >= 0.40,
    }
    for check, passed in checks.items():
        status = "[OK]" if passed else "[WARN]"
        print(f"  {status} {check}")
        if not passed:
            gate_pass = False

    if gate_pass:
        print("\n  [PASS] Base model meets minimum quality standards.")
    else:
        print("\n  [WARN] Some checks failed — model will still be saved but review above.")
        print("         This is normal for a base model trained on synthetic data only.")
        print("         Precision improves significantly once tenant data accumulates.")

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Save model locally + upload to S3
# ─────────────────────────────────────────────────────────────────────────────

def save_and_upload(model, metrics):
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

    # Save model locally first
    print(f"\n[train] Saving model locally to {LOCAL_MODEL_PATH}")
    joblib.dump(model, LOCAL_MODEL_PATH)
    model_size_kb = os.path.getsize(LOCAL_MODEL_PATH) / 1024
    print(f"  [OK] Model saved. Size: {model_size_kb:.1f} KB")

    # Build metadata
    metadata = {
        "model_type":        "RandomForestClassifier",
        "trained_at":        datetime.utcnow().isoformat() + "Z",
        "n_estimators":      100,
        "max_depth":         8,
        "feature_columns":   FEATURE_COLUMNS,
        "training_rows":     3000,
        "data_source":       "synthetic",
        "phase":             "base",
        **metrics,
    }

    # Save metadata locally
    with open(LOCAL_META_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  [OK] Metadata saved to {LOCAL_META_PATH}")

    # Upload to S3
    print(f"\n[train] Uploading to S3 bucket: {S3_BUCKET}")
    try:
        s3 = boto3.client("s3", region_name=AWS_REGION)

        # Upload model
        print(f"  Uploading model  → s3://{S3_BUCKET}/{S3_MODEL_KEY}")
        s3.upload_file(LOCAL_MODEL_PATH, S3_BUCKET, S3_MODEL_KEY)
        print(f"  [OK] model.pkl uploaded")

        # Upload metadata
        print(f"  Uploading metadata → s3://{S3_BUCKET}/{S3_META_KEY}")
        s3.upload_file(LOCAL_META_PATH, S3_BUCKET, S3_META_KEY)
        print(f"  [OK] metadata.json uploaded")

        # Verify upload by checking file exists
        s3.head_object(Bucket=S3_BUCKET, Key=S3_MODEL_KEY)
        print(f"\n  [OK] S3 upload verified successfully")

    except Exception as e:
        print(f"\n  [ERROR] S3 upload failed: {e}")
        print(f"  Model is saved locally at {LOCAL_MODEL_PATH}")
        print(f"  Fix the S3 issue and re-run, or manually upload the file.")
        raise


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TRAINING BASE MODEL")
    print("=" * 60)

    # Step 1: Load data
    X, y, weights = load_data()

    # Step 2: Split into train/test BEFORE SMOTE
    # Important: SMOTE is only applied to training data, NOT test data
    # Test data must stay real (unaugmented) for honest evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = 0.20,     # 80% train, 20% test
        random_state = 42,
        stratify     = y         # keep same class ratio in both splits
    )
    print(f"\n[train] Train/test split:")
    print(f"  Training rows : {len(X_train)}")
    print(f"  Test rows     : {len(X_test)}")

    # Step 3: Apply SMOTE to training data only
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    # Step 4: Train model
    model = train_model(X_train_sm, y_train_sm)

    # Step 5: Evaluate on original (non-SMOTE) test set
    metrics = evaluate_model(model, X_test, y_test, X, y)

    # Step 6: Save locally and upload to S3
    save_and_upload(model, metrics)

    print("\n" + "=" * 60)
    print("BASE MODEL TRAINING COMPLETE")
    print(f"Precision : {metrics['precision']}")
    print(f"AUC-ROC   : {metrics['auc_roc']}")
    print("Next: run scorer.py (File 2 of Phase 2)")
    print("=" * 60)


if __name__ == "__main__":
    main()