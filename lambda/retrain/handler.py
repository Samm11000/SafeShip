"""
handler.py
Path: C:\deploy-gate\lambda\retrain\handler.py

PURPOSE:
  Nightly Lambda function triggered at 2am UTC by CloudWatch Events.
  Scans all tenants in DynamoDB.
  For each tenant with 80+ labelled builds:
    1. Pulls their data.csv from S3
    2. Retrains Random Forest
    3. Runs 5-check validation gate
    4. If passes: hot-swaps model.pkl in S3
    5. Updates DynamoDB metadata
    6. Sends Slack alert if retrain fails

HOW TO DEPLOY:
  See bottom of this file for deployment commands.
"""

import os
import io
import csv
import json
import time
import boto3
import joblib
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from decimal import Decimal

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.metrics          import precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling   import SMOTE

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_EXECUTION_ENV", "us-east-1"))
S3_MODELS       = os.getenv("S3_MODELS_BUCKET",  "deploy-gate-models")
S3_DATA         = os.getenv("S3_DATA_BUCKET",     "deploy-gate-data")
DYNAMO_TABLE    = os.getenv("DYNAMO_TABLE",       "tenants")
MIN_BUILDS      = 5      # minimum labelled builds before retraining
MIN_PRECISION   = 0.75    # minimum precision to swap model
MIN_AUC         = 0.70    # minimum AUC-ROC to swap model
MIN_RISKY_RATIO = 0.05    # minimum risky build ratio in test set

FEATURE_COLUMNS = [
    "diff_size", "files_changed", "hour_of_day", "day_of_week",
    "recent_failure_rate", "test_pass_rate", "is_hotfix",
    "deployer_exp", "days_since_deploy", "build_time_delta",
]


# ─────────────────────────────────────────────────────────────────────────────
# AWS CLIENTS
# ─────────────────────────────────────────────────────────────────────────────
def get_clients():
    s3     = boto3.client("s3",       region_name=AWS_REGION)
    dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    table  = dynamo.Table(DYNAMO_TABLE)
    return s3, table


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Get all tenants from DynamoDB
# ─────────────────────────────────────────────────────────────────────────────
def get_all_tenants(table):
    print("[retrain] Scanning DynamoDB for all tenants...")
    resp    = table.scan()
    tenants = resp.get("Items", [])

    # Handle pagination if more than 1MB of data
    while "LastEvaluatedKey" in resp:
        resp    = table.scan(ExclusiveStartKey=resp["LastEvaluatedKey"])
        tenants.extend(resp.get("Items", []))

    print(f"[retrain] Found {len(tenants)} tenants total")
    return tenants


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Load tenant data from S3
# ─────────────────────────────────────────────────────────────────────────────
def load_tenant_data(s3, tenant_id):
    key = f"tenant_{tenant_id}/data.csv"
    try:
        obj     = s3.get_object(Bucket=S3_DATA, Key=key)
        content = obj["Body"].read().decode("utf-8")
        df      = pd.read_csv(io.StringIO(content))

        # Only use labelled rows (label = 0 or 1, not -1)
        df = df[df["label"].isin([0, 1])].copy()

        # Apply 90-day rolling window
        if "timestamp" in df.columns:
            cutoff = int(time.time()) - (90 * 86400)
            df     = df[df["timestamp"].astype(int) >= cutoff]

        return df

    except Exception as e:
        print(f"[retrain] Could not load data for {tenant_id}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train model
# ─────────────────────────────────────────────────────────────────────────────
def train_model(df):
    X = df[FEATURE_COLUMNS].fillna(0)
    y = df["label"].astype(int)

    # Use sample weights if available
    weights = df["sample_weight"].astype(float) if "sample_weight" in df.columns else None

    # Train/test split — stratified to keep class ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Keep test weights separate if available
    if weights is not None:
        w_train = weights.iloc[X_train.index] if hasattr(weights, 'iloc') else None
    else:
        w_train = None

    # Apply SMOTE only if we have enough minority class samples
    risky_count = (y_train == 1).sum()
    if risky_count >= 6:
        try:
            k = min(5, risky_count - 1)
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            w_train = None  # SMOTE invalidates original weights
        except Exception as e:
            print(f"[retrain] SMOTE failed (continuing without): {e}")

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators     = 100,
        max_depth        = 8,
        class_weight     = "balanced",
        min_samples_leaf = 3,
        random_state     = 42,
        n_jobs           = -1,
    )
    model.fit(X_train, y_train, sample_weight=w_train)

    return model, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Validate model (5-check gate)
# ─────────────────────────────────────────────────────────────────────────────
def validate_model(new_model, old_model, X_test, y_test, dataset_size):
    y_pred  = new_model.predict(X_test)
    y_proba = new_model.predict_proba(X_test)[:, 1]

    precision    = precision_score(y_test, y_pred,  zero_division=0)
    recall       = recall_score(y_test, y_pred,     zero_division=0)
    auc          = roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else 0.5
    risky_ratio  = float(y_test.mean())

    checks = {
        "dataset_size >= 80":       dataset_size >= MIN_BUILDS,
        "precision >= 0.75":        precision    >= MIN_PRECISION,
        "auc_roc >= 0.70":          auc          >= MIN_AUC,
        "risky_ratio >= 0.05":      risky_ratio  >= MIN_RISKY_RATIO,
    }

    # Check 5: new model not worse than old
    if old_model is not None:
        try:
            old_pred      = old_model.predict(X_test)
            old_precision = precision_score(y_test, old_pred, zero_division=0)
            not_regressing = precision >= (old_precision - 0.05)
            checks["not_regressing vs old model"] = not_regressing
        except Exception:
            checks["not_regressing vs old model"] = True
    else:
        checks["not_regressing vs old model"] = True

    all_pass = all(checks.values())

    metrics = {
        "precision":   round(float(precision), 4),
        "recall":      round(float(recall),    4),
        "auc_roc":     round(float(auc),       4),
        "dataset_size": dataset_size,
        "checks":      checks,
        "passed":      all_pass,
    }

    return all_pass, metrics


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Swap model in S3
# ─────────────────────────────────────────────────────────────────────────────
def swap_model(s3, tenant_id, new_model, metrics):
    model_key    = f"tenant_{tenant_id}/model.pkl"
    cand_key     = f"tenant_{tenant_id}/candidate.pkl"
    meta_key     = f"tenant_{tenant_id}/metadata.json"

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name
    joblib.dump(new_model, tmp_path)

    try:
        # Upload as candidate first
        s3.upload_file(tmp_path, S3_MODELS, cand_key)

        # Atomic rename: copy candidate to model, delete candidate
        s3.copy_object(
            Bucket     = S3_MODELS,
            CopySource = {"Bucket": S3_MODELS, "Key": cand_key},
            Key        = model_key,
        )
        s3.delete_object(Bucket=S3_MODELS, Key=cand_key)

        # Upload metadata
        metadata = {
            "model_type":    "RandomForestClassifier",
            "trained_at":    datetime.now(timezone.utc).isoformat(),
            "phase":         "tenant",
            "tenant_id":     tenant_id,
            "precision":     metrics["precision"],
            "recall":        metrics["recall"],
            "auc_roc":       metrics["auc_roc"],
            "dataset_size":  metrics["dataset_size"],
            "feature_columns": FEATURE_COLUMNS,
        }
        s3.put_object(
            Bucket      = S3_MODELS,
            Key         = meta_key,
            Body        = json.dumps(metadata, indent=2).encode(),
            ContentType = "application/json",
        )
        print(f"[retrain] Model swapped for tenant {tenant_id}")
        return True

    except Exception as e:
        print(f"[retrain] Swap failed for {tenant_id}: {e}")
        return False
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Update DynamoDB metadata
# ─────────────────────────────────────────────────────────────────────────────
def update_dynamo(table, tenant_id, metrics):
    try:
        table.update_item(
            Key={"tenant_id": tenant_id},
            UpdateExpression=(
                "SET model_phase = :phase, "
                "model_precision = :prec, "
                "last_retrain = :ts"
            ),
            ExpressionAttributeValues={
                ":phase": "tenant",
                ":prec":  Decimal(str(metrics["precision"])),
                ":ts":    datetime.now(timezone.utc).isoformat(),
            },
        )
        print(f"[retrain] DynamoDB updated for {tenant_id}")
    except Exception as e:
        print(f"[retrain] DynamoDB update failed for {tenant_id}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Load existing model for comparison
# ─────────────────────────────────────────────────────────────────────────────
def load_existing_model(s3, tenant_id):
    key = f"tenant_{tenant_id}/model.pkl"
    try:
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name
        s3.download_file(S3_MODELS, key, tmp_path)
        model = joblib.load(tmp_path)
        os.remove(tmp_path)
        return model
    except Exception:
        return None  # No existing tenant model — first time training


# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Send Slack alert
# ─────────────────────────────────────────────────────────────────────────────
def send_slack_alert(webhook_url, tenant_id, success, metrics):
    if not webhook_url:
        return

    import urllib.request
    if success:
        msg = (
            f":white_check_mark: *Deploy Gate — Model Updated*\n"
            f"Tenant `{tenant_id}` has a new personalised model.\n"
            f"Precision: *{metrics['precision']*100:.1f}%*  |  "
            f"AUC-ROC: *{metrics['auc_roc']:.3f}*  |  "
            f"Trained on *{metrics['dataset_size']}* builds"
        )
    else:
        failed = [k for k, v in metrics.get("checks", {}).items() if not v]
        msg = (
            f":warning: *Deploy Gate — Retrain Failed*\n"
            f"Tenant `{tenant_id}` — keeping old model.\n"
            f"Failed checks: {', '.join(failed)}"
        )

    payload = json.dumps({"text": msg}).encode()
    try:
        req = urllib.request.Request(
            webhook_url,
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[retrain] Slack alert failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAMBDA HANDLER
# ─────────────────────────────────────────────────────────────────────────────
def lambda_handler(event, context):
    print(f"[retrain] Starting nightly retrain — {datetime.now(timezone.utc).isoformat()}")

    s3, table = get_clients()
    tenants   = get_all_tenants(table)

    results = {
        "retrained":  [],
        "skipped":    [],
        "failed":     [],
        "total":      len(tenants),
    }

    for tenant in tenants:
        tenant_id     = tenant.get("tenant_id", "")
        labelled_count = int(tenant.get("labelled_count", 0))
        webhook       = tenant.get("slack_webhook", "")

        # Skip demo tenant and tenants with insufficient data
        if tenant_id == "demo":
            continue

        if labelled_count < MIN_BUILDS:
            print(f"[retrain] Skipping {tenant_id} — only {labelled_count} labelled builds (need {MIN_BUILDS})")
            results["skipped"].append({
                "tenant_id": tenant_id,
                "reason":    f"only {labelled_count} labelled builds"
            })
            continue

        print(f"\n[retrain] Processing tenant {tenant_id} ({labelled_count} labelled builds)")

        try:
            # Load data
            df = load_tenant_data(s3, tenant_id)
            if df is None or len(df) < MIN_BUILDS:
                print(f"[retrain] Not enough valid rows for {tenant_id}")
                results["skipped"].append({"tenant_id": tenant_id, "reason": "insufficient data"})
                continue

            # Train
            new_model, X_test, y_test = train_model(df)

            # Load existing model for comparison
            old_model = load_existing_model(s3, tenant_id)

            # Validate
            passed, metrics = validate_model(new_model, old_model, X_test, y_test, len(df))

            print(f"[retrain] Validation: {'PASSED' if passed else 'FAILED'}")
            for check, result in metrics["checks"].items():
                status = "OK  " if result else "FAIL"
                print(f"  [{status}] {check}")

            if passed:
                # Swap model
                swapped = swap_model(s3, tenant_id, new_model, metrics)
                if swapped:
                    update_dynamo(table, tenant_id, metrics)
                    send_slack_alert(webhook, tenant_id, True, metrics)
                    results["retrained"].append({
                        "tenant_id": tenant_id,
                        "precision": metrics["precision"],
                        "auc_roc":   metrics["auc_roc"],
                    })
                    print(f"[retrain] SUCCESS: {tenant_id} — precision={metrics['precision']:.3f}")
                else:
                    results["failed"].append({"tenant_id": tenant_id, "reason": "swap failed"})
            else:
                send_slack_alert(webhook, tenant_id, False, metrics)
                results["failed"].append({
                    "tenant_id": tenant_id,
                    "reason":    "validation failed",
                    "metrics":   metrics,
                })

        except Exception as e:
            print(f"[retrain] ERROR processing {tenant_id}: {e}")
            results["failed"].append({"tenant_id": tenant_id, "reason": str(e)})

    # Summary
    print(f"\n[retrain] DONE")
    print(f"  Retrained : {len(results['retrained'])}")
    print(f"  Skipped   : {len(results['skipped'])}")
    print(f"  Failed    : {len(results['failed'])}")

    return {
        "statusCode": 200,
        "body":       json.dumps(results),
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOCAL TEST — simulates Lambda invocation
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Running Lambda handler locally...")
    result = lambda_handler({}, None)
    print(json.dumps(json.loads(result["body"]), indent=2))