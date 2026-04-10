"""
drift_handler.py
Path: C:\deploy-gate\lambda\drift\handler.py

PURPOSE:
  Weekly Lambda triggered every Sunday at 3am UTC.
  Detects if a tenant's feature distributions have shifted significantly.
  If drift detected: alerts tenant via Slack and flags for forced retrain.
"""

import os
import io
import json
import boto3
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from scipy.stats import ks_2samp

AWS_REGION = os.getenv("AWS_DEFAULT_REGION", os.getenv("AWS_EXECUTION_ENV", "us-east-1"))
S3_DATA      = os.getenv("S3_DATA_BUCKET",    "deploy-gate-data")
DYNAMO_TABLE = os.getenv("DYNAMO_TABLE",      "tenants")

FEATURE_COLUMNS = [
    "diff_size", "files_changed", "hour_of_day", "day_of_week",
    "recent_failure_rate", "test_pass_rate", "is_hotfix",
    "deployer_exp", "days_since_deploy", "build_time_delta",
]

# If p-value below this, distributions are significantly different
DRIFT_THRESHOLD = 0.05
# How many features must drift to trigger alert
MIN_DRIFTED_FEATURES = 3


def detect_drift(df):
    """
    Compares last 30 builds vs previous 30 builds using KS-test.
    Returns list of drifted features.
    """
    if len(df) < 60:
        return []

    df_sorted = df.sort_values("timestamp") if "timestamp" in df.columns else df
    recent    = df_sorted.tail(30)
    previous  = df_sorted.iloc[-60:-30]

    drifted = []
    for feat in FEATURE_COLUMNS:
        if feat not in df.columns:
            continue
        try:
            stat, p_value = ks_2samp(
                previous[feat].dropna().astype(float),
                recent[feat].dropna().astype(float)
            )
            if p_value < DRIFT_THRESHOLD:
                drifted.append({
                    "feature": feat,
                    "p_value": round(float(p_value), 4),
                    "stat":    round(float(stat),    4),
                })
        except Exception:
            continue

    return drifted


def send_drift_alert(webhook_url, tenant_id, drifted_features):
    if not webhook_url:
        return
    import urllib.request
    feat_list = "\n".join([
        f"  • {d['feature']} (p={d['p_value']})"
        for d in drifted_features
    ])
    msg = (
        f":chart_with_upwards_trend: *Deploy Gate — Feature Drift Detected*\n"
        f"Tenant `{tenant_id}` — build patterns have shifted.\n"
        f"Drifted features:\n{feat_list}\n"
        f"_Model will be force-retrained tonight at 2am._"
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
        print(f"[drift] Slack alert failed: {e}")


def lambda_handler(event, context):
    print(f"[drift] Starting drift detection — {datetime.now(timezone.utc).isoformat()}")

    s3     = boto3.client("s3",         region_name=AWS_REGION)
    dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
    table  = dynamo.Table(DYNAMO_TABLE)

    tenants = table.scan().get("Items", [])
    results = {"drifted": [], "stable": [], "skipped": []}

    for tenant in tenants:
        tenant_id = tenant.get("tenant_id", "")
        webhook   = tenant.get("slack_webhook", "")

        if tenant_id == "demo":
            continue

        # Load data
        try:
            key     = f"tenant_{tenant_id}/data.csv"
            obj     = s3.get_object(Bucket=S3_DATA, Key=key)
            content = obj["Body"].read().decode("utf-8")
            df      = pd.read_csv(io.StringIO(content))
        except Exception:
            results["skipped"].append(tenant_id)
            continue

        if len(df) < 60:
            results["skipped"].append(tenant_id)
            continue

        # Detect drift
        drifted = detect_drift(df)

        if len(drifted) >= MIN_DRIFTED_FEATURES:
            print(f"[drift] DRIFT DETECTED for {tenant_id}: {len(drifted)} features")
            send_drift_alert(webhook, tenant_id, drifted)

            # Flag in DynamoDB — retrain Lambda will pick this up tonight
            try:
                table.update_item(
                    Key={"tenant_id": tenant_id},
                    UpdateExpression="SET drift_alert_sent = :t",
                    ExpressionAttributeValues={":t": True},
                )
            except Exception as e:
                print(f"[drift] DynamoDB update failed: {e}")

            results["drifted"].append({
                "tenant_id":       tenant_id,
                "drifted_features": drifted,
            })
        else:
            print(f"[drift] Stable: {tenant_id} ({len(drifted)} features drifted)")
            results["stable"].append(tenant_id)

    print(f"\n[drift] DONE")
    print(f"  Drifted : {len(results['drifted'])}")
    print(f"  Stable  : {len(results['stable'])}")
    print(f"  Skipped : {len(results['skipped'])}")

    return {"statusCode": 200, "body": json.dumps(results)}


if __name__ == "__main__":
    result = lambda_handler({}, None)
    print(json.dumps(json.loads(result["body"]), indent=2))