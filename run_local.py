"""
Run SafeShip locally against an in-process mock AWS (moto) — no real
credentials or infra needed. Seeds the base model, a demo tenant, and a
handful of sample builds, then serves the real Flask app so you can click
through the UI in a browser.

    .venv/bin/python run_local.py

Then open http://127.0.0.1:5000  (login creds are printed on startup).
"""
import os
import sys
import time

# ── Mock-AWS env MUST be set before boto3 clients are created ───────────────
os.environ.update({
    "AWS_ACCESS_KEY_ID":     "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN":    "testing",
    "AWS_SESSION_TOKEN":     "testing",
    "AWS_DEFAULT_REGION":    "ap-south-1",
    "AWS_REGION":            "ap-south-1",
    "S3_MODELS_BUCKET":      "deploy-gate-models",
    "S3_DATA_BUCKET":        "deploy-gate-data",
    "DYNAMO_TABLE":          "tenants",
    "SECRET_KEY":            "local-dev-secret",
})

import boto3
from moto import mock_aws

REPO   = os.path.dirname(os.path.abspath(__file__))
REGION = "ap-south-1"
MODEL  = os.path.join(REPO, "ml", "data", "base_model.pkl")
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "ml"))

# Start the mock and keep it running for the whole server lifetime.
_mock = mock_aws()
_mock.start()

# ── Seed mock S3 + DynamoDB ─────────────────────────────────────────────────
s3 = boto3.client("s3", region_name=REGION)
for bucket in ("deploy-gate-models", "deploy-gate-data"):
    s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": REGION})
with open(MODEL, "rb") as f:
    s3.put_object(Bucket="deploy-gate-models", Key="base/model.pkl", Body=f.read())

boto3.client("dynamodb", region_name=REGION).create_table(
    TableName="tenants",
    KeySchema=[{"AttributeName": "tenant_id", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "tenant_id", "AttributeType": "S"}],
    BillingMode="PAY_PER_REQUEST",
)

import dynamo_client as dc
from routes.score import _write_build

creds = dc.create_tenant(email="demo@local.test")
TID, KEY = creds["tenant_id"], creds["api_key"]

# A few sample builds so the dashboard has charts/history to render.
# (score, label) — label -1 = pending, 0 = safe, 1 = risky
SAMPLES = [
    (22, 0), (35, 0), (68, 1), (81, 1), (44, 0),
    (15, 0), (73, 1), (58, -1), (90, 1), (30, 0),
]
now = int(time.time())
for i, (score, label) in enumerate(SAMPLES):
    bid = f"dg-{TID[:8]}-seed{i:02d}"
    _write_build(TID, {
        "build_id": bid,
        "timestamp": now - (len(SAMPLES) - i) * 3600,
        "predicted_score": score,
        "diff_size": 50 + i * 40, "files_changed": 2 + i,
        "hour_of_day": 10 + (i % 8), "day_of_week": i % 7,
        "recent_failure_rate": round((i % 5) / 10, 2), "test_pass_rate": 0.95,
        "is_hotfix": 1 if score > 80 else 0, "deployer_exp": 5 + i,
        "days_since_deploy": float(i % 4), "build_time_delta": 0.0,
        "job_name": "payments-service", "branch_name": "main", "triggered_by": "demo-user",
        "label": label, "label_source": "seed" if label != -1 else "pending",
    })
    dc.increment_build_count(TID)
    if label != -1:
        dc.increment_labelled_count(TID)

# macOS runs AirPlay Receiver on port 5000, so PORT is honoured here to give an
# easy way out:  PORT=5001 python run_local.py
PORT = int(os.getenv("PORT", "5000"))
BASE = f"http://127.0.0.1:{PORT}"

print("\n" + "=" * 64)
print("  SafeShip running locally on mock AWS (no real cloud touched)")
print("=" * 64)
print(f"  URL        : {BASE}")
print(f"  Login page : {BASE}/login")
print(f"  tenant_id  : {TID}")
print(f"  api_key    : {KEY}")
print(f"  Quick link : {BASE}/dashboard?tenant_id={TID}&api_key={KEY}")
print(f"  Demo (no login): {BASE}/demo")
print(f"  Readiness  : {BASE}/ready      (probes mock S3 + DynamoDB)")
print(f"  Metrics    : {BASE}/metrics")
print("=" * 64 + "\n")

# IMPORTANT: use_reloader=False — the reloader spawns a child process that
# would NOT have the moto mock active.
import main
main.app.run(host="127.0.0.1", port=PORT, debug=False, use_reloader=False)
