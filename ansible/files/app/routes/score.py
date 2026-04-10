"""
score.py
Path: C:\deploy-gate\app\routes\score.py

PURPOSE:
  The /score endpoint — the core of the entire product.
  Jenkins calls this before every deploy.
  Returns risk score 0-100, verdict, and top 3 reasons.

  Also handles /log (records build to S3)
  and /signup (creates new tenant).
"""

import os
import csv
import uuid
import boto3
import io
import sys
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify

# Add parent directory to path so we can import app modules
# Add both app/ and ml/ to path
_app_dir     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ml_dir      = os.path.join(os.path.dirname(_app_dir), "ml")
_project_dir = os.path.dirname(_app_dir)
sys.path.insert(0, _app_dir)
sys.path.insert(0, _ml_dir)
sys.path.insert(0, _project_dir)

from validator        import BuildFeatures, LogRequest, SignupRequest
from scorer           import score_build
from dynamo_client    import validate_tenant, increment_build_count, create_tenant
from slack_notifier   import send_alert

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "deploy-gate-data")
AWS_REGION     = os.getenv("AWS_REGION",     "ap-south-1")

score_bp = Blueprint("score", __name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — append one row to tenant's S3 CSV
# ─────────────────────────────────────────────────────────────────────────────

def _append_to_s3_csv(tenant_id, row_dict):
    """
    Appends a single row to tenant's data.csv in S3.
    If file doesn't exist yet, creates it with headers.
    Uses read-modify-write pattern (safe for low-frequency writes).
    """
    s3  = boto3.client("s3", region_name=AWS_REGION)
    key = f"tenant_{tenant_id}/data.csv"

    COLUMNS = [
        "build_id", "timestamp", "diff_size", "files_changed",
        "hour_of_day", "day_of_week", "recent_failure_rate",
        "test_pass_rate", "is_hotfix", "deployer_exp",
        "days_since_deploy", "build_time_delta",
        "predicted_score", "label", "label_source",
        "sample_weight", "triggered_by", "job_name", "branch_name",
    ]

    # Try to read existing CSV
    existing_rows = []
    try:
        obj      = s3.get_object(Bucket=S3_DATA_BUCKET, Key=key)
        content  = obj["Body"].read().decode("utf-8")
        reader   = csv.DictReader(io.StringIO(content))
        existing_rows = list(reader)
    except s3.exceptions.NoSuchKey:
        pass  # First build — file doesn't exist yet
    except Exception as e:
        print(f"[score] Warning: could not read existing CSV: {e}")

    # Add defaults for missing fields
    row_dict.setdefault("label",        -1)
    row_dict.setdefault("label_source", "pending")
    row_dict.setdefault("sample_weight", 1.0)

    existing_rows.append(row_dict)

    # Write back
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=COLUMNS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(existing_rows)

    s3.put_object(
        Bucket      = S3_DATA_BUCKET,
        Key         = key,
        Body        = output.getvalue().encode("utf-8"),
        ContentType = "text/csv",
    )
    return len(existing_rows)


# ─────────────────────────────────────────────────────────────────────────────
# POST /score
# ─────────────────────────────────────────────────────────────────────────────

@score_bp.route("/score", methods=["POST"])
def score():
    """
    Main scoring endpoint. Called by Jenkins before every deploy.

    Request JSON:
        tenant_id, api_key, hour_of_day, day_of_week,
        + any of the 10 features (all optional with defaults)

    Response JSON:
        score, verdict, color, model_phase, top_reasons, build_id
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    # Step 1: Validate input with Pydantic
    try:
        features = BuildFeatures(**data)
    except ValidationError as e:
        errors = e.errors()
        return jsonify({
            "error":   "Validation failed",
            "details": [f"{err['loc'][0]}: {err['msg']}" for err in errors]
        }), 422
    except Exception as e:
        return jsonify({"error": f"Invalid request: {str(e)}"}), 400

    # Step 2: Authenticate tenant
    tenant = validate_tenant(features.tenant_id, features.api_key)
    if not tenant:
        return jsonify({"error": "Invalid tenant_id or api_key"}), 401

    # Step 3: Get custom thresholds from tenant settings
    thresh_yellow = int(tenant.get("threshold_yellow", 40))
    thresh_red    = int(tenant.get("threshold_red",    70))

    # Step 4: Run ML scoring
    model_input = features.to_model_input()
    result      = score_build(model_input, features.tenant_id)

    # Step 5: Apply tenant's custom thresholds to verdict
    score_val = result["score"]
    if score_val <= thresh_yellow:
        result["verdict"] = "SAFE"
        result["color"]   = "green"
    elif score_val <= thresh_red:
        result["verdict"] = "WARNING"
        result["color"]   = "yellow"
    else:
        result["verdict"] = "BLOCKED"
        result["color"]   = "red"

    # Step 6: Generate a unique build_id for this score
    build_id = f"dg-{features.tenant_id[:8]}-{uuid.uuid4().hex[:8]}"
    result["build_id"] = build_id

    # Step 7: Send Slack alert (non-blocking — never crash if Slack fails)
    try:
        job_name     = features.job_name or "unknown-job"
        build_number = data.get("build_number", "?")
        send_alert(job_name, build_number, result, tenant)
    except Exception as e:
        print(f"[score] Slack alert failed (non-fatal): {e}")

    # Step 8: Log to S3 asynchronously (non-blocking)
    try:
        row = features.to_log_dict()
        row.update({
            "build_id":        build_id,
            "timestamp":       int(datetime.now(timezone.utc).timestamp()),
            "predicted_score": score_val,
        })
        total_builds = _append_to_s3_csv(features.tenant_id, row)
        increment_build_count(features.tenant_id)
        result["total_builds"] = total_builds
    except Exception as e:
        print(f"[score] S3 logging failed (non-fatal): {e}")

    return jsonify(result), 200


# ─────────────────────────────────────────────────────────────────────────────
# POST /log  (outcome logger — called 6h after deploy)
# ─────────────────────────────────────────────────────────────────────────────

@score_bp.route("/log", methods=["POST"])
def log_outcome():
    """
    Updates label on an existing build row in S3.
    Called by Jenkins post-build step 6 hours after deploy.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    tenant_id  = data.get("tenant_id", "")
    api_key    = data.get("api_key",   "")
    build_id   = data.get("build_id",  "")
    label      = data.get("label",      -1)   # 0=safe, 1=risky
    label_src  = data.get("label_source", "manual")

    if not tenant_id or not api_key or not build_id:
        return jsonify({"error": "tenant_id, api_key, build_id are required"}), 400

    tenant = validate_tenant(tenant_id, api_key)
    if not tenant:
        return jsonify({"error": "Invalid credentials"}), 401

    # Update the label in S3 CSV
    try:
        s3  = boto3.client("s3", region_name=AWS_REGION)
        key = f"tenant_{tenant_id}/data.csv"

        obj     = s3.get_object(Bucket=S3_DATA_BUCKET, Key=key)
        content = obj["Body"].read().decode("utf-8")
        reader  = csv.DictReader(io.StringIO(content))
        rows    = list(reader)
        cols    = reader.fieldnames

        updated = False
        for row in rows:
            if row.get("build_id") == build_id:
                row["label"]        = str(label)
                row["label_source"] = label_src
                row["sample_weight"] = "1.0" if label_src in ["failure", "safe"] else "0.7"
                updated = True
                break

        if not updated:
            return jsonify({"error": f"build_id {build_id} not found"}), 404

        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        s3.put_object(
            Bucket=S3_DATA_BUCKET, Key=key,
            Body=output.getvalue().encode("utf-8"),
            ContentType="text/csv"
        )

        return jsonify({"status": "updated", "build_id": build_id, "label": label}), 200

    except Exception as e:
        print(f"[log] Error updating label: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# POST /signup
# ─────────────────────────────────────────────────────────────────────────────

@score_bp.route("/signup", methods=["POST"])
def signup():
    """
    Creates a new tenant. Returns tenant_id and api_key.
    This is the only time the plain api_key is ever returned.
    """
    data  = request.get_json(silent=True) or {}
    email = data.get("email", "").strip()

    try:
        result = create_tenant(email=email)
        return jsonify({
            "tenant_id": result["tenant_id"],
            "api_key":   result["api_key"],
            "message":   "Save your api_key — it will not be shown again.",
            "next_step": "Add the Jenkinsfile stage from /dashboard"
        }), 201
    except Exception as e:
        print(f"[signup] Error creating tenant: {e}")
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /health  (used by load balancers and monitoring)
# ─────────────────────────────────────────────────────────────────────────────

@score_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "deploy-gate"}), 200