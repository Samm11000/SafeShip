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
import uuid
import json
import boto3
import sys
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify
from pydantic import ValidationError

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

from observability import get_logger

log = get_logger("routes.score")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
S3_DATA_BUCKET = os.getenv("S3_DATA_BUCKET", "deploy-gate-data")
AWS_REGION     = os.getenv("AWS_REGION",     "ap-south-1")

score_bp = Blueprint("score", __name__)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — append-only per-build storage in S3
#
# Each build is its own small object: tenant_<id>/builds/<build_id>.json
# This makes /score writes O(1) (one tiny PUT) instead of rewriting the
# tenant's entire history on every deploy, and removes the lost-update race
# that the old read-modify-write CSV pattern had under concurrent builds.
# ─────────────────────────────────────────────────────────────────────────────

def _build_key(tenant_id, build_id):
    return f"tenant_{tenant_id}/builds/{build_id}.json"


def _write_build(tenant_id, row_dict):
    """Persist one build as its own object. O(1), no whole-file rewrite, no race."""
    row_dict.setdefault("label",        -1)
    row_dict.setdefault("label_source", "pending")
    row_dict.setdefault("sample_weight", 1.0)

    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(
        Bucket      = S3_DATA_BUCKET,
        Key         = _build_key(tenant_id, row_dict["build_id"]),
        Body        = json.dumps(row_dict).encode("utf-8"),
        ContentType = "application/json",
    )


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
        log.warning("slack alert failed (non-fatal)", extra={"err": str(e)})

    # Step 8: Log to S3 asynchronously (non-blocking)
    try:
        row = features.to_log_dict()
        row.update({
            "build_id":        build_id,
            "timestamp":       int(datetime.now(timezone.utc).timestamp()),
            "predicted_score": score_val,
        })
        _write_build(features.tenant_id, row)
        # build_count lives in DynamoDB (atomic counter) — no need to count S3 objects
        result["total_builds"] = increment_build_count(features.tenant_id)
    except Exception as e:
        log.error("S3 build write failed (non-fatal)", extra={"err": str(e)})

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

    # Update the label on just this build's object — O(1), no whole-history rewrite
    try:
        s3  = boto3.client("s3", region_name=AWS_REGION)
        key = _build_key(tenant_id, build_id)

        try:
            obj = s3.get_object(Bucket=S3_DATA_BUCKET, Key=key)
        except s3.exceptions.NoSuchKey:
            return jsonify({"error": f"build_id {build_id} not found"}), 404

        row = json.loads(obj["Body"].read().decode("utf-8"))
        row["label"]         = label
        row["label_source"]  = label_src
        row["sample_weight"] = 1.0 if label_src in ["failure", "safe"] else 0.7

        s3.put_object(
            Bucket=S3_DATA_BUCKET, Key=key,
            Body=json.dumps(row).encode("utf-8"),
            ContentType="application/json"
        )

        return jsonify({"status": "updated", "build_id": build_id, "label": label}), 200

    except Exception as e:
        log.error("label update failed", extra={"err": str(e)})
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
        log.error("tenant signup failed", extra={"err": str(e)})
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# GET /health  (used by load balancers and monitoring)
# ─────────────────────────────────────────────────────────────────────────────

@score_bp.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "deploy-gate"}), 200