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
import time
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
from dynamo_client    import (validate_tenant, increment_build_count, create_tenant,
                              actor_build_count, increment_actor_build)
from imputation       import impute, invalidate as invalidate_medians
from labels           import describe as describe_label
from labels           import is_observed as already_observed
from features         import FEATURES
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

    # Step 4: Resolve features, then score.
    #
    # deployer_exp is derived server-side from this tenant's own build history —
    # never taken from the request. A client could otherwise claim
    # deployer_exp=999 to look like a veteran and lower its own risk score.
    raw   = features.raw_features()
    actor = (features.triggered_by or "unknown").strip() or "unknown"

    server_exp = actor_build_count(features.tenant_id, actor)
    hint       = raw.get("deployer_exp")

    if server_exp > 0:
        if hint is not None and int(hint) != server_exp:
            log.info(
                "ignoring client-supplied deployer_exp in favour of server history",
                extra={"tenant_id": features.tenant_id, "actor": actor,
                       "claimed": hint, "actual": server_exp},
            )
        raw["deployer_exp"] = server_exp
        exp_source = "server_history"
    else:
        # No history for this actor yet, so there is nothing to check the claim
        # against. Impute a baseline first and bound the hint by it: a hint is
        # only credible when it works AGAINST the caller. Claiming inexperience
        # is self-penalising and therefore believable; claiming to be a veteran
        # is unverifiable and is precisely the spoof. Accepting an arbitrary
        # number here would reopen it — a caller need only rotate triggered_by
        # on every build to be permanently "new", and permanently a veteran.
        raw["deployer_exp"] = None
        exp_source = None

    values, imputed, source = impute(raw, features.tenant_id)

    if exp_source == "server_history":
        source["deployer_exp"] = exp_source
        if "deployer_exp" in imputed:
            imputed.remove("deployer_exp")
    elif hint is not None:
        baseline = values["deployer_exp"]
        if int(hint) <= baseline:
            values["deployer_exp"] = int(hint)
            source["deployer_exp"] = "client_hint"
            if "deployer_exp" in imputed:
                imputed.remove("deployer_exp")
        else:
            # Keep the imputed value, and leave it reported as imputed — that is
            # what actually happened.
            log.info(
                "capping optimistic deployer_exp hint from an unknown actor",
                extra={"tenant_id": features.tenant_id, "actor": actor,
                       "claimed": int(hint), "capped_at": baseline},
            )

    model_input = [values[f] for f in FEATURES]
    # imputed/source go in so top_reasons can mark an estimated value as one,
    # rather than presenting a median as something we measured.
    result      = score_build(model_input, features.tenant_id,
                              imputed=imputed, sources=source)

    # Tell the caller what was measured and what was guessed. A confident number
    # built mostly on medians should not look identical to a measured one.
    result["imputed"]         = imputed
    result["feature_sources"] = source
    if imputed:
        log.info("scored with imputed features",
                 extra={"tenant_id": features.tenant_id, "imputed": imputed,
                        "n_imputed": len(imputed)})

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
        # Persist the values the model actually scored, including imputed ones —
        # otherwise retraining learns from rows full of nulls.
        row.update(values)
        row.update({
            "build_id":        build_id,
            "timestamp":       int(datetime.now(timezone.utc).timestamp()),
            "predicted_score": score_val,
            "imputed":         imputed,
            "actor":           actor,
        })
        _write_build(features.tenant_id, row)
        # build_count lives in DynamoDB (atomic counter) — no need to count S3 objects
        result["total_builds"] = increment_build_count(features.tenant_id)
        increment_actor_build(features.tenant_id, actor)
        # New history invalidates the cached medians for this tenant.
        invalidate_medians(features.tenant_id)
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

        # Provenance decides how much this row counts when retraining. The rule
        # used to be an inline `1.0 if label_src in ["failure", "safe"] else 0.7`,
        # which meant safeship_ci's "success" fell through to 0.7 — the most
        # common label in the dataset, down-weighted by a string mismatch that
        # nothing would ever surface. See app/labels.py.
        canonical, weight, observed = describe_label(label_src)

        # An observation outranks an inference, whichever arrives second.
        # Pipelines commonly do both: Sentinel probes production and reports what
        # it saw, then a post-build hook reports the pipeline's exit status as a
        # fallback. Letting the fallback win would replace "production degraded"
        # with "the pipeline was green" — discarding the better evidence and, in
        # the case that matters most, inverting the label.
        if already_observed(row.get("label_source")) and not observed:
            log.info("keeping the observed label over an inferred one",
                     extra={"tenant_id": tenant_id, "build_id": build_id,
                            "kept": row.get("label_source"), "ignored": canonical})
            return jsonify({
                "status": "kept",
                "build_id": build_id,
                "label": row.get("label"),
                "label_source": row.get("label_source"),
                "sample_weight": row.get("sample_weight"),
                "observed": True,
                "note": (f"{canonical} ignored: this build was already labelled by "
                         f"{row.get('label_source')}, which observed production "
                         "rather than inferring from pipeline status"),
            }), 200

        row["label"]         = label
        row["label_source"]  = canonical
        row["sample_weight"] = weight
        row["label_observed"] = observed
        row["labelled_at"]   = int(time.time())

        s3.put_object(
            Bucket=S3_DATA_BUCKET, Key=key,
            Body=json.dumps(row).encode("utf-8"),
            ContentType="application/json"
        )

        log.info("build labelled",
                 extra={"tenant_id": tenant_id, "build_id": build_id,
                        "label": label, "label_source": canonical,
                        "sample_weight": weight, "observed": observed})

        return jsonify({
            "status": "updated", "build_id": build_id, "label": label,
            # Echoed so a caller can see that its source was recognised. A label
            # accepted at 0.5 because the source was a typo should be visible.
            "label_source": canonical,
            "sample_weight": weight,
            "observed": observed,
        }), 200

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
            # Was "Add the Jenkinsfile stage from /dashboard", which assumed
            # Jenkins for everyone. /setup asks which CI you actually use.
            "next_step": "Connect your pipeline at /setup",
            "setup_url": "/setup",
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