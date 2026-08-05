
"""
dashboard.py - Fixed with session-based auth + email login
"""
import os, sys, csv, io, json, boto3, secrets
from flask import (Blueprint, render_template, request, jsonify,
                   redirect, session, url_for)

_app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ml_dir  = os.path.join(os.path.dirname(_app_dir), "ml")
sys.path.insert(0, _app_dir)
sys.path.insert(0, _ml_dir)

from dynamo_client import (validate_tenant, create_tenant,
                            lookup_by_email, get_tenant,
                            update_slack_webhook, update_thresholds)
from scorer    import score_build
from validator import BuildFeatures
from features   import FEATURES
import integrations
from integrations import public_base_url

S3_DATA   = os.getenv("S3_DATA_BUCKET", "deploy-gate-data")
AWS_REGION= os.getenv("AWS_REGION",     "ap-south-1")

# Kept in step with lambda/retrain/handler.py::MIN_BUILDS, which cannot be
# imported from here because its package is literally named `lambda`, a Python
# keyword. Pinned by tests/test_setup.py so the two cannot drift.
RETRAIN_MIN_BUILDS = 200

dashboard_bp = Blueprint("dashboard", __name__)

from observability import get_logger

log = get_logger("routes.dashboard")


def _load_builds(tenant_id, limit=30):
    """
    Returns the most recent `limit` builds for a tenant.

    Reads append-only per-build objects (tenant_<id>/builds/*.json) and only
    fetches the newest `limit` of them, so dashboard cost does not grow with
    full history. Falls back to the legacy data.csv for pre-migration data.
    """
    s3   = boto3.client("s3", region_name=AWS_REGION)
    rows = []

    # New per-build objects — list metadata (cheap), GET only the newest `limit`
    try:
        entries = []
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_DATA, Prefix=f"tenant_{tenant_id}/builds/"):
            for o in page.get("Contents", []):
                entries.append((o["LastModified"], o["Key"]))
        entries.sort()
        for _, key in entries[-limit:]:
            try:
                obj = s3.get_object(Bucket=S3_DATA, Key=key)
                rows.append(json.loads(obj["Body"].read().decode()))
            except Exception:
                pass
    except Exception:
        pass

    # Legacy CSV fallback (data written before the per-build migration)
    if not rows:
        try:
            obj = s3.get_object(Bucket=S3_DATA, Key=f"tenant_{tenant_id}/data.csv")
            rows = list(csv.DictReader(io.StringIO(obj["Body"].read().decode())))
        except Exception:
            pass

    return rows[-limit:] if len(rows) > limit else rows


def _get_session_tenant():
    """Returns tenant dict if user is logged in via session."""
    tid = session.get("tenant_id")
    key = session.get("api_key")
    if not tid or not key:
        return None, None
    tenant = validate_tenant(tid, key)
    return tenant, key


# ── PUBLIC PAGES ───────────────────────────────────────────────

@dashboard_bp.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@dashboard_bp.route("/about", methods=["GET"])
def about():
    return render_template("about.html")

@dashboard_bp.route("/demo", methods=["GET"])
def demo():
    return render_template("demo.html")

@dashboard_bp.route("/demo/score", methods=["POST"])
def demo_score():
    data = request.get_json(silent=True) or {}
    data["tenant_id"] = "demo"
    data["api_key"]   = "demo"
    try:
        features = BuildFeatures(**data)
        # impute_features, not to_model_input: the latter throws away which values
        # were guessed. The demo is the most public surface there is, so it should
        # not be the one place that shows a median as though it were measured.
        values, imputed, source = features.impute_features("demo")
        result = score_build([values[f] for f in FEATURES], "demo",
                             imputed=imputed, sources=source)
        result["imputed"]         = imputed
        result["feature_sources"] = source
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ── AUTH ───────────────────────────────────────────────────────

@dashboard_bp.route("/signup", methods=["GET"])
def signup_page():
    # If already logged in, go to dashboard
    if session.get("tenant_id"):
        return redirect("/dashboard")
    return render_template("signup.html", error="")

@dashboard_bp.route("/login", methods=["GET"])
def login_page():
    if session.get("tenant_id"):
        return redirect("/dashboard")
    return render_template("login.html", error="")

@dashboard_bp.route("/login", methods=["POST"])
def login():
    """
    Login supports two methods:
    1. tenant_id + api_key  (classic)
    2. email lookup         (finds tenant by email, still needs api_key)
    """
    data      = request.get_json(silent=True) or {}
    email     = data.get("email",     "").strip()
    tenant_id = data.get("tenant_id", "").strip()
    api_key   = data.get("api_key",   "").strip()

    # Method 1: email + api_key
    if email and api_key and not tenant_id:
        tenant = lookup_by_email(email)
        if not tenant:
            return jsonify({"error": "No account found for this email."}), 401
        tenant_id = tenant["tenant_id"]

    # Method 2: tenant_id + api_key
    if not tenant_id or not api_key:
        return jsonify({"error": "Provide tenant_id + api_key, or email + api_key."}), 400

    tenant = validate_tenant(tenant_id, api_key)
    if not tenant:
        return jsonify({"error": "Invalid credentials. Check your tenant_id and api_key."}), 401

    # Save to session
    session["tenant_id"] = tenant_id
    session["api_key"]   = api_key
    session.permanent    = True

    return jsonify({
        "success":   True,
        "tenant_id": tenant_id,
        "redirect":  "/dashboard"
    }), 200

@dashboard_bp.route("/logout", methods=["GET"])
def logout():
    session.clear()
    return redirect("/login")


# ── DASHBOARD (session protected) ─────────────────────────────

@dashboard_bp.route("/dashboard", methods=["GET"])
def dashboard():
    # Support both session login AND URL params (for backward compat)
    tenant_id = request.args.get("tenant_id") or session.get("tenant_id", "")
    api_key   = request.args.get("api_key")   or session.get("api_key",   "")

    if not tenant_id or not api_key:
        return redirect("/login")

    tenant = validate_tenant(tenant_id, api_key)
    if not tenant:
        return redirect("/login?error=invalid")

    # Save to session on URL-param login
    session["tenant_id"] = tenant_id
    session["api_key"]   = api_key
    session.permanent    = True

    builds = _load_builds(tenant_id, 30)
    scores     = [int(b.get("predicted_score", 0)) for b in builds]
    build_nums = list(range(1, len(scores)+1))
    colors     = ["#16a34a" if s<=40 else "#d97706" if s<=70 else "#dc2626" for s in scores]

    build_count = len(builds)

    labelled_count = sum(
        1 for b in builds
        if str(b.get("label", "")).strip() in ["0", "1"]
    )
    model_phase    = tenant.get("model_phase",   "base")
    precision      = float(tenant.get("model_precision", 0.851))
    # Progress toward getting your own model, so it has to use the threshold the
    # retrain job actually enforces. This divided by 5 — the old MIN_BUILDS —
    # and so showed 100% at five labelled builds while retraining now needs 200.
    progress_pct   = min(100, int(labelled_count / RETRAIN_MIN_BUILDS * 100))

    try:
        from scorer import _cache, FEATURE_COLUMNS
        model, _ = _cache.get_model(tenant_id)

        feat_imp = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
        )[:5]

        feat_names = [f[0].replace("_"," ").title() for f in feat_imp]
        feat_values = [round(f[1]*100,1) for f in feat_imp]

    except Exception as e:
        log.warning("feature importance chart failed", extra={"err": str(e)})

        feat_names = [
            "Recent Failure Rate",
            "Test Pass Rate",
            "Diff Size",
            "Hotfix",
            "Hour"
            ]

        feat_values = [28,25,18,11,8]
    # The pipeline snippet used to be built right here: ~80 lines of Jenkins
    # Groovy, rendered unconditionally whatever CI the tenant actually used, and
    # hardcoding seven of the ten features ("files_changed":5,
    # "recent_failure_rate":0.0, "test_pass_rate":1.0, "deployer_exp":30, ...).
    # Anyone who followed it got a score that barely depended on their build.
    # It now comes from app/integrations/<platform>.py and calls safeship_ci.
    integration = None
    ci_platform = tenant.get("ci_platform", "")
    if integrations.is_valid(ci_platform):
        try:
            integration = integrations.describe(
                ci_platform, tenant_id, api_key, public_base_url())
        except Exception as exc:            # pragma: no cover - never break the page
            log.warning("could not build integration snippet",
                        extra={"tenant_id": tenant_id, "err": str(exc)})

    return render_template("dashboard.html",
        tenant=tenant, tenant_id=tenant_id, api_key=api_key,
        scores=json.dumps(scores), build_nums=json.dumps(build_nums),
        colors=json.dumps(colors),
        build_count=build_count, labelled_count=labelled_count,
        model_phase=model_phase, precision=round(precision*100,1),
        progress_pct=progress_pct,
        recent_builds=list(reversed(builds))[:10],
        feat_names=json.dumps(feat_names), feat_values=json.dumps(feat_values),
        integration=integration,
        ci_platform=ci_platform,
        slack_webhook=tenant.get("slack_webhook",""),
        thresh_yellow=int(tenant.get("threshold_yellow",40)),
        thresh_red=int(tenant.get("threshold_red",70)),
    )


@dashboard_bp.route("/settings", methods=["POST"])
def save_settings():
    data      = request.get_json(silent=True) or {}
    tenant_id = data.get("tenant_id") or session.get("tenant_id","")
    api_key   = data.get("api_key")   or session.get("api_key","")
    tenant    = validate_tenant(tenant_id, api_key)
    if not tenant:
        return jsonify({"error":"Invalid credentials"}), 401
    webhook = data.get("slack_webhook","").strip()
    yellow  = int(data.get("threshold_yellow", 40))
    red     = int(data.get("threshold_red",    70))
    if webhook:
        update_slack_webhook(tenant_id, webhook)
    update_thresholds(tenant_id, yellow, red)
    return jsonify({"status":"saved"}), 200