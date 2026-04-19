
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

S3_DATA   = os.getenv("S3_DATA_BUCKET", "deploy-gate-data")
AWS_REGION= os.getenv("AWS_REGION",     "ap-south-1")

dashboard_bp = Blueprint("dashboard", __name__)


def _load_builds(tenant_id, limit=30):
    try:
        s3  = boto3.client("s3", region_name=AWS_REGION)
        obj = s3.get_object(Bucket=S3_DATA, Key=f"tenant_{tenant_id}/data.csv")
        rows = list(csv.DictReader(io.StringIO(obj["Body"].read().decode())))
        return rows[-limit:] if len(rows) > limit else rows
    except Exception:
        return []


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
        return jsonify(score_build(features.to_model_input(), "demo")), 200
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
    progress_pct   = min(100, int(labelled_count / 5 * 100))

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
        print("Feature chart error:", e)

        feat_names = [
            "Recent Failure Rate",
            "Test Pass Rate",
            "Diff Size",
            "Hotfix",
            "Hour"
            ]

        feat_values = [28,25,18,11,8]
    env_block = f"""environment {{
    SAFESHIP_URL       = 'http://54.89.160.150'
    SAFESHIP_TENANT_ID = '{tenant_id}'
    SAFESHIP_API_KEY   = '{api_key}'
    }}"""
    risk_stage = """stage('SafeShip Risk Check') {
  steps {
    script {

      def diffStat = sh(
        script: "git diff --stat HEAD~1 HEAD 2>/dev/null | tail -1 || echo '0'",
        returnStdout: true
      ).trim()

      def added   = (diffStat =~ /(\\d+) insertion/) ? (diffStat =~ /(\\d+) insertion/)[0][1].toInteger() : 0
      def deleted = (diffStat =~ /(\\d+) deletion/)  ? (diffStat =~ /(\\d+) deletion/)[0][1].toInteger() : 0
      def diffSize = added + deleted

      def now = new Date()

      def payload = \"\"\"{
        "tenant_id":"${env.SAFESHIP_TENANT_ID}",
        "api_key":"${env.SAFESHIP_API_KEY}",
        "diff_size":${diffSize},
        "files_changed":5,
        "hour_of_day":${now.hours},
        "day_of_week":${now.day},
        "recent_failure_rate":0.0,
        "test_pass_rate":1.0,
        "is_hotfix":0,
        "deployer_exp":30,
        "days_since_deploy":2,
        "build_time_delta":0.0
      }\"\"\"

      def res = sh(
        script: "curl -s -X POST ${env.SAFESHIP_URL}/score -H 'Content-Type: application/json' -d '${payload}'",
        returnStdout:true
      ).trim()

      def result = readJSON text: res

      env.DG_BUILD_ID = result.build_id

      echo "SafeShip Score: ${result.score}"

      if (result.verdict == 'BLOCKED') {
        error("SafeShip blocked deployment")
      }
    }
  }
}"""
    post_block = """script {

    if (env.DG_BUILD_ID?.trim()) {

        def finalLabel = (currentBuild.currentResult == 'SUCCESS') ? 0 : 1
        def sourceTxt  = (finalLabel == 0) ? "safe" : "failure"

        def payload = \"\"\"{
        "tenant_id":"${env.SAFESHIP_TENANT_ID}",
        "api_key":"${env.SAFESHIP_API_KEY}",
        "build_id":"${env.DG_BUILD_ID}",
        "label":${finalLabel},
        "label_source":"${sourceTxt}"
        }\"\"\"

        sh "curl -s -X POST ${env.SAFESHIP_URL}/log -H 'Content-Type: application/json' -d '${payload}'"
    }

    }"""
    jenkinsfile = f"""stage('SafeShip Risk Check') {{
    steps {{
        script {{
            def res = sh(script: \"\"\"curl -s -X POST http://YOUR-EC2-IP/score \\\\
              -H 'Content-Type: application/json' \\\\
              -d '{{"tenant_id":"{tenant_id}","api_key":"{api_key}","hour_of_day":${{new Date().hours}},"day_of_week":${{new Date().day}},"diff_size":${{env.GIT_DIFF_SIZE ?: 100}},"recent_failure_rate":0.0}}'\"\"\", returnStdout:true).trim()
            def result = readJSON text: res
            if (result.verdict == 'BLOCKED') error("SafeShip: ${{result.score}}/100 blocked")
            echo "SafeShip: ${{result.score}}/100 - ${{result.verdict}}"
        }}
    }}
}}"""

    return render_template("dashboard.html",
        tenant=tenant, tenant_id=tenant_id, api_key=api_key,
        scores=json.dumps(scores), build_nums=json.dumps(build_nums),
        colors=json.dumps(colors),
        build_count=build_count, labelled_count=labelled_count,
        model_phase=model_phase, precision=round(precision*100,1),
        progress_pct=progress_pct,
        recent_builds=list(reversed(builds))[:10],
        feat_names=json.dumps(feat_names), feat_values=json.dumps(feat_values),
        jenkinsfile=jenkinsfile,
        env_block=env_block,
risk_stage=risk_stage,
post_block=post_block,
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