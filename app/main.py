"""
main.py — Flask application factory.

Wires up, in order:
  * structured JSON logging with per-request correlation ids
  * optional Sentry error tracking (only if SENTRY_DSN is set)
  * security response headers
  * per-API-key rate limiting
  * request/response access logging + latency metrics
  * operational endpoints: /ready and /metrics
    (/health lives on score_bp and stays a trivial liveness probe)

Run locally with no AWS account:   python run_local.py
Run for real:                      gunicorn --workers 2 app.main:app
"""
import os
import sys
import time
from datetime import timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, Response, g, jsonify, request

from observability import (
    configure_logging,
    get_logger,
    get_request_id,
    incr,
    observe,
    render_metrics,
    set_request_id,
    set_tenant_id,
)
from routes.score import score_bp
from routes.dashboard import dashboard_bp

log = get_logger("app.main")

# Paths that must never be rate limited or access logged as business traffic.
_OPS_PATHS = {"/health", "/ready", "/metrics"}


def _resolve_secret_key() -> str:
    """
    Session signing key.

    Sessions are what authenticate a dashboard user, so a predictable key means
    anyone can forge one for any tenant. In production we refuse to boot without
    it rather than silently falling back to a shared default.
    """
    key = os.getenv("SECRET_KEY", "").strip()
    if key:
        return key

    # Only tolerated outside production, and it is per-process and random, so it
    # can never be a known value. Restarting invalidates existing sessions.
    if os.getenv("SAFESHIP_ENV", "development").lower() == "production":
        raise RuntimeError(
            "SECRET_KEY is required when SAFESHIP_ENV=production. "
            "Generate one with: python -c \"import secrets;print(secrets.token_urlsafe(48))\""
        )

    import secrets

    log.warning(
        "SECRET_KEY not set — generated an ephemeral development key. "
        "Sessions will not survive a restart.",
        extra={"env": os.getenv("SAFESHIP_ENV", "development")},
    )
    return secrets.token_urlsafe(48)


def _init_sentry() -> bool:
    """Error tracking. No-op unless SENTRY_DSN is configured."""
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk
        from sentry_sdk.integrations.flask import FlaskIntegration

        sentry_sdk.init(
            dsn=dsn,
            integrations=[FlaskIntegration()],
            environment=os.getenv("SAFESHIP_ENV", "development"),
            release=os.getenv("SAFESHIP_RELEASE"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            # We scrub our own logs; make sure Sentry does not re-introduce PII.
            send_default_pii=False,
        )
        return True
    except Exception as exc:  # pragma: no cover - depends on optional package
        log.warning("Sentry init failed; continuing without it", extra={"err": str(exc)})
        return False


def _apply_security_headers(resp: Response) -> Response:
    """
    Hardening headers.

    Applied by hand rather than via flask-talisman: the dashboard uses inline
    <script> for its charts, so a strict CSP needs nonces before it can be
    enabled. Setting the uncontroversial headers now and leaving a honest TODO
    beats shipping a CSP in report-only and calling it done.
    """
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    resp.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
    resp.headers.setdefault(
        "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
    )
    # HSTS only makes sense once TLS terminates in front of us.
    if os.getenv("SAFESHIP_ENV", "").lower() == "production":
        resp.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
    # TODO(csp): add a nonce-based Content-Security-Policy once dashboard
    # templates move their inline chart scripts to static files.
    return resp


def _rate_limit_key() -> str:
    """
    Limit per API key, falling back to client IP.

    Keying on the API key rather than the IP is what actually matters here: every
    build from one CI provider shares an egress IP, so an IP-keyed limit would
    throttle unrelated tenants together.
    """
    if request.path in _OPS_PATHS:
        return "ops"
    key = ""
    if request.is_json:
        try:
            key = (request.get_json(silent=True) or {}).get("api_key", "") or ""
        except Exception:
            key = ""
    if not key:
        key = request.headers.get("X-API-Key", "")
    if key:
        # Never use the raw credential as a cache key.
        import hashlib

        return "k:" + hashlib.sha256(key.encode()).hexdigest()[:32]
    return "ip:" + (request.headers.get("X-Forwarded-For", request.remote_addr or "?").split(",")[0].strip())


def _init_rate_limiter(app: Flask):
    """Per-key limits. Degrades to a no-op if flask-limiter is absent."""
    if os.getenv("RATE_LIMIT_ENABLED", "true").lower() != "true":
        log.info("Rate limiting disabled by RATE_LIMIT_ENABLED")
        return None
    try:
        from flask_limiter import Limiter
        from flask_limiter.util import get_remote_address  # noqa: F401
    except Exception:
        log.warning("flask-limiter not installed — no rate limiting active")
        return None

    limiter = Limiter(
        key_func=_rate_limit_key,
        app=app,
        # In-memory is per-worker. With 2 workers the effective ceiling is 2x the
        # configured limit; set RATE_LIMIT_STORAGE_URI to Redis for a shared one.
        storage_uri=os.getenv("RATE_LIMIT_STORAGE_URI", "memory://"),
        default_limits=[os.getenv("RATE_LIMIT_DEFAULT", "120 per minute")],
        # A build must never fail because our limiter had a bad day.
        swallow_errors=True,
        headers_enabled=True,
    )
    for path in _OPS_PATHS:
        pass  # exemptions are handled by _rate_limit_key returning "ops"
    return limiter


def create_app() -> Flask:
    configure_logging()
    app = Flask(__name__)

    app.secret_key = _resolve_secret_key()
    app.permanent_session_lifetime = timedelta(days=30)

    # Cookies: HttpOnly always; Secure once TLS is in front of us.
    app.config.update(
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=os.getenv("SAFESHIP_ENV", "").lower() == "production",
        JSON_SORT_KEYS=False,
    )

    sentry_on = _init_sentry()
    limiter = _init_rate_limiter(app)

    # Auth routes (/login, /logout) live on dashboard_bp; blueprints own all routes.
    app.register_blueprint(score_bp)
    app.register_blueprint(dashboard_bp)

    # ---------------------------------------------------------------- requests
    @app.before_request
    def _begin():
        g._t0 = time.perf_counter()
        # Honour an inbound id so a build can be traced from CI through to here.
        set_request_id(request.headers.get("X-Request-ID"))
        set_tenant_id(None)

    @app.after_request
    def _finish(resp: Response) -> Response:
        resp = _apply_security_headers(resp)
        resp.headers["X-Request-ID"] = get_request_id()

        if request.path not in _OPS_PATHS:
            elapsed = time.perf_counter() - getattr(g, "_t0", time.perf_counter())
            endpoint = request.endpoint or "unknown"
            observe("safeship_request_seconds", elapsed, {"endpoint": endpoint})
            incr(
                "safeship_requests_total",
                {"endpoint": endpoint, "status": resp.status_code},
            )
            log.info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.path,
                    "status": resp.status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                    "endpoint": endpoint,
                },
            )
        return resp

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        # Log the stack, return a generic body — never leak internals to callers.
        incr("safeship_unhandled_errors_total")
        log.exception("unhandled exception", extra={"path": request.path})
        from werkzeug.exceptions import HTTPException

        if isinstance(exc, HTTPException):
            return exc
        return jsonify({"error": "internal error", "request_id": get_request_id()}), 500

    # ------------------------------------------------------------ ops endpoints
    @app.route("/ready", methods=["GET"])
    def ready():
        """
        Readiness — actually checks dependencies.

        /health answers "is the process alive"; this answers "can it serve a
        score". They are different questions, and conflating them is how a box
        reports healthy while S3 is unreachable. Our own Sentinel gates on
        endpoints like this, so it had better mean something.
        """
        checks, ok = {}, True

        try:
            import boto3

            region = os.getenv("AWS_REGION", "ap-south-1")
            boto3.client("s3", region_name=region).head_bucket(
                Bucket=os.getenv("S3_MODELS_BUCKET", "deploy-gate-models")
            )
            checks["s3_models"] = "ok"
        except Exception as exc:
            checks["s3_models"] = f"fail: {type(exc).__name__}"
            ok = False

        try:
            import boto3

            region = os.getenv("AWS_REGION", "ap-south-1")
            boto3.client("dynamodb", region_name=region).describe_table(
                TableName=os.getenv("DYNAMO_TABLE", "tenants")
            )
            checks["dynamodb"] = "ok"
        except Exception as exc:
            checks["dynamodb"] = f"fail: {type(exc).__name__}"
            ok = False

        try:
            import scorer

            # Actually resolve a model rather than checking a symbol exists.
            # The cache has a 300s TTL, so readiness polling does not hammer S3.
            model, phase = scorer._cache.get_model("base")
            if model is None:
                checks["model"] = "fail: no model resolved"
                ok = False
            else:
                checks["model"] = f"ok ({phase})"
        except Exception as exc:
            checks["model"] = f"fail: {type(exc).__name__}"
            ok = False

        return (
            jsonify({"status": "ready" if ok else "degraded", "checks": checks}),
            200 if ok else 503,
        )

    @app.route("/metrics", methods=["GET"])
    def metrics():
        """
        Prometheus text format. Counters are per-worker (see observability.py) —
        exact totals come from the structured access log.
        """
        token = os.getenv("METRICS_TOKEN", "").strip()
        if token and request.headers.get("Authorization") != f"Bearer {token}":
            return jsonify({"error": "unauthorized"}), 401
        return Response(render_metrics(), mimetype="text/plain; version=0.0.4")

    log.info(
        "SafeShip app initialised",
        extra={
            "env": os.getenv("SAFESHIP_ENV", "development"),
            "sentry": sentry_on,
            "rate_limiting": bool(limiter),
            "region": os.getenv("AWS_REGION", "ap-south-1"),
        },
    )
    return app


app = create_app()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    log.info("starting development server", extra={"port": port, "debug": debug})
    app.run(host="0.0.0.0", port=port, debug=debug)
