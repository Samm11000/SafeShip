"""
End-to-end endpoint verification for the SafeShip Flask API.

Uses moto to mock S3 + DynamoDB entirely in-memory, so every endpoint
(including /score, /log, /signup, /dashboard) can be exercised locally
with no real AWS credentials or infrastructure.

Run:
    python3 -m pytest tests/test_endpoints.py -v
"""

import os
import sys
import json

# ── AWS env MUST be set before any boto3 client is created ──────────────────
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
    "SECRET_KEY":            "test-secret",
})

import boto3
import pytest
from moto import mock_aws

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "ml"))

REGION     = "ap-south-1"
MODEL_PATH = os.path.join(REPO, "ml", "data", "base_model.pkl")

# A full, valid feature payload (hour_of_day / day_of_week are required).
SAMPLE = {
    "hour_of_day": 14, "day_of_week": 2,
    "diff_size": 120, "files_changed": 4,
    "recent_failure_rate": 0.1, "test_pass_rate": 0.95, "is_hotfix": 0,
    "deployer_exp": 20, "days_since_deploy": 2.0, "build_time_delta": 0.0,
    "job_name": "qa-job", "branch_name": "main", "triggered_by": "qa",
}


@pytest.fixture(scope="session")
def env():
    """Stands up mock AWS, seeds the base model + a tenant, builds a test client."""
    with mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        for bucket in ("deploy-gate-models", "deploy-gate-data"):
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": REGION},
            )
        with open(MODEL_PATH, "rb") as f:
            s3.put_object(Bucket="deploy-gate-models", Key="base/model.pkl", Body=f.read())

        boto3.client("dynamodb", region_name=REGION).create_table(
            TableName="tenants",
            KeySchema=[{"AttributeName": "tenant_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "tenant_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )

        # Imported INSIDE the mock so scorer's module-level boto3 client is mocked.
        import dynamo_client
        creds = dynamo_client.create_tenant(email="qa@safeship.test")

        import main
        main.app.config.update(TESTING=True)

        yield {"client": main.app.test_client(), "creds": creds, "app": main.app}


@pytest.fixture()
def client(env):
    return env["client"]


@pytest.fixture()
def creds(env):
    return env["creds"]


def _auth(creds, **extra):
    return {**SAMPLE, "tenant_id": creds["tenant_id"], "api_key": creds["api_key"], **extra}


@pytest.fixture()
def scored_build(client, creds):
    """Posts a real /score and returns its build_id (for /log)."""
    r = client.post("/score", json=_auth(creds))
    assert r.status_code == 200, r.data
    return r.get_json()["build_id"]


# ── Health ──────────────────────────────────────────────────────────────────
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


# ── Static pages render ───────────────────────────────────────────────────────
@pytest.mark.parametrize("path", ["/", "/about", "/demo", "/login", "/signup"])
def test_static_pages(client, path):
    r = client.get(path)
    assert r.status_code == 200, f"{path} -> {r.status_code}"
    assert b"<" in r.data  # rendered HTML


def test_logout_redirects(client):
    r = client.get("/logout")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


# ── Demo scoring (no auth; route injects demo tenant) ─────────────────────────
def test_demo_score(client):
    r = client.post("/demo/score", json=SAMPLE)
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert "score" in body and "verdict" in body and "top_reasons" in body
    assert 0 <= body["score"] <= 100


# ── Signup ────────────────────────────────────────────────────────────────────
def test_signup(client):
    r = client.post("/signup", json={"email": "new-tenant@safeship.test"})
    assert r.status_code == 201, r.data
    body = r.get_json()
    assert body["tenant_id"] and body["api_key"]


# ── Login ─────────────────────────────────────────────────────────────────────
def test_login_ok(client, creds):
    r = client.post("/login", json={"tenant_id": creds["tenant_id"], "api_key": creds["api_key"]})
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["success"] is True
    assert body["redirect"] == "/dashboard"


def test_login_bad_credentials(client, creds):
    r = client.post("/login", json={"tenant_id": creds["tenant_id"], "api_key": "wrong-key"})
    assert r.status_code == 401


# ── Scoring ───────────────────────────────────────────────────────────────────
def test_score_ok(client, creds):
    r = client.post("/score", json=_auth(creds))
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert 0 <= body["score"] <= 100
    assert body["verdict"] in ("SAFE", "WARNING", "BLOCKED")
    assert body["build_id"]


def test_score_bad_credentials(client, creds):
    r = client.post("/score", json=_auth(creds, api_key="wrong-key"))
    assert r.status_code == 401


def test_score_validation_returns_422(client, creds):
    """hour_of_day=99 is out of range -> ValidationError -> 422 (NOT a 500/NameError)."""
    r = client.post("/score", json=_auth(creds, hour_of_day=99))
    assert r.status_code == 422, r.data
    assert "details" in r.get_json()


# ── Outcome logging ───────────────────────────────────────────────────────────
def test_log_outcome(client, creds, scored_build):
    r = client.post("/log", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "build_id": scored_build, "label": 0, "label_source": "success",
    })
    assert r.status_code == 200, r.data
    body = r.get_json()
    assert body["status"] == "updated"
    assert body["build_id"] == scored_build


# ── Storage: append-only per-build objects, no whole-CSV rewrite ──────────────
def test_score_writes_per_build_object(client, creds):
    tid = creds["tenant_id"]
    r = client.post("/score", json=_auth(creds))
    build_id = r.get_json()["build_id"]

    s3 = boto3.client("s3", region_name=REGION)
    # The build is its own small object...
    head = s3.head_object(Bucket="deploy-gate-data", Key=f"tenant_{tid}/builds/{build_id}.json")
    assert head["ContentLength"] < 4096  # tiny, not a growing history file

    # ...and the old O(history) data.csv is NOT created/rewritten.
    listing = s3.list_objects_v2(Bucket="deploy-gate-data", Prefix=f"tenant_{tid}/")
    keys = [o["Key"] for o in listing.get("Contents", [])]
    assert f"tenant_{tid}/data.csv" not in keys
    assert any(k.startswith(f"tenant_{tid}/builds/") for k in keys)


def test_log_updates_only_that_build_object(client, creds):
    tid = creds["tenant_id"]
    build_id = client.post("/score", json=_auth(creds)).get_json()["build_id"]
    s3 = boto3.client("s3", region_name=REGION)
    key = f"tenant_{tid}/builds/{build_id}.json"

    before = json.loads(s3.get_object(Bucket="deploy-gate-data", Key=key)["Body"].read())
    assert before["label"] == -1  # pending

    client.post("/log", json={
        "tenant_id": tid, "api_key": creds["api_key"],
        "build_id": build_id, "label": 1, "label_source": "failure",
    })
    after = json.loads(s3.get_object(Bucket="deploy-gate-data", Key=key)["Body"].read())
    assert after["label"] == 1
    assert after["label_source"] == "failure"


def test_log_unknown_build_404(client, creds):
    r = client.post("/log", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "build_id": "does-not-exist", "label": 0,
    })
    assert r.status_code == 404


# ── Dashboard (URL-param auth) ────────────────────────────────────────────────
def test_dashboard(client, creds):
    r = client.get(f"/dashboard?tenant_id={creds['tenant_id']}&api_key={creds['api_key']}")
    assert r.status_code == 200, r.data
    assert b"<" in r.data


# ── Settings ──────────────────────────────────────────────────────────────────
def test_settings(client, creds):
    r = client.post("/settings", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "threshold_yellow": 45, "threshold_red": 75,
    })
    assert r.status_code == 200, r.data
    assert r.get_json()["status"] == "saved"


# ── Regression: the /login route collision must be gone ───────────────────────
def test_no_login_route_collision(env):
    post_rules = [
        r for r in env["app"].url_map.iter_rules()
        if str(r) == "/login" and "POST" in r.methods
    ]
    assert len(post_rules) == 1, f"expected one POST /login rule, got {[r.endpoint for r in post_rules]}"
    assert post_rules[0].endpoint == "dashboard.login"
