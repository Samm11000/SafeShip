"""
Tests for the platform-aware onboarding wizard and the integration snippets.

THE BUG THESE REPLACE
    Signup handed every user the same Jenkins Groovy, whatever CI they actually
    ran, generated inline in the middle of a dashboard route — and that snippet
    hardcoded seven of the ten features:

        "files_changed":5, "recent_failure_rate":0.0, "test_pass_rate":1.0,
        "deployer_exp":30, "days_since_deploy":2, "build_time_delta":0.0

    So the official instructions produced a score that barely depended on the
    build. The strongest test in this file is therefore a negative one: no
    generated snippet may contain a hardcoded feature value ever again.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys

import pytest

os.environ.update({
    "AWS_ACCESS_KEY_ID": "testing",
    "AWS_SECRET_ACCESS_KEY": "testing",
    "AWS_SECURITY_TOKEN": "testing",
    "AWS_SESSION_TOKEN": "testing",
    "AWS_DEFAULT_REGION": "ap-south-1",
    "AWS_REGION": "ap-south-1",
    "S3_MODELS_BUCKET": "deploy-gate-models",
    "S3_DATA_BUCKET": "deploy-gate-data",
    "DYNAMO_TABLE": "tenants",
    "SECRET_KEY": "test-secret",
    "RATE_LIMIT_ENABLED": "false",
})

import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "ml"))

REGION = "ap-south-1"
MODEL_PATH = os.path.join(REPO, "ml", "data", "base_model.pkl")

import integrations  # noqa: E402
from features import FEATURES  # noqa: E402

TENANT = "t" * 16
KEY = "k" * 32
BASE = "https://safeship.example"


def _ensure_infra():
    """Idempotent — see the note in tests/test_imputation.py."""
    s3 = boto3.client("s3", region_name=REGION)
    for b in ("deploy-gate-models", "deploy-gate-data"):
        try:
            s3.create_bucket(Bucket=b,
                             CreateBucketConfiguration={"LocationConstraint": REGION})
        except s3.exceptions.ClientError as exc:
            if exc.response["Error"]["Code"] not in (
                "BucketAlreadyOwnedByYou", "BucketAlreadyExists"
            ):
                raise
    with open(MODEL_PATH, "rb") as f:
        s3.put_object(Bucket="deploy-gate-models", Key="base/model.pkl", Body=f.read())

    ddb = boto3.client("dynamodb", region_name=REGION)
    try:
        ddb.create_table(
            TableName="tenants",
            KeySchema=[{"AttributeName": "tenant_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "tenant_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
    except ddb.exceptions.ResourceInUseException:
        pass


@pytest.fixture(scope="module")
def env():
    with mock_aws():
        _ensure_infra()
        import dynamo_client
        creds = dynamo_client.create_tenant(email="setup@safeship.test")
        import main
        main.app.config.update(TESTING=True)
        yield {"client": main.app.test_client(), "creds": creds,
               "dynamo": dynamo_client}


@pytest.fixture()
def client(env):
    return env["client"]


@pytest.fixture()
def creds(env):
    return env["creds"]


def _login(client, creds):
    r = client.post("/login", json={"tenant_id": creds["tenant_id"],
                                    "api_key": creds["api_key"]})
    assert r.status_code == 200, r.data
    return r


ALL = list(integrations.ORDER)


def _described(platform):
    return integrations.describe(platform, TENANT, KEY, BASE)


# ═══ the registry ════════════════════════════════════════════════════════════

def test_the_three_platforms_are_registered_in_display_order():
    # GitHub first: it needs the least setup, so it is the best default path.
    assert ALL == ["github-actions", "jenkins", "bitbucket"]


def test_platform_ids_match_the_extractor_adapter_names():
    """
    The id is written to the tenant record and is the same string safeship_ci
    detects on. If these diverged, a tenant could be onboarded onto a platform
    the extractor has no adapter for.
    """
    from safeship_ci.adapters import BY_NAME

    for platform in ALL:
        assert platform in BY_NAME, (
            f"{platform!r} has no matching safeship_ci adapter "
            f"(adapters: {sorted(BY_NAME)})"
        )


def test_an_unknown_platform_is_rejected_not_guessed():
    assert integrations.is_valid("teamcity") is False
    with pytest.raises(integrations.UnknownPlatform):
        integrations.get("teamcity")


def test_platform_ids_are_matched_case_insensitively():
    assert integrations.get("GitHub-Actions").ID == "github-actions"
    assert integrations.is_valid("  JENKINS  ") is True


def test_summaries_carry_no_credentials():
    # The picker renders before a platform is chosen and needs no secrets.
    blob = repr(integrations.summaries())
    assert TENANT not in blob and KEY not in blob


# ═══ the shape every platform must satisfy ═══════════════════════════════════

@pytest.mark.parametrize("platform", ALL)
def test_every_platform_returns_the_same_shape(platform):
    """setup.html is platform-agnostic, so a missing key is a broken page."""
    d = _described(platform)
    for field in ("id", "name", "tagline", "docs_url", "setup_effort",
                  "secrets", "secrets_location", "prerequisites",
                  "history_note", "snippet", "verify_hint"):
        assert field in d, f"{platform} is missing {field}"

    assert d["id"] == platform
    for key in ("filename", "language", "code"):
        assert d["snippet"][key], f"{platform} snippet has no {key}"


@pytest.mark.parametrize("platform", ALL)
def test_the_credentials_are_handed_over_verbatim(platform):
    d = _described(platform)
    by_name = {s["name"]: s for s in d["secrets"]}

    assert by_name["SAFESHIP_TENANT_ID"]["value"] == TENANT
    assert by_name["SAFESHIP_API_KEY"]["value"] == KEY
    assert by_name["SAFESHIP_URL"]["value"] == BASE
    for secret in d["secrets"]:
        assert secret["where"], f"{secret['name']} does not say where it goes"


@pytest.mark.parametrize("platform", ALL)
def test_every_prerequisite_explains_itself_and_shows_the_fix(platform):
    """
    A prerequisite without a "why" gets skipped, and every one of these fails
    SILENTLY — you still get a score, it is just built on medians.
    """
    prereqs = _described(platform)["prerequisites"]
    assert prereqs, f"{platform} lists no prerequisites"
    for pre in prereqs:
        assert pre["title"].strip()
        assert len(pre["why"]) > 60, f"{platform}: {pre['title']!r} barely explains itself"
        assert pre["fix"].strip()


# ═══ THE REGRESSION: no hardcoded features in any snippet ════════════════════

@pytest.mark.parametrize("platform", ALL)
def test_no_snippet_hardcodes_a_feature_value(platform):
    """
    The whole reason this package exists. The old Groovy shipped
    "recent_failure_rate":0.0 and "test_pass_rate":1.0 — the two heaviest
    features, pinned to their most reassuring values, in the official
    instructions.
    """
    code = _described(platform)["snippet"]["code"]
    for feature in FEATURES:
        assert f'"{feature}"' not in code, (
            f"{platform} snippet hardcodes {feature}"
        )
        # Also catch YAML/shell spellings like `diff_size: 120` or --diff-size 120
        assert not re.search(rf"\b{feature}\s*[:=]\s*[-\d.]", code), (
            f"{platform} snippet assigns a literal to {feature}"
        )
        assert not re.search(rf"--{feature.replace('_', '-')}\s+[-\d.]", code), (
            f"{platform} snippet passes a literal --{feature} flag"
        )


@pytest.mark.parametrize("platform", ALL)
def test_every_snippet_invokes_the_real_extractor(platform):
    code = _described(platform)["snippet"]["code"]
    assert ("safeship_ci" in code or "action/gate" in code), (
        f"{platform} snippet does not call safeship_ci — it is measuring nothing"
    )
    # The old snippets curl'd /score with a hand-built JSON body.
    assert "curl" not in code, f"{platform} snippet still hand-rolls the request"


@pytest.mark.parametrize("platform", ALL)
def test_every_snippet_closes_the_learning_loop(platform):
    """Without an outcome label the model never learns from a real deploy."""
    d = _described(platform)
    code = d["snippet"]["code"]
    assert ("safeship_ci log" in code or "mode:  log" in code
            or "mode: log" in code), f"{platform} never reports the outcome"


@pytest.mark.parametrize("platform", ALL)
def test_every_snippet_starts_advisory(platform):
    """
    Enforcing on day one, on a model the user has not learned to trust, is how a
    gate gets removed on day two.
    """
    code = _described(platform)["snippet"]["code"]
    assert "fail-open" in code or "--fail-open" in code


# ═══ the platform-specific traps ═════════════════════════════════════════════

def test_github_documents_the_two_invisible_prerequisites():
    d = _described("github-actions")
    blob = " ".join(p["title"] + p["why"] + p["fix"] for p in d["prerequisites"])
    assert "actions: read" in blob
    assert "fetch-depth" in blob
    # And the snippet the user actually pastes must already contain both.
    code = d["snippet"]["code"]
    assert "actions: read" in code
    assert "fetch-depth: 2" in code


def test_jenkins_asks_for_an_api_token_not_a_password():
    d = _described("jenkins")
    blob = " ".join(p["why"] + p["fix"] for p in d["prerequisites"])
    assert "API token" in blob
    assert "never" in blob and "password" in blob


def test_jenkins_uses_its_native_test_api():
    # Jenkins is the only platform that has one; not using it would be a waste.
    d = _described("jenkins")
    blob = " ".join(p["title"] + p["fix"] for p in d["prerequisites"])
    assert "junit" in blob.lower()


def test_bitbucket_explains_the_repository_access_token_and_clone_depth():
    d = _described("bitbucket")
    blob = " ".join(p["title"] + p["why"] + p["fix"] for p in d["prerequisites"])
    assert "Repository Access Token" in blob
    assert "revocable" in blob, "the reason to prefer it over an app password"
    assert "depth" in blob
    assert "depth: 2" in d["snippet"]["code"]


def test_bitbucket_sorts_pipelines_the_only_way_that_works():
    # -completed_on returns invalid-sort-attribute; the adapter uses -created_on.
    # Nothing in the snippet should suggest otherwise.
    assert "completed_on" not in _described("bitbucket")["snippet"]["code"]


# ═══ the base URL ════════════════════════════════════════════════════════════

def test_the_public_url_is_configurable(monkeypatch):
    # A wizard whose value is being copy-paste trustworthy cannot ship a
    # hardcoded IP with no way to change it.
    monkeypatch.setenv("SAFESHIP_PUBLIC_URL", "https://safeship.io/")
    assert integrations.public_base_url() == "https://safeship.io"


def test_the_public_url_has_a_default(monkeypatch):
    monkeypatch.delenv("SAFESHIP_PUBLIC_URL", raising=False)
    assert integrations.public_base_url().startswith("http")


# ═══ the wizard routes ═══════════════════════════════════════════════════════

def test_setup_requires_a_login(client):
    client.get("/logout")
    r = client.get("/setup")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_status_requires_a_login(client):
    client.get("/logout")
    assert client.get("/api/setup/status").status_code == 401


def test_setup_offers_all_three_platforms_before_one_is_chosen(client, creds):
    _login(client, creds)
    body = client.get("/setup").get_data(as_text=True)
    for platform in ALL:
        assert integrations.get(platform).NAME in body
    # Nothing platform-specific should be rendered yet.
    assert "Add these secrets" not in body


def test_choosing_a_platform_persists_it(client, creds, env):
    _login(client, creds)
    r = client.post("/setup/platform", json={"platform": "bitbucket"})
    assert r.status_code == 200
    assert r.get_json()["platform"] == "bitbucket"

    tenant = env["dynamo"].get_tenant(creds["tenant_id"])
    assert tenant["ci_platform"] == "bitbucket"


def test_an_unknown_platform_is_a_400_listing_the_valid_ones(client, creds):
    _login(client, creds)
    r = client.post("/setup/platform", json={"platform": "teamcity"})
    assert r.status_code == 400
    assert set(r.get_json()["expected"]) == set(ALL)


def test_the_chosen_platform_drives_the_rendered_instructions(client, creds):
    _login(client, creds)
    client.post("/setup/platform", json={"platform": "jenkins"})

    body = client.get("/setup").get_data(as_text=True)
    assert "Jenkinsfile" in body
    assert creds["api_key"] in body, "the user needs their key to paste it"
    # And not another platform's instructions.
    assert "bitbucket-pipelines.yml" not in body


def test_the_platform_can_be_previewed_without_being_saved(client, creds, env):
    _login(client, creds)
    client.post("/setup/platform", json={"platform": "jenkins"})

    body = client.get("/setup?platform=github-actions").get_data(as_text=True)
    assert "actions: read" in body
    # The saved choice is unchanged — ?platform= only previews.
    assert env["dynamo"].get_tenant(creds["tenant_id"])["ci_platform"] == "jenkins"


def test_a_garbage_platform_in_the_query_falls_back_to_the_picker(client, creds):
    _login(client, creds)
    body = client.get("/setup?platform=<script>").get_data(as_text=True)
    assert "Pick a platform" in body


def test_a_plain_http_endpoint_is_called_out(client, creds, monkeypatch):
    """
    The wizard asks the user to paste an API key into their pipeline. Over plain
    HTTP that key crosses the network in cleartext on every build, and a wizard
    that stays quiet about it has not earned the trust it is asking for.
    """
    monkeypatch.setenv("SAFESHIP_PUBLIC_URL", "http://54.89.160.150")
    _login(client, creds)
    client.post("/setup/platform", json={"platform": "github-actions"})
    body = client.get("/setup").get_data(as_text=True)
    assert "plain HTTP" in body
    assert "cleartext" in body


def test_no_warning_when_the_endpoint_is_https(client, creds, monkeypatch):
    monkeypatch.setenv("SAFESHIP_PUBLIC_URL", "https://safeship.example")
    _login(client, creds)
    client.post("/setup/platform", json={"platform": "github-actions"})
    body = client.get("/setup").get_data(as_text=True)
    assert "plain HTTP" not in body


# ═══ step 4: verification ════════════════════════════════════════════════════

def test_a_fresh_tenant_is_not_connected(client, env):
    import dynamo_client
    fresh = dynamo_client.create_tenant(email="fresh@safeship.test")
    _login(client, fresh)

    body = client.get("/api/setup/status").get_json()
    assert body["connected"] is False
    assert body["build_count"] == 0
    assert body["latest"] is None


def test_status_flips_to_connected_after_a_real_score(client, env):
    """
    The step that turns "I pasted some YAML" into "it works". Reporting the
    actual score proves the whole path ran, not merely that a request arrived.
    """
    import dynamo_client
    fresh = dynamo_client.create_tenant(email="connects@safeship.test")

    r = client.post("/score", json={
        "tenant_id": fresh["tenant_id"], "api_key": fresh["api_key"],
        "diff_size": 90, "files_changed": 3, "is_hotfix": 0,
        "job_name": "checkout-service", "branch_name": "main",
        "triggered_by": "someone",
    })
    assert r.status_code == 200, r.data

    _login(client, fresh)
    body = client.get("/api/setup/status").get_json()

    assert body["connected"] is True
    assert body["build_count"] >= 1
    assert body["latest"]["score"] == r.get_json()["score"]
    assert body["latest"]["verdict"] in ("SAFE", "WARNING", "BLOCKED")
    assert body["latest"]["job_name"] == "checkout-service"


def test_status_discloses_what_was_estimated(client, env):
    """
    A "Connected" banner over a score built entirely on medians would be
    technically true and practically misleading.
    """
    import dynamo_client
    fresh = dynamo_client.create_tenant(email="partial@safeship.test")
    client.post("/score", json={
        "tenant_id": fresh["tenant_id"], "api_key": fresh["api_key"],
        "diff_size": 90, "files_changed": 3, "is_hotfix": 0,
    })
    _login(client, fresh)
    latest = client.get("/api/setup/status").get_json()["latest"]
    assert latest["imputed"], "several features were not sent and must be disclosed"


# ═══ the dashboard, after the inline Groovy was removed ══════════════════════

def test_the_dashboard_no_longer_hardcodes_features():
    """
    dashboard.py used to build ~80 lines of Groovy inline with seven feature
    values baked in. It is gone; this makes sure it does not come back.
    """
    src = open(os.path.join(REPO, "app", "routes", "dashboard.py"),
               "r", encoding="utf-8").read()
    # Strip the comment that explains the removal, which legitimately names them.
    body = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
    for literal in ('"recent_failure_rate":0.0', '"test_pass_rate":1.0',
                    '"deployer_exp":30', '"files_changed":5'):
        assert literal not in body, f"the hardcoded snippet is back: {literal}"
    assert "readJSON" not in body, "inline Groovy is back in the route"


def test_the_dashboard_prompts_setup_when_no_platform_is_chosen(client, env):
    import dynamo_client
    fresh = dynamo_client.create_tenant(email="noplatform@safeship.test")
    _login(client, fresh)
    body = client.get("/dashboard").get_data(as_text=True)
    assert "/setup" in body
    assert "Connect your pipeline" in body


def test_the_dashboard_shows_only_the_chosen_platform(client, creds):
    _login(client, creds)
    client.post("/setup/platform", json={"platform": "github-actions"})
    body = client.get("/dashboard").get_data(as_text=True)

    assert "GitHub Actions" in body
    assert "Jenkinsfile" not in body
    assert "Change platform" in body


def test_the_retrain_threshold_shown_matches_the_one_enforced():
    """
    The dashboard's "progress to your own model" bar divided by 5 — the old
    MIN_BUILDS — so it hit 100% at five labelled builds while retraining now
    requires 200. A progress bar that lies is worse than no progress bar.
    """
    spec = importlib.util.spec_from_file_location(
        "retrain_handler_for_setup",
        os.path.join(REPO, "lambda", "retrain", "handler.py"))
    handler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler)

    from routes.dashboard import RETRAIN_MIN_BUILDS

    assert RETRAIN_MIN_BUILDS == handler.MIN_BUILDS, (
        "app/routes/dashboard.py::RETRAIN_MIN_BUILDS has drifted from "
        "lambda/retrain/handler.py::MIN_BUILDS"
    )
