"""
Tests for the post-login redirect.

TWO THINGS AT ONCE
    The feature: asking for /setup while logged out should land you on /setup
    after logging in, not on /dashboard with the page you wanted simply gone.

    The hazard it introduces: "redirect to whatever ?next= says" is one of the
    oldest phishing setups there is. Our domain, our login page, a real
    authentication — and then the browser lands on the attacker's site having
    arrived from us. So most of this file is about what must NOT be honoured.

    The subtle case is `//evil.test`. It starts with "/", so the obvious
    `startswith("/")` guard passes it, and browsers resolve it as
    https://evil.test. That one is the reason this is a module and not an
    inline expression.
"""
from __future__ import annotations

import os
import sys
from urllib.parse import parse_qs, unquote, urlsplit

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

from redirects import login_url, safe_next  # noqa: E402


# ═══ what must never be honoured ═════════════════════════════════════════════

@pytest.mark.parametrize("hostile", [
    "https://evil.test/steal",
    "http://evil.test",
    "//evil.test/steal",          # protocol-relative — the one people miss
    "//evil.test",
    "/\\evil.test",               # browsers normalise /\ to //
    "/\\\\evil.test",
    "javascript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "\\\\evil.test\\share",
    "evil.test",                  # scheme-relative-ish, not a path
    "../../etc/passwd",
])
def test_an_external_target_is_refused(hostile):
    assert safe_next(hostile) == "/dashboard", (
        f"{hostile!r} would have sent an authenticated user off-site"
    )


def test_a_protocol_relative_url_is_refused_despite_starting_with_a_slash():
    """
    Called out on its own because it is the failure mode a naive check has.
    `"//evil.test".startswith("/")` is True, and the browser goes to evil.test.
    """
    assert "//evil.test".startswith("/")
    assert safe_next("//evil.test") == "/dashboard"


@pytest.mark.parametrize("smuggled", [
    "/setup\r\nSet-Cookie: admin=1",
    "/setup\nLocation: https://evil.test",
    "/setup\tx",
    "/setup\x00",
])
def test_control_characters_are_refused(smuggled):
    # Nothing here builds a header from this today, but a value that can hide
    # the rest of a log line should not survive validation.
    assert safe_next(smuggled) == "/dashboard"


def test_an_empty_or_missing_target_uses_the_default():
    for empty in (None, "", "   "):
        assert safe_next(empty) == "/dashboard"


def test_the_default_is_overridable():
    assert safe_next(None, default="/setup") == "/setup"
    assert safe_next("https://evil.test", default="/setup") == "/setup"


# ═══ what must be honoured ═══════════════════════════════════════════════════

@pytest.mark.parametrize("internal", [
    "/setup",
    "/setup?platform=github-actions",
    "/dashboard",
    "/",
    "/a/deeply/nested/path",
    "/setup?platform=bitbucket&x=1",
])
def test_an_internal_path_is_preserved(internal):
    assert safe_next(internal) == internal


# ═══ building the login URL ══════════════════════════════════════════════════

def _next_of(url):
    return unquote(parse_qs(urlsplit(url).query)["next"][0])


def test_login_url_round_trips_through_safe_next():
    assert _next_of(login_url("/setup")) == "/setup"


def test_login_url_carries_the_platform_being_viewed():
    url = login_url("/setup", {"platform": "bitbucket"})
    assert _next_of(url) == "/setup?platform=bitbucket"


def test_login_url_drops_parameters_that_are_not_allowlisted():
    """
    THE IMPORTANT ONE. /dashboard accepts tenant_id and api_key in the query
    string. Folding those into ?next= would write an API key into browser
    history, the Referer header on the next navigation, and every access log in
    between — a worse leak than the convenience is worth.
    """
    url = login_url("/dashboard", {"tenant_id": "t-123", "api_key": "sk-secret",
                                   "platform": "jenkins"})
    assert "sk-secret" not in url
    assert "t-123" not in url
    assert _next_of(url) == "/dashboard?platform=jenkins"


def test_login_url_refuses_to_encode_an_external_target():
    assert login_url("https://evil.test") == "/login"
    assert login_url("//evil.test") == "/login"


def test_login_url_escapes_the_target():
    # The next value lands inside a query string; it has to be encoded so it
    # cannot introduce parameters of its own.
    url = login_url("/setup?platform=github-actions")
    assert "?platform=github-actions" not in url.split("next=")[1] or "%3F" in url


# ═══ end to end through the app ══════════════════════════════════════════════

def _ensure_infra():
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
        creds = dynamo_client.create_tenant(email="redirect@safeship.test")
        import main
        main.app.config.update(TESTING=True)
        yield {"client": main.app.test_client(), "creds": creds}


@pytest.fixture()
def client(env):
    c = env["client"]
    c.get("/logout")
    return c


def test_asking_for_setup_while_logged_out_remembers_setup(client):
    r = client.get("/setup")
    assert r.status_code == 302
    location = r.headers["Location"]
    assert location.startswith("/login")
    assert _next_of(location) == "/setup"


def test_the_platform_being_viewed_survives_the_login_round_trip(client):
    r = client.get("/setup?platform=bitbucket")
    assert _next_of(r.headers["Location"]) == "/setup?platform=bitbucket"


def test_logging_in_returns_you_to_setup(client, env):
    """The actual complaint: you asked for /setup and got dumped on /dashboard."""
    creds = env["creds"]
    r = client.post("/login", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "next": "/setup?platform=github-actions",
    })
    assert r.status_code == 200
    assert r.get_json()["redirect"] == "/setup?platform=github-actions"


def test_logging_in_with_no_next_still_lands_on_the_dashboard(client, env):
    creds = env["creds"]
    r = client.post("/login", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
    })
    assert r.get_json()["redirect"] == "/dashboard"


def test_login_will_not_redirect_off_site_even_when_asked(client, env):
    """
    The phishing flow, end to end: an attacker sends a victim to
    /login?next=https://evil.test, the victim really does log in, and the
    browser follows our redirect to the attacker with our page as the referrer.
    """
    creds = env["creds"]
    r = client.post("/login", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "next": "https://evil.test/harvest",
    })
    assert r.get_json()["redirect"] == "/dashboard"

    r = client.post("/login", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "next": "//evil.test/harvest",
    })
    assert r.get_json()["redirect"] == "/dashboard"


def test_the_login_page_never_renders_a_hostile_next(client):
    body = client.get("/login?next=https://evil.test").get_data(as_text=True)
    assert "evil.test" not in body
    assert '"/dashboard"' in body


def test_the_login_page_renders_a_valid_next(client):
    body = client.get("/login?next=/setup").get_data(as_text=True)
    assert '"/setup"' in body


def test_an_already_logged_in_user_is_sent_straight_through(client, env):
    creds = env["creds"]
    client.post("/login", json={"tenant_id": creds["tenant_id"],
                                "api_key": creds["api_key"]})
    r = client.get("/login?next=/setup")
    assert r.status_code == 302
    assert r.headers["Location"] == "/setup"


def test_an_already_logged_in_user_cannot_be_bounced_off_site(client, env):
    creds = env["creds"]
    client.post("/login", json={"tenant_id": creds["tenant_id"],
                                "api_key": creds["api_key"]})
    r = client.get("/login?next=https://evil.test")
    assert r.headers["Location"] == "/dashboard"


def test_the_dashboard_never_leaks_credentials_into_the_login_url(client):
    """
    /dashboard takes tenant_id and api_key as query params for backward
    compatibility. A bad key must not end up echoed into ?next=, from where it
    would reach browser history, logs, and the Referer header.
    """
    r = client.get("/dashboard?tenant_id=t-abc&api_key=sk-should-not-leak")
    location = r.headers["Location"]
    assert "sk-should-not-leak" not in location
    assert "t-abc" not in location
    assert _next_of(location) == "/dashboard"
