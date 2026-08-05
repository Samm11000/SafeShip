"""
Guards how SafeShip handles features it could not measure, and guards
deployer_exp against being supplied by the caller.

The bugs these tests exist to prevent:

  1. An unmeasured feature was scored as a *perfect* one. `recent_failure_rate`
     defaulted to 0.0 and `test_pass_rate` to 1.0, while the training medians are
     0.261 and 0.819. Those two carry 52.8% of the model's weight, so any
     integration sending partial data always came back SAFE — which is precisely
     backwards for a risk product.

  2. `deployer_exp` came from the request. A caller could claim `deployer_exp=999`
     to look like a veteran and lower its own risk score.
"""
from __future__ import annotations

import os
import sys

import pytest

# Mock-AWS env must be set before boto3 clients are constructed anywhere.
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
    "RATE_LIMIT_ENABLED": "false",   # 429s would make these tests flaky
})

import boto3  # noqa: E402
from moto import mock_aws  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "app"))
sys.path.insert(0, os.path.join(REPO, "ml"))

REGION = "ap-south-1"
MODEL_PATH = os.path.join(REPO, "ml", "data", "base_model.pkl")

MINIMAL = {"hour_of_day": 14, "day_of_week": 2, "diff_size": 120,
           "files_changed": 4, "is_hotfix": 0, "triggered_by": "alice"}


def _ensure_infra():
    """
    Idempotent mock-AWS setup.

    test_endpoints.py owns a SESSION-scoped mock_aws, so when the whole suite
    runs this module's mock nests inside it and shares one moto backend — the
    buckets and table already exist. Creating them again raises
    BucketAlreadyOwnedByYou / ResourceInUseException, so both cases are tolerated
    and this fixture works standalone or as part of the suite.
    """
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
        creds = dynamo_client.create_tenant(email="imp@safeship.test")
        import main
        main.app.config.update(TESTING=True)
        yield {"client": main.app.test_client(), "creds": creds}


@pytest.fixture()
def client(env):
    return env["client"]


@pytest.fixture()
def creds(env):
    return env["creds"]


def _post(client, creds, **extra):
    body = {**MINIMAL, "tenant_id": creds["tenant_id"],
            "api_key": creds["api_key"], **extra}
    r = client.post("/score", json=body)
    assert r.status_code == 200, r.data
    return r.get_json()


# ── The core regression: unknown must not mean perfect ───────────────────────

def test_missing_features_are_reported_not_silently_defaulted(client, creds):
    body = _post(client, creds)
    assert "imputed" in body, "the response must disclose what was guessed"
    assert "recent_failure_rate" in body["imputed"]
    assert "test_pass_rate" in body["imputed"]


def test_unmeasured_test_pass_rate_is_not_treated_as_perfect():
    """The specific old bug: absent test_pass_rate became 1.0."""
    from imputation import TRAINING_MEDIANS, impute

    values, imputed, source = impute({"test_pass_rate": None}, tenant_id="nobody")
    assert "test_pass_rate" in imputed
    assert values["test_pass_rate"] != 1.0, "unknown must not mean 'all tests passed'"
    assert values["test_pass_rate"] == TRAINING_MEDIANS["test_pass_rate"]
    assert source["test_pass_rate"] == "training_median"


def test_unmeasured_failure_rate_is_not_treated_as_flawless():
    from imputation import TRAINING_MEDIANS, impute

    values, _, source = impute({"recent_failure_rate": None}, tenant_id="nobody")
    assert values["recent_failure_rate"] != 0.0, "unknown must not mean 'never fails'"
    assert values["recent_failure_rate"] == TRAINING_MEDIANS["recent_failure_rate"]
    assert source["recent_failure_rate"] == "training_median"


def test_training_medians_reflect_the_real_training_distribution():
    """
    These constants are measured from ml/data/synthetic_builds.csv, not invented.
    Imputing from a different distribution than the model was trained on is a
    quiet bug. tools/recompute_medians.py --check enforces this in CI.
    """
    from imputation import TRAINING_MEDIANS

    # Sanity: the two heavyweight features must sit in plausible ranges, and
    # notably NOT at the old optimistic extremes.
    assert 0.15 < TRAINING_MEDIANS["recent_failure_rate"] < 0.45
    assert 0.70 < TRAINING_MEDIANS["test_pass_rate"] < 0.95


def test_provided_values_are_never_overwritten():
    from imputation import impute

    given = {"diff_size": 777, "test_pass_rate": 0.5}
    values, imputed, source = impute(given, tenant_id="nobody")
    assert values["diff_size"] == 777
    assert values["test_pass_rate"] == 0.5
    assert "diff_size" not in imputed and "test_pass_rate" not in imputed
    assert source["diff_size"] == "provided"


def test_clock_features_come_from_the_server_clock():
    """Better than any median, and not spoofable by a client claiming it is 2pm."""
    import time

    from imputation import impute

    stamp = time.struct_time((2026, 8, 6, 9, 0, 0, 3, 218, 0))  # Thu 09:00
    values, imputed, source = impute({}, tenant_id="nobody", now=stamp)
    assert values["hour_of_day"] == 9
    assert values["day_of_week"] == 3
    assert source["hour_of_day"] == "clock"


def test_integer_features_stay_integers_after_imputation():
    from imputation import impute

    values, _, _ = impute({}, tenant_id="nobody")
    for f in ("diff_size", "files_changed", "deployer_exp", "is_hotfix"):
        assert isinstance(values[f], int), f"{f} imputed as {type(values[f])}"


# ── deployer_exp must be server-derived ──────────────────────────────────────

def test_deployer_exp_is_derived_from_server_history(client, creds):
    """After a few builds by one actor, the count comes from our own records."""
    for _ in range(3):
        _post(client, creds, triggered_by="carol")
    body = _post(client, creds, triggered_by="carol")
    assert body["feature_sources"]["deployer_exp"] == "server_history"


def test_spoofed_deployer_exp_is_ignored_once_history_exists(client, creds):
    """The attack: claim to be a veteran to lower your own risk score."""
    for _ in range(3):
        _post(client, creds, triggered_by="dave")

    honest = _post(client, creds, triggered_by="dave")
    spoofed = _post(client, creds, triggered_by="dave", deployer_exp=999)

    assert spoofed["feature_sources"]["deployer_exp"] == "server_history"
    # The claim must not have moved the needle in the caller's favour.
    assert spoofed["score"] >= honest["score"] - 1, (
        f"claiming deployer_exp=999 lowered the score "
        f"({honest['score']} -> {spoofed['score']})"
    )


def test_client_hint_is_accepted_only_for_a_brand_new_actor(client, creds):
    body = _post(client, creds, triggered_by="brand-new-person", deployer_exp=12)
    assert body["feature_sources"]["deployer_exp"] == "client_hint"


def test_actor_counter_is_atomic_and_per_actor():
    """A nested ADD, so two concurrent builds cannot lose an increment."""
    import dynamo_client

    tid = dynamo_client.create_tenant(email="counter@test")["tenant_id"]
    assert dynamo_client.actor_build_count(tid, "eve") == 0
    for expected in (1, 2, 3):
        assert dynamo_client.increment_actor_build(tid, "eve") == expected
    dynamo_client.increment_actor_build(tid, "frank")
    assert dynamo_client.actor_build_count(tid, "eve") == 3, "actors must not share a counter"
    assert dynamo_client.actor_build_count(tid, "frank") == 1


# ── The persisted row must record what was scored ────────────────────────────

def test_persisted_build_records_imputed_values_not_nulls(client, creds):
    """
    Retraining reads these rows. If they stored nulls, the model would learn from
    holes; if they stored only what the client sent, the row would not match what
    was actually scored.
    """
    import json

    body = _post(client, creds, triggered_by="grace")
    s3 = boto3.client("s3", region_name=REGION)
    key = f"tenant_{creds['tenant_id']}/builds/{body['build_id']}.json"
    row = json.loads(s3.get_object(Bucket="deploy-gate-data", Key=key)["Body"].read())

    assert row["test_pass_rate"] is not None
    assert row["recent_failure_rate"] is not None
    assert row["actor"] == "grace"
    assert "test_pass_rate" in row["imputed"]
