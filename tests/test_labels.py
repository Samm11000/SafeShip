"""
Tests for label provenance, weighting, and Sentinel's automatic reporting.

THE BUG THAT STARTED THIS
    /log decided how much a row counts in one inline expression:

        row["sample_weight"] = 1.0 if label_src in ["failure", "safe"] else 0.7

    Two integrations disagreed about the string. The dashboard's Groovy sent
    "safe" for a good deploy; safeship_ci sent "success". Only one was on that
    list, so **every successful deploy logged by safeship_ci was weighted 0.7** —
    the majority class in the dataset, quietly down-weighted by a typo, with
    nothing anywhere to reveal it.

WHY THE WEIGHTS DIFFER AT ALL
    A label is the only ground truth SafeShip gets, and the sources are not
    equally good evidence. Sentinel probing production is a measurement; a
    pipeline's exit status is an inference — and a pipeline that fails before the
    deploy step never deployed, so it says nothing about deployment risk.
"""
from __future__ import annotations

import json
import os
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
sys.path.insert(0, os.path.join(REPO, "sentinel"))

REGION = "ap-south-1"
MODEL_PATH = os.path.join(REPO, "ml", "data", "base_model.pkl")

import labels  # noqa: E402
import safeship_sentinel as sentinel  # noqa: E402


# ═══ the taxonomy ════════════════════════════════════════════════════════════

def test_an_observed_outcome_outweighs_an_inferred_one():
    """
    Sentinel measured production. A pipeline exit code guessed. The whole reason
    to record provenance is so the model can tell those apart.
    """
    assert labels.weight_for("sentinel_degraded") > labels.weight_for("ci_failure")
    assert labels.weight_for("sentinel_healthy") > labels.weight_for("ci_success")
    assert labels.weight_for("rollback") > labels.weight_for("ci_failure")


def test_an_assumption_is_the_weakest_evidence():
    # "nothing broke while we weren't looking" is worth having but must never
    # outvote something that actually looked.
    assumed = labels.weight_for("assumed_ok")
    assert assumed < labels.weight_for("ci_success")
    assert assumed < labels.weight_for("sentinel_healthy")
    assert assumed > 0, "it is still evidence, just weak"


def test_legacy_source_strings_still_work_and_are_canonicalised():
    """
    Old pipelines are still out there sending these. They must keep working, and
    land under one name so the stored vocabulary stays consistent.
    """
    assert labels.normalise("failure") == "ci_failure"
    assert labels.normalise("success") == "ci_success"
    assert labels.normalise("safe") == "ci_success"
    assert labels.normalise("risky") == "ci_failure"


def test_the_string_mismatch_that_started_this_is_fixed():
    """
    THE REGRESSION. "success" and "safe" meant the same thing, and only one of
    them was on the trusted list — so identical outcomes got different weights
    depending on which integration reported them.
    """
    assert labels.weight_for("success") == labels.weight_for("safe")
    assert labels.weight_for("failure") == labels.weight_for("ci_failure")
    # And neither of them silently lands on the unknown-source default.
    for legacy in ("failure", "success", "safe", "risky"):
        assert labels.weight_for(legacy) != labels.DEFAULT_WEIGHT


def test_case_and_whitespace_do_not_change_the_weight():
    assert labels.weight_for("  SENTINEL_DEGRADED  ") == labels.weight_for("sentinel_degraded")


def test_an_unknown_source_is_kept_but_trusted_less():
    """
    A new integration inventing a string is more likely than an attack, so the
    row is worth keeping — but it should not be trusted like a known source, and
    it should appear in the data as itself so it can be added deliberately.
    """
    assert labels.normalise("some_new_thing") == "some_new_thing"
    assert labels.weight_for("some_new_thing") == labels.DEFAULT_WEIGHT
    assert labels.DEFAULT_WEIGHT < min(labels.OBSERVED.values())


def test_a_missing_source_falls_back_to_manual():
    for empty in (None, "", "   "):
        assert labels.normalise(empty) == labels.DEFAULT_SOURCE


def test_synthetic_rows_keep_full_weight():
    # Regenerating the base model must not change behaviour as a side effect of
    # this taxonomy landing.
    assert labels.weight_for("synthetic") == 1.0
    assert labels.weight_for("seed") == 1.0


def test_every_weight_is_a_sane_fraction():
    for source, weight in labels.WEIGHTS.items():
        assert 0 < weight <= 1.0, f"{source} has weight {weight}"


def test_describe_reports_source_weight_and_whether_it_was_observed():
    canonical, weight, observed = labels.describe("failure")
    assert (canonical, weight, observed) == ("ci_failure", 0.7, False)

    canonical, weight, observed = labels.describe("sentinel_degraded")
    assert (canonical, weight, observed) == ("sentinel_degraded", 1.0, True)


def test_only_sources_that_watched_production_count_as_observed():
    assert labels.is_observed("sentinel_degraded")
    assert labels.is_observed("rollback")
    assert not labels.is_observed("ci_failure"), "a pipeline exit code is not an observation"
    assert not labels.is_observed("manual")
    assert not labels.is_observed("assumed_ok")


# ═══ Sentinel's own labelling ════════════════════════════════════════════════

class _Sent:
    """Captures what report_outcome would have POSTed."""

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def __call__(self, req, *a, **kw):
        self.calls.append({
            "url": req.full_url,
            "body": json.loads(req.data.decode()),
        })
        if self.fail:
            raise OSError("connection reset")

        class _R:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *exc):
                return False

            def read(self_inner):
                return b'{"status":"updated"}'

        return _R()


def _report(healthy):
    # Report.healthy is derived from verdict, so verdict is the only thing that
    # decides the label.
    return sentinel.Report(
        verdict="HEALTHY" if healthy else "DEGRADED",
        total=10, errors=0 if healthy else 6,
        error_rate=0.0 if healthy else 0.6,
        latency_p50=40.0, latency_p95=80.0, baseline_p50=38.0,
        reasons=[] if healthy else ["error rate 60% exceeds 20%"],
    )


def test_sentinel_label_sources_match_the_server_taxonomy():
    """
    Sentinel carries its own copies because it ships standalone into pipelines
    and cannot import server code. If they drifted, its labels would silently
    land on the unknown-source weight.
    """
    assert sentinel.LABEL_HEALTHY in labels.OBSERVED
    assert sentinel.LABEL_DEGRADED in labels.OBSERVED
    assert labels.weight_for(sentinel.LABEL_DEGRADED) == 1.0


def test_a_degraded_watch_reports_label_one():
    sent = _Sent()
    status = sentinel.report_outcome(_report(False), "https://ss.test", "t1", "k1",
                                     "b-1", opener=sent)
    body = sent.calls[0]["body"]
    assert sent.calls[0]["url"] == "https://ss.test/log"
    assert body["label"] == 1
    assert body["label_source"] == "sentinel_degraded"
    assert body["build_id"] == "b-1"
    assert "recorded" in status


def test_a_healthy_watch_also_reports():
    """
    Reporting only failures would be the worst possible sampling: successes are
    the majority class, and a model that has never seen a normal deploy cannot
    recognise an abnormal one.
    """
    sent = _Sent()
    sentinel.report_outcome(_report(True), "https://ss.test", "t1", "k1", "b-2",
                            opener=sent)
    body = sent.calls[0]["body"]
    assert body["label"] == 0
    assert body["label_source"] == "sentinel_healthy"


def test_labelling_is_skipped_without_credentials_rather_than_crashing():
    for url, tid, key in (("", "t", "k"), ("u", "", "k"), ("u", "t", "")):
        status = sentinel.report_outcome(_report(True), url, tid, key, "b-1")
        assert "skipped" in status


def test_labelling_is_skipped_without_a_build_id():
    status = sentinel.report_outcome(_report(True), "https://ss.test", "t", "k", "")
    assert "skipped" in status
    assert "build_id" in status


def test_a_failed_post_is_reported_but_never_raises():
    """
    Losing a label costs the model one row. Failing the pipeline costs the user a
    deploy. Those are not comparable, so this can never raise.
    """
    sent = _Sent(fail=True)
    status = sentinel.report_outcome(_report(False), "https://ss.test", "t1", "k1",
                                     "b-1", opener=sent)
    assert "could not record" in status


def test_the_build_id_is_read_from_the_file_the_gate_wrote(tmp_path):
    f = tmp_path / "safeship_build_id.txt"
    f.write_text("  dg-abc-123\n")
    assert sentinel.read_build_id(None, str(f)) == "dg-abc-123"
    # An explicit id wins over the file.
    assert sentinel.read_build_id("explicit", str(f)) == "explicit"
    # A missing file is not an error.
    assert sentinel.read_build_id(None, str(tmp_path / "nope.txt")) == ""


def test_the_exit_code_reflects_health_not_labelling(monkeypatch, tmp_path):
    """Labelling is bookkeeping; it must not decide whether the pipeline passes."""
    monkeypatch.setattr(sentinel, "run_watch", lambda **kw: _report(True))
    monkeypatch.setattr(sentinel, "report_outcome",
                        lambda *a, **kw: "could not record label (boom)")
    code = sentinel.main(["--url", "http://x/health", "--json",
                          "--build-id-file", str(tmp_path / "none.txt")])
    assert code == 0


def test_no_label_flag_disables_reporting(monkeypatch, capsys):
    monkeypatch.setattr(sentinel, "run_watch", lambda **kw: _report(True))
    called = []
    monkeypatch.setattr(sentinel, "report_outcome",
                        lambda *a, **kw: called.append(1) or "x")
    sentinel.main(["--url", "http://x/health", "--no-label", "--json"])
    assert not called
    assert "disabled" in json.loads(capsys.readouterr().out)["label"]


def test_credentials_come_from_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SAFESHIP_URL", "https://env.test")
    monkeypatch.setenv("SAFESHIP_TENANT_ID", "env-tenant")
    monkeypatch.setenv("SAFESHIP_API_KEY", "env-key")
    monkeypatch.setattr(sentinel, "run_watch", lambda **kw: _report(False))

    seen = {}
    monkeypatch.setattr(sentinel, "report_outcome",
                        lambda report, **kw: seen.update(kw) or "ok")
    f = tmp_path / "id.txt"
    f.write_text("dg-1")
    sentinel.main(["--url", "http://x/health", "--json", "--build-id-file", str(f)])

    assert seen["safeship_url"] == "https://env.test"
    assert seen["tenant_id"] == "env-tenant"
    assert seen["build_id"] == "dg-1"


# ═══ /log end to end ═════════════════════════════════════════════════════════

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
        creds = dynamo_client.create_tenant(email="labels@safeship.test")
        import main
        main.app.config.update(TESTING=True)
        yield {"client": main.app.test_client(), "creds": creds}


def _score_then_log(env, **log_extra):
    client, creds = env["client"], env["creds"]
    build_id = client.post("/score", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "diff_size": 100, "files_changed": 4, "is_hotfix": 0,
    }).get_json()["build_id"]

    r = client.post("/log", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "build_id": build_id, "label": 0, **log_extra,
    })
    assert r.status_code == 200, r.data

    s3 = boto3.client("s3", region_name=REGION)
    key = f"tenant_{creds['tenant_id']}/builds/{build_id}.json"
    row = json.loads(s3.get_object(Bucket="deploy-gate-data", Key=key)["Body"].read())
    return r.get_json(), row


def test_log_stores_the_weight_for_the_source(env):
    body, row = _score_then_log(env, label_source="sentinel_healthy")
    assert row["label_source"] == "sentinel_healthy"
    assert row["sample_weight"] == 1.0
    assert row["label_observed"] is True
    assert row["labelled_at"] > 0


def test_log_echoes_back_how_the_source_was_understood(env):
    """
    A label accepted at the unknown-source weight because of a typo should be
    visible to whoever sent it, not silently absorbed.
    """
    body, _ = _score_then_log(env, label_source="totally_made_up")
    assert body["label_source"] == "totally_made_up"
    assert body["sample_weight"] == labels.DEFAULT_WEIGHT
    assert body["observed"] is False


def test_a_success_from_safeship_ci_is_no_longer_penalised(env):
    """
    End-to-end version of the original bug: the exact string safeship_ci sends
    for a good deploy must not land on a reduced weight.
    """
    from safeship_ci import http as ci_http

    sent = {}

    def fake_post(url, path, payload, **kw):
        sent.update(payload)
        return {"status": "updated"}

    original, ci_http.post = ci_http.post, fake_post
    try:
        ci_http.log_outcome("https://ss.test", "t", "k", "b", 0)
    finally:
        ci_http.post = original

    source = sent["label_source"]
    assert source == "ci_success"
    assert labels.weight_for(source) != labels.DEFAULT_WEIGHT, (
        "the source safeship_ci sends is not recognised by the server"
    )

    _, row = _score_then_log(env, label_source=source)
    assert row["sample_weight"] == labels.weight_for("ci_success")


def test_sentinel_labels_are_worth_more_than_ci_labels_end_to_end(env):
    _, ci_row = _score_then_log(env, label_source="ci_success")
    _, sentinel_row = _score_then_log(env, label_source="sentinel_healthy")
    assert sentinel_row["sample_weight"] > ci_row["sample_weight"]


# ── an observation must not be overwritten by an inference ────────────────────

def _score(env):
    client, creds = env["client"], env["creds"]
    return client.post("/score", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "diff_size": 100, "files_changed": 4, "is_hotfix": 0,
    }).get_json()["build_id"]


def _log(env, build_id, label, source):
    creds = env["creds"]
    r = env["client"].post("/log", json={
        "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
        "build_id": build_id, "label": label, "label_source": source,
    })
    assert r.status_code == 200, r.data
    return r.get_json()


def _row(env, build_id):
    s3 = boto3.client("s3", region_name=REGION)
    key = f"tenant_{env['creds']['tenant_id']}/builds/{build_id}.json"
    return json.loads(s3.get_object(Bucket="deploy-gate-data", Key=key)["Body"].read())


def test_a_pipeline_fallback_cannot_overwrite_what_sentinel_observed(env):
    """
    The realistic sequence: Sentinel probes production, sees it degrade, records
    label=1. Then a post-build hook fires and reports the pipeline's own status.
    If the fallback won, "production degraded" would be replaced by "the pipeline
    was green" — throwing away the better evidence and inverting the label in
    exactly the case that matters most.
    """
    build_id = _score(env)
    _log(env, build_id, 1, "sentinel_degraded")

    body = _log(env, build_id, 0, "ci_success")

    assert body["status"] == "kept"
    assert body["label"] == 1
    row = _row(env, build_id)
    assert row["label"] == 1
    assert row["label_source"] == "sentinel_degraded"
    assert row["sample_weight"] == 1.0


def test_an_observation_can_still_correct_an_earlier_observation(env):
    # Two observations are equally trustworthy, so the later one wins — a rollback
    # after a watch window closed is genuinely newer information.
    build_id = _score(env)
    _log(env, build_id, 0, "sentinel_healthy")
    body = _log(env, build_id, 1, "rollback")

    assert body["status"] == "updated"
    assert _row(env, build_id)["label_source"] == "rollback"


def test_an_inference_can_label_a_build_nothing_observed(env):
    """
    The fallback has to work when it is the only thing that ran — if Sentinel
    never got the chance, an inferred label is much better than none.
    """
    build_id = _score(env)
    body = _log(env, build_id, 1, "ci_failure")
    assert body["status"] == "updated"
    assert _row(env, build_id)["label"] == 1


def test_a_manual_label_can_correct_an_inferred_one(env):
    # A human overruling a pipeline guess is the point of manual labelling.
    build_id = _score(env)
    _log(env, build_id, 0, "ci_success")
    body = _log(env, build_id, 1, "manual")
    assert body["status"] == "updated"
    assert _row(env, build_id)["label"] == 1
