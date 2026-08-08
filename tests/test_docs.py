"""
Tests that the documentation describes the software that exists.

Docs drift silently and nothing compiles them, which is how the README ended up
telling people to run `safeship_gate.py` — a file that sent 5 of 10 features as
hardcoded constants and no longer exists at all. Since the whole point of the
recent work is that SafeShip should not claim more than it measured, a README
that overstates is the same bug in prose.

These are cheap structural checks, not prose review:

  - the documented /score response keys really come back from /score
  - the copy-paste snippets do not hardcode feature values
  - references to files and action paths actually resolve
  - the numbers quoted for the model carry their caveat
"""
from __future__ import annotations

import json
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

from features import FEATURES  # noqa: E402


@pytest.fixture(scope="module")
def readme():
    with open(os.path.join(REPO, "README.md"), "r", encoding="utf-8") as fh:
        return fh.read()


def flat(text):
    """
    Prose with markdown emphasis and hard-wrapped lines collapsed to one line.

    A sentence in the README may wrap anywhere, so searching the raw text for a
    phrase makes the test fail on reflow rather than on meaning.
    """
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", ""))


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
def live_score():
    """A real /score response to compare the documented one against."""
    with mock_aws():
        _ensure_infra()
        import dynamo_client
        creds = dynamo_client.create_tenant(email="docs@safeship.test")
        import main
        main.app.config.update(TESTING=True)
        client = main.app.test_client()
        r = client.post("/score", json={
            "tenant_id": creds["tenant_id"], "api_key": creds["api_key"],
            "diff_size": 800, "files_changed": 40,
            "recent_failure_rate": 0.4, "is_hotfix": 1, "triggered_by": "docs",
        })
        assert r.status_code == 200, r.data
        yield r.get_json()


# ── the documented response must be the real one ─────────────────────────────

def _documented_block(readme):
    m = re.search(r"```json\n(\{.*?\n\})\n```", readme, re.S)
    assert m, "the README no longer shows an example /score response"
    return json.loads(m.group(1))


def test_the_documented_score_response_keys_all_exist(readme, live_score):
    documented = _documented_block(readme)
    missing = set(documented) - set(live_score)
    assert not missing, (
        f"the README documents /score fields the API does not return: {sorted(missing)}"
    )


def test_the_documented_reason_keys_all_exist(readme, live_score):
    documented = _documented_block(readme)["top_reasons"][0]
    actual = live_score["top_reasons"][0]
    missing = set(documented) - set(actual)
    assert not missing, (
        f"the README documents top_reasons fields that do not exist: {sorted(missing)}"
    )


def test_the_readme_documents_the_provenance_fields(readme, live_score):
    """
    These are the fields that keep an estimate from reading as a measurement, so
    an integrator has to know they are there.
    """
    for field in ("imputed", "feature_sources"):
        assert field in live_score
        assert field in readme, f"{field} is not documented"
    assert "imputed" in live_score["top_reasons"][0]
    assert "source" in live_score["top_reasons"][0]


def test_the_documented_feature_table_matches_the_contract(readme):
    section = readme.split("### Input Features")[1].split("##")[0]
    listed = re.findall(r"\|\s*\d+\s*\|\s*`(\w+)`", section)
    assert listed == list(FEATURES), (
        "the README's feature table has drifted from ml/features.py\n"
        f"  README: {listed}\n  contract: {list(FEATURES)}"
    )


# ── snippets must not hardcode features ──────────────────────────────────────

def _code_blocks(text):
    return re.findall(r"```(?:yaml|groovy|bash|json)?\n(.*?)```", text, re.S)


def test_no_readme_snippet_hardcodes_a_feature_value(readme):
    """
    The README used to end its integration section with a curl example carrying
    "diff_size":120,"files_changed":4 — instructions that produce a score which
    does not depend on the build.

    The documented /score *response* legitimately contains feature names with
    values, so JSON blocks are exempt; the pastable pipeline snippets are not.
    """
    for block in _code_blocks(readme):
        if '"score"' in block or '"verdict"' in block:
            continue                      # the example response, not an instruction
        for feature in FEATURES:
            assert f'"{feature}":' not in block, (
                f"a README snippet hardcodes {feature}:\n{block[:200]}"
            )


def test_the_readme_tells_people_to_close_the_learning_loop(readme):
    # Without labels the model never improves, so this cannot be a footnote.
    assert "mode:  log" in readme or "mode: log" in readme
    assert "safeship_ci log" in readme


def test_every_platform_has_an_integration_section(readme):
    import integrations

    for platform in integrations.ORDER:
        name = integrations.get(platform).NAME
        assert f"### {name}" in readme, f"{name} has no README section"


# ── references must resolve ──────────────────────────────────────────────────

def test_referenced_paths_exist(readme):
    """A README pointing at a deleted file is how safeship_gate.py survived."""
    referenced = set(re.findall(r"\[`([^`\]]+)`\]\(([^)]+)\)", readme))
    missing = []
    for _label, target in referenced:
        if target.startswith(("http://", "https://", "#")):
            continue
        if not os.path.exists(os.path.join(REPO, target)):
            missing.append(target)
    assert not missing, f"the README links to paths that do not exist: {missing}"


def test_the_readme_does_not_reference_the_deleted_gate_scripts(readme):
    for gone in ("safeship_gate.py", "safeship_log.py"):
        assert gone not in readme, (
            f"{gone} was deleted — it hardcoded features — but the README still "
            "tells people to run it"
        )


def test_the_composite_action_reference_matches_the_integration_module(readme):
    from integrations import github

    assert github.ACTION_REF in readme, (
        f"the README does not use the action reference the wizard hands out "
        f"({github.ACTION_REF})"
    )
    # And the action it points at has to be in this repo.
    path = github.ACTION_REF.split("@")[0].split("/", 2)[2]
    assert os.path.exists(os.path.join(REPO, path, "action.yml")), (
        f"{github.ACTION_REF} does not resolve to an action.yml"
    )


# ── the model claims must carry their caveat ──────────────────────────────────

def test_quoted_model_metrics_are_not_presented_as_predictive_accuracy(readme):
    """
    The README quoted "Precision ~85%, AUC-ROC ~0.93" with no caveat. Those are
    measured against synthetic labels generated by a hand-written rule, so they
    describe how well the forest reproduced that rule — not how well SafeShip
    predicts deployment failures. Quoting them bare is the most misleading thing
    the README could do.
    """
    section = readme.split("## 17.")[1].split("## 18.")[0]
    assert "0.935" in section or "0.93" in section, "the metrics were removed entirely"
    assert "synthetic" in section, "the metrics are quoted with no mention of synthetic labels"
    assert "not" in section.lower()


def test_the_synthetic_data_section_says_the_base_model_encodes_a_rule(readme):
    section = readme.split("## 8.")[1].split("## 9.")[0]
    assert "rule" in section.lower()
    # The concrete comparison that makes the point unarguable.
    assert "0.968" in section, "the rule-vs-forest AUC comparison is missing"


def test_the_selective_labels_limitation_is_stated(readme):
    # A gate cannot learn from the deploys it blocked. Not stating this invites
    # trusting the model more than the data supports.
    assert "cannot learn from the deploys it blocks" in flat(readme)
    assert "selective-labels" in flat(readme)


def test_the_retrain_floors_documented_match_the_code(readme):
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "retrain_handler_for_docs",
        os.path.join(REPO, "lambda", "retrain", "handler.py"))
    handler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(handler)

    section = readme.split("## 17.")[1].split("## 18.")[0]
    assert str(handler.MIN_BUILDS) in section, (
        f"the README does not state the real minimum ({handler.MIN_BUILDS})"
    )
    assert str(handler.MIN_TEST_ROWS) in section
    # The precision bar is a multiple of the tenant's own failure rate, so the
    # README has to describe it that way rather than quoting a fixed number.
    assert f"{handler.MIN_PRECISION_LIFT}×" in section or \
           f"{handler.MIN_PRECISION_LIFT}x" in section, \
        "the README still quotes a fixed precision threshold"
    assert str(handler.MIN_AUC) in section


def test_https_is_named_as_the_top_priority(readme):
    # Every build currently sends its API key in cleartext. A roadmap that buries
    # that under feature work is not being honest about the state of the product.
    section = readme.split("## 18.")[1].split("## 19.")[0]
    assert "HTTPS" in section
    assert "cleartext" in section
