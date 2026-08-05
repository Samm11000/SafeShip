"""
Tests for safeship_ci — the client-side feature extractor.

WHAT THESE ARE GUARDING
    Every integration SafeShip shipped before this package sent hardcoded
    constants: diff_size=120, test_pass_rate=1.0, recent_failure_rate=0.0. Since
    those two features carry 52.8% of the model's weight, the verdict barely
    depended on the build. The rule that replaced it is:

        measured, or None. Never a plausible-looking constant.

    So most of these tests assert an *absence*: that an unmeasurable feature comes
    back None rather than optimistic, and that a broken API degrades to None
    rather than raising into the customer's pipeline.

No network is touched. urllib.request.urlopen is replaced by a router over
recorded fixtures in tests/fixtures/ci/, whose shapes come from each provider's
current API docs.
"""
from __future__ import annotations

import io
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pytest

from safeship_ci import gitinfo, http, junit
from safeship_ci.collect import (apply_overrides, env_overrides)
from safeship_ci.collect import collect as collect_features
from safeship_ci.adapters import (BY_NAME, REGISTRY, BitbucketPipelines,
                                  Generic, GitHubActions, Jenkins, detect)
from safeship_ci.cli import EXIT_BLOCKED, EXIT_OK, EXIT_USAGE, main
from safeship_ci.contract import FEATURES

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "ci")


def fixture(name):
    with open(os.path.join(FIXTURES, name), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ── the expectations every adapter must meet ─────────────────────────────────
# All three fixtures encode the SAME underlying build history, so all three
# adapters must derive the same numbers from it. That is the real contract: a
# score should not depend on which CI you happen to use.

EXPECTED_FAILURE_RATE = 0.4      # 2 failures out of 5 decided builds
EXPECTED_MEDIAN_SECS = 270.0     # median of [60, 240, 240, 300, 360, 1800]
CURRENT_DURATION = 600.0         # faked "this build has run for 10 minutes"
LAST_SUCCESS_ISO = "2026-08-05T10:05:00Z"


def _iso(epoch, frac=False):
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00" if frac else "%Y-%m-%dT%H:%M:%SZ")


def expected_days_since_deploy():
    """Computed independently of the adapter's own parsing."""
    last = datetime(2026, 8, 5, 10, 5, 0, tzinfo=timezone.utc).timestamp()
    return (time.time() - last) / 86400.0


# ── the fake transport ───────────────────────────────────────────────────────

class _Resp(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


class Router:
    """
    Replaces urllib.request.urlopen. Routes are (url_substring, handler) pairs;
    a handler is a payload, a callable(url) -> payload, or an Exception to raise.
    """

    def __init__(self, *routes):
        self.routes = list(routes)
        self.calls = []

    def __call__(self, req, *args, **kwargs):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        self.calls.append(url)
        for needle, handler in self.routes:
            if needle in url:
                if isinstance(handler, BaseException):
                    raise handler
                payload = handler(url) if callable(handler) else handler
                return _Resp(json.dumps(payload).encode("utf-8"))
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, io.BytesIO(b"{}"))

    def install(self, monkeypatch):
        monkeypatch.setattr(urllib.request, "urlopen", self)
        return self


def http_error(code, body=b'{"message":"nope"}'):
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body))


# ── environments ─────────────────────────────────────────────────────────────

GITHUB_ENV = {
    "GITHUB_ACTIONS": "true",
    "GITHUB_REPOSITORY": "acme/app",
    "GITHUB_RUN_ID": "999",
    "GITHUB_RUN_NUMBER": "42",
    "GITHUB_TOKEN": "ghs_faketoken",
    "GITHUB_WORKFLOW_REF": "acme/app/.github/workflows/ci.yml@refs/heads/main",
    "GITHUB_WORKFLOW": "CI",
    "GITHUB_ACTOR": "octocat",
    "GITHUB_TRIGGERING_ACTOR": "hubot",
    "GITHUB_SHA": "a" * 40,
    "GITHUB_REF_NAME": "main",
}

JENKINS_ENV = {
    "JENKINS_URL": "https://ci.acme.test/",
    "JOB_NAME": "team/payments/main",
    "BUILD_NUMBER": "42",
    "BUILD_URL": "https://ci.acme.test/job/team/job/payments/job/main/42/",
    "JENKINS_USER": "svc",
    "JENKINS_TOKEN": "11aabb",
    "BUILD_USER_ID": "aparna",
    "GIT_COMMIT": "b" * 40,
    "BRANCH_NAME": "main",
}

BITBUCKET_ENV = {
    "BITBUCKET_BUILD_NUMBER": "42",
    "BITBUCKET_REPO_FULL_NAME": "acme/app",
    "BITBUCKET_WORKSPACE": "acme",
    "BITBUCKET_REPO_SLUG": "app",
    "BITBUCKET_PIPELINE_UUID": "{aaaaaaaa-0000-0000-0000-000000000042}",
    "BITBUCKET_ACCESS_TOKEN": "bbat_fake",
    "BITBUCKET_STEP_TRIGGERER_UUID": "{user-uuid-1}",
    "BITBUCKET_COMMIT": "c" * 40,
    "BITBUCKET_BRANCH": "main",
}


def github_adapter(monkeypatch, runs=None):
    env = dict(GITHUB_ENV)
    env["SAFESHIP_RUN_STARTED_AT"] = _iso(time.time() - CURRENT_DURATION)
    Router(("/actions/workflows/ci.yml/runs",
            runs if runs is not None else fixture("github_workflow_runs.json"))
           ).install(monkeypatch)
    return GitHubActions(env)


def jenkins_adapter(monkeypatch):
    Router(
        ("/42/api/json", lambda url: {"timestamp": int((time.time() - CURRENT_DURATION) * 1000)}),
        ("testReport/api/json", fixture("jenkins_testreport.json")),
        ("/api/json", fixture("jenkins_builds.json")),
    ).install(monkeypatch)
    return Jenkins(dict(JENKINS_ENV))


def bitbucket_adapter(monkeypatch):
    Router(
        ("/pipelines/%7B" , lambda url: {"created_on": _iso(time.time() - CURRENT_DURATION, frac=True)}),
        ("/pipelines/{", lambda url: {"created_on": _iso(time.time() - CURRENT_DURATION, frac=True)}),
        ("/pipelines?", fixture("bitbucket_pipelines.json")),
    ).install(monkeypatch)
    return BitbucketPipelines(dict(BITBUCKET_ENV))


ADAPTER_BUILDERS = {
    "github-actions": github_adapter,
    "jenkins": jenkins_adapter,
    "bitbucket": bitbucket_adapter,
}


# ═══ contract ════════════════════════════════════════════════════════════════

def test_client_contract_matches_the_server_contract():
    """
    safeship_ci carries its own copy of FEATURES because it ships standalone into
    customers' pipelines and cannot import server code. If the two ever disagree,
    the client silently stops sending a feature the model needs.
    """
    from ml.features import FEATURES as SERVER_FEATURES

    assert set(FEATURES) == set(SERVER_FEATURES), (
        "safeship_ci/contract.py has drifted from ml/features.py — "
        f"client-only={set(FEATURES) - set(SERVER_FEATURES)}, "
        f"server-only={set(SERVER_FEATURES) - set(FEATURES)}"
    )


def test_registry_order_puts_generic_last():
    # Generic.detect() always returns True, so anything after it is unreachable.
    assert REGISTRY[-1] is Generic
    assert [c.name for c in REGISTRY[:-1]] == ["github-actions", "jenkins", "bitbucket"]


@pytest.mark.parametrize("env,expected", [
    ({"GITHUB_ACTIONS": "true"}, "github-actions"),
    ({"JENKINS_URL": "https://ci"}, "jenkins"),
    ({"BITBUCKET_BUILD_NUMBER": "7"}, "bitbucket"),
    ({}, "generic"),
])
def test_detection_from_environment(env, expected):
    assert detect(env=env).name == expected


def test_forced_platform_overrides_detection():
    # This is what makes the adapters testable offline, and lets a user correct
    # a bad guess without editing their pipeline.
    assert detect(env={"GITHUB_ACTIONS": "true"}, force="jenkins").name == "jenkins"
    assert detect(env={}, force="bitbucket").name == "bitbucket"


def test_unknown_forced_platform_is_a_usage_error():
    with pytest.raises(ValueError) as exc:
        detect(env={}, force="teamcity")
    assert "teamcity" in str(exc.value)
    assert "github-actions" in str(exc.value)


# ═══ adapters: the same history must yield the same features ═════════════════

@pytest.mark.parametrize("platform", sorted(ADAPTER_BUILDERS))
def test_every_adapter_derives_the_same_features_from_equivalent_history(
        platform, monkeypatch):
    """
    The three fixtures encode identical build history in three different response
    shapes. A score must not depend on which CI the customer happens to run.
    """
    ad = ADAPTER_BUILDERS[platform](monkeypatch)
    feats = ad.features_from_history()

    assert feats["recent_failure_rate"] == EXPECTED_FAILURE_RATE
    assert feats["days_since_deploy"] == pytest.approx(
        expected_days_since_deploy(), abs=0.01)
    # (600 - 270) / 270
    assert feats["build_time_delta"] == pytest.approx(
        (CURRENT_DURATION - EXPECTED_MEDIAN_SECS) / EXPECTED_MEDIAN_SECS, abs=0.05)


@pytest.mark.parametrize("platform", sorted(ADAPTER_BUILDERS))
def test_the_in_flight_build_is_never_part_of_its_own_history(platform, monkeypatch):
    ad = ADAPTER_BUILDERS[platform](monkeypatch)
    records = ad.history()
    assert len(records) == 6, "the currently-running build leaked into history"
    # A running build has no verdict; if it were included, `decided` would be 6.
    assert sum(1 for r in records if r.succeeded is not None) == 5


@pytest.mark.parametrize("platform", sorted(ADAPTER_BUILDERS))
def test_a_cancelled_build_is_not_counted_as_a_failure(platform, monkeypatch):
    """
    Someone hitting cancel is not a broken deploy. Counting it as one inflates
    recent_failure_rate — the single heaviest feature in the model.
    """
    ad = ADAPTER_BUILDERS[platform](monkeypatch)
    records = ad.history()
    assert sum(1 for r in records if r.succeeded is None) == 1
    # 2/5 rather than 3/6 proves the undecided build was excluded, not failed.
    assert ad.features_from_history()["recent_failure_rate"] == 0.4


@pytest.mark.parametrize("platform", sorted(ADAPTER_BUILDERS))
def test_adapters_report_an_actor_for_server_side_deployer_exp(platform, monkeypatch):
    # The server keys its build counter on this. No actor means no deployer_exp.
    ad = ADAPTER_BUILDERS[platform](monkeypatch)
    assert ad.actor()


def test_github_prefers_the_triggering_actor():
    # On a re-run, GITHUB_ACTOR is the original author but the person taking the
    # risk right now is whoever pressed the button.
    assert GitHubActions(dict(GITHUB_ENV)).actor() == "hubot"
    env = dict(GITHUB_ENV)
    del env["GITHUB_TRIGGERING_ACTOR"]
    assert GitHubActions(env).actor() == "octocat"


def test_github_pull_request_uses_the_source_branch_not_the_merge_ref():
    # On a pull_request event GITHUB_REF_NAME is "123/merge", which no hotfix
    # pattern would ever match.
    env = dict(GITHUB_ENV, GITHUB_REF_NAME="123/merge", GITHUB_HEAD_REF="hotfix/pay")
    assert GitHubActions(env).branch() == "hotfix/pay"
    assert gitinfo.is_hotfix("hotfix/pay") == 1
    assert gitinfo.is_hotfix("123/merge") == 0


def test_jenkins_folder_job_names_become_nested_url_paths():
    from safeship_ci.adapters.jenkins import _job_path

    # "team/payments/main" is a folder path, not one job called that.
    assert _job_path("team/payments/main") == "job/team/job/payments/job/main"
    assert _job_path("my job") == "job/my%20job"


def test_jenkins_reads_test_results_from_its_native_api(monkeypatch):
    ad = jenkins_adapter(monkeypatch)
    # 47 passed, 3 failed, 5 skipped -> 47/50. Skips are excluded, not failed.
    assert ad.test_pass_rate() == 0.94


@pytest.mark.parametrize("platform", ["github-actions", "bitbucket"])
def test_platforms_without_a_test_api_return_none_not_one(platform, monkeypatch):
    """
    Only Jenkins has a native test-results API. The others must say "I don't
    know" so JUnit XML (or the server's imputation) can answer instead. Returning
    1.0 here is exactly the bug this package replaced.
    """
    ad = ADAPTER_BUILDERS[platform](monkeypatch)
    assert ad.test_pass_rate() is None


def test_bitbucket_sorts_by_created_on(monkeypatch):
    # sort=-completed_on returns invalid-sort-attribute from the Bitbucket API.
    router = Router(("/pipelines", fixture("bitbucket_pipelines.json"))).install(monkeypatch)
    BitbucketPipelines(dict(BITBUCKET_ENV)).history()
    assert any("sort=-created_on" in u for u in router.calls)
    assert not any("completed_on" in u for u in router.calls)


def test_github_scopes_history_to_this_workflow_file(monkeypatch):
    """
    Unscoped, history mixes in lint/docs workflows and recent_failure_rate becomes
    noise about unrelated jobs.
    """
    router = Router(("/actions/", fixture("github_workflow_runs.json"))).install(monkeypatch)
    GitHubActions(dict(GITHUB_ENV)).history()
    assert any("/actions/workflows/ci.yml/runs" in u for u in router.calls)


def test_github_says_so_when_history_is_not_scoped(monkeypatch):
    env = dict(GITHUB_ENV)
    del env["GITHUB_WORKFLOW_REF"]
    del env["GITHUB_WORKFLOW"]
    Router(("/actions/runs", fixture("github_workflow_runs.json"))).install(monkeypatch)
    ad = GitHubActions(env)
    ad.history()
    assert any("not scoped" in n for n in ad.notes)


# ═══ degradation: a broken API must never raise ══════════════════════════════

@pytest.mark.parametrize("failure", [
    http_error(500),
    http_error(502),
    urllib.error.URLError("connection refused"),
    TimeoutError("timed out"),
])
@pytest.mark.parametrize("platform", sorted(ADAPTER_BUILDERS))
def test_api_failure_degrades_to_unknown_instead_of_raising(platform, failure, monkeypatch):
    """
    THE FAIL-OPEN CONTRACT, at the adapter level. A build is waiting; an outage in
    the CI's own API must cost us the feature, not the deploy.
    """
    builders = {
        "github-actions": (GitHubActions, GITHUB_ENV, "/actions/"),
        "jenkins": (Jenkins, JENKINS_ENV, "/api/json"),
        "bitbucket": (BitbucketPipelines, BITBUCKET_ENV, "/pipelines"),
    }
    cls, env, needle = builders[platform]
    Router((needle, failure)).install(monkeypatch)
    ad = cls(dict(env))

    feats = ad.features_from_history()          # must not raise
    assert feats == {"recent_failure_rate": None,
                     "days_since_deploy": None,
                     "build_time_delta": None}
    assert ad.notes, "a degraded feature must explain itself"


def test_github_403_names_the_exact_permission_to_add(monkeypatch):
    """
    Repos created after Feb 2023 give GITHUB_TOKEN contents+metadata only, so the
    runs API 403s. This is the most common misconfiguration by far — the message
    has to be actionable, not "history unavailable".
    """
    Router(("/actions/", http_error(403))).install(monkeypatch)
    ad = GitHubActions(dict(GITHUB_ENV))
    assert ad.history() is None
    assert any("actions: read" in n for n in ad.notes)


def test_missing_github_token_tells_you_how_to_pass_one(monkeypatch):
    env = dict(GITHUB_ENV)
    del env["GITHUB_TOKEN"]
    ad = GitHubActions(env)
    assert ad.history() is None
    assert any("github-token" in n or "GITHUB_TOKEN" in n for n in ad.notes)


def test_bitbucket_explains_that_it_needs_a_repository_access_token():
    env = dict(BITBUCKET_ENV)
    del env["BITBUCKET_ACCESS_TOKEN"]
    ad = BitbucketPipelines(env)
    assert ad.history() is None
    assert any("Repository Access Token" in n for n in ad.notes)


def test_jenkins_403_asks_for_an_api_token(monkeypatch):
    Router(("/api/json", http_error(403))).install(monkeypatch)
    env = dict(JENKINS_ENV)
    del env["JENKINS_USER"]
    ad = Jenkins(env)
    assert ad.history() is None
    assert any("JENKINS_TOKEN" in n for n in ad.notes)


def test_malformed_json_is_not_a_crash(monkeypatch):
    def bad(req, *a, **k):
        return _Resp(b"<html>504 Gateway Time-out</html>")

    monkeypatch.setattr(urllib.request, "urlopen", bad)
    ad = GitHubActions(dict(GITHUB_ENV))
    assert ad.features_from_history()["recent_failure_rate"] is None


def test_empty_history_is_unknown_not_zero_failures(monkeypatch):
    """
    A brand-new repository has no history. recent_failure_rate=0.0 would read as
    "this project has never failed" — a maximally reassuring claim from no data.
    """
    ad = github_adapter(monkeypatch, runs={"total_count": 0, "workflow_runs": []})
    feats = ad.features_from_history()
    assert feats["recent_failure_rate"] is None
    assert any("no build history" in n for n in ad.notes)


def test_history_of_only_cancelled_builds_yields_no_failure_rate(monkeypatch):
    runs = {"workflow_runs": [
        {"id": 1, "conclusion": "cancelled",
         "created_at": "2026-08-01T10:00:00Z", "updated_at": "2026-08-01T10:01:00Z"},
        {"id": 2, "conclusion": "skipped",
         "created_at": "2026-08-01T09:00:00Z", "updated_at": "2026-08-01T09:01:00Z"},
    ]}
    ad = github_adapter(monkeypatch, runs=runs)
    assert ad.features_from_history()["recent_failure_rate"] is None


# ═══ timestamp parsing ═══════════════════════════════════════════════════════

def test_iso_timestamps_parse_independently_of_the_local_timezone(monkeypatch):
    """
    time.mktime would read these fields as local time and ignore the offset, so a
    runner in a DST-observing timezone would be an hour out for half the year —
    enough to move days_since_deploy. Both adapters use calendar.timegm.
    """
    from safeship_ci.adapters.bitbucket import _epoch as bb_epoch
    from safeship_ci.adapters.github import _epoch as gh_epoch

    expected = datetime(2026, 8, 5, 10, 5, 0, tzinfo=timezone.utc).timestamp()
    assert gh_epoch("2026-08-05T10:05:00Z") == expected
    assert bb_epoch("2026-08-05T10:05:00.000000+00:00") == expected
    # A non-UTC offset must be honoured, not dropped.
    assert gh_epoch("2026-08-05T15:35:00+0530") == expected
    assert bb_epoch("2026-08-05T15:35:00.000000+05:30") == expected
    assert gh_epoch(None) is None
    assert gh_epoch("not a date") is None


# ═══ git ═════════════════════════════════════════════════════════════════════

def _git(cwd, *args):
    subprocess.run(("git",) + args, cwd=cwd, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real two-commit git repository."""
    # gitinfo.branch_name() consults CI env vars first; a stray GITHUB_REF_NAME in
    # the test runner's own environment would leak into these assertions.
    for var in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME", "BRANCH_NAME",
                "BITBUCKET_BRANCH", "CI_COMMIT_REF_NAME", "GITHUB_ACTIONS"):
        monkeypatch.delenv(var, raising=False)

    path = tmp_path / "repo"
    path.mkdir()
    p = str(path)
    _git(p, "init", "-q", "-b", "main")
    _git(p, "config", "user.email", "t@example.com")
    _git(p, "config", "user.name", "Test")
    _git(p, "config", "commit.gpgsign", "false")

    (path / "a.txt").write_text("one\ntwo\nthree\n")
    _git(p, "add", "-A")
    _git(p, "commit", "-qm", "first")
    return path


def test_first_commit_diffs_against_the_empty_tree(repo):
    # A repository's first commit really is "everything is new" — 3 added lines.
    size, files, note = gitinfo.diff_stats(cwd=str(repo))
    assert (size, files) == (3, 1)
    assert "empty-tree" in note


def test_diff_size_is_churn_not_net_lines(repo):
    """
    A change that deletes 400 lines and adds 10 is not a 390-line-smaller,
    lower-risk change. diff_size counts insertions + deletions.
    """
    # a.txt: "one/two/three" -> "ONE/two"  = +1 (ONE) -2 (one, three)
    # b.txt: new file                      = +1
    (repo / "a.txt").write_text("ONE\ntwo\n")
    (repo / "b.txt").write_text("new\n")
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "second")

    size, files, note = gitinfo.diff_stats(cwd=str(repo))
    assert files == 2
    # Net lines would be 0 (2 in, 2 out). Churn is 4, and churn is what carries risk.
    assert size == 4, "expected 2 insertions + 2 deletions as churn, not the net"
    assert "head~1" in note


def test_an_explicit_base_commit_wins_over_head_parent(repo):
    first = gitinfo.head_sha(cwd=str(repo))
    for n in range(2, 5):
        (repo / "a.txt").write_text("line\n" * n)
        _git(str(repo), "add", "-A")
        _git(str(repo), "commit", "-qm", f"c{n}")

    # Against HEAD~1 this is a small change; against the PR base it is the whole
    # branch. Using the wrong base is how a big PR scores as a one-line tweak.
    small, _, _ = gitinfo.diff_stats(cwd=str(repo))
    whole, _, note = gitinfo.diff_stats(base=first, cwd=str(repo))
    assert whole > small
    assert note == "vs platform"


def test_an_all_zero_base_is_ignored(repo):
    # A push event to a new branch reports before=0000000000...; it is not a rev.
    _, _, note = gitinfo.diff_stats(base="0" * 40, cwd=str(repo))
    assert "head~1" in note or "empty-tree" in note


def test_a_nonexistent_base_falls_back_rather_than_failing(repo):
    size, files, _ = gitinfo.diff_stats(base="deadbeef" * 5, cwd=str(repo))
    assert size is not None and files is not None


def test_shallow_clone_reports_unknown_and_names_the_fix(tmp_path, repo, monkeypatch):
    """
    THE GOTCHA THIS PACKAGE EXISTS FOR: actions/checkout defaults to
    fetch-depth: 1, so `git diff HEAD~1` fails. The old integration papered over
    that with diff_size=120. We report None and print the one-line fix.
    """
    for n in range(2, 6):
        (repo / "a.txt").write_text("line\n" * n)
        _git(str(repo), "add", "-A")
        _git(str(repo), "commit", "-qm", f"c{n}")

    shallow = tmp_path / "shallow"
    _git(str(tmp_path), "clone", "-q", "--depth", "1",
         "file://" + str(repo), str(shallow))
    assert gitinfo.is_shallow(cwd=str(shallow)) is True

    size, files, note = gitinfo.diff_stats(cwd=str(shallow))
    assert size is None and files is None, "a shallow clone must not fabricate a diff"
    assert "shallow" in note

    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert "fetch-depth: 2" in gitinfo.shallow_hint()
    monkeypatch.delenv("GITHUB_ACTIONS")
    monkeypatch.setenv("BITBUCKET_BUILD_NUMBER", "1")
    assert "clone: depth" in gitinfo.shallow_hint()


def test_outside_a_repository_nothing_is_invented(tmp_path):
    size, files, note = gitinfo.diff_stats(cwd=str(tmp_path / "nope"))
    assert (size, files) == (None, None)
    assert "not a git repository" in note or "no parent" in note


def test_an_empty_diff_is_zero_not_unknown(repo):
    # Genuinely zero churn is a measurement. It must not become None.
    size, files, _ = gitinfo.diff_stats(base=gitinfo.head_sha(cwd=str(repo)),
                                        cwd=str(repo))
    assert (size, files) == (0, 0)


def test_binary_files_count_as_changed_files_without_line_counts(repo):
    (repo / "logo.png").write_bytes(bytes(range(256)) * 4)
    _git(str(repo), "add", "-A")
    _git(str(repo), "commit", "-qm", "add binary")

    size, files, _ = gitinfo.diff_stats(cwd=str(repo))
    # git numstat reports "-\t-\t" for a binary: one file, no measurable lines.
    assert files == 1
    assert size == 0


@pytest.mark.parametrize("branch,expected", [
    ("hotfix/payments", 1),
    ("HOTFIX-123", 1),
    ("revert-bad-deploy", 1),
    ("emergency", 1),
    ("release/2.1-patch", 1),
    ("feature/new-dashboard", 0),
    ("main", 0),
    ("", 0),
])
def test_hotfix_detection(branch, expected):
    assert gitinfo.is_hotfix(branch) == expected


# ═══ JUnit XML ═══════════════════════════════════════════════════════════════

def write_xml(tmp_path, name, body):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body)
    return path


def test_pass_rate_from_a_bare_testsuite(tmp_path):
    write_xml(tmp_path, "junit.xml",
              '<testsuite tests="10" failures="2" errors="0" skipped="0"/>')
    rate, note = junit.pass_rate(str(tmp_path))
    assert rate == 0.8
    assert "8/10" in note


def test_errors_count_against_the_pass_rate(tmp_path):
    # An erroring test did not pass, whatever the runner calls it.
    write_xml(tmp_path, "junit.xml",
              '<testsuite tests="10" failures="1" errors="1" skipped="0"/>')
    assert junit.pass_rate(str(tmp_path))[0] == 0.8


def test_skipped_tests_are_excluded_from_the_denominator(tmp_path):
    """
    90 of 100 skipped and 10 passing is a 100% pass rate, not 10%. Teams skip
    platform-specific tests; punishing them for it would make the feature noise.
    """
    write_xml(tmp_path, "junit.xml",
              '<testsuite tests="100" failures="0" errors="0" skipped="90"/>')
    rate, note = junit.pass_rate(str(tmp_path))
    assert rate == 1.0
    assert "90 skipped (excluded)" in note


def test_all_skipped_is_unknown_rather_than_perfect(tmp_path):
    write_xml(tmp_path, "junit.xml",
              '<testsuite tests="20" failures="0" errors="0" skipped="20"/>')
    rate, note = junit.pass_rate(str(tmp_path))
    assert rate is None, "no tests actually ran — that is not a 100% pass rate"
    assert "skipped" in note


def test_a_testsuites_wrapper_is_not_double_counted(tmp_path):
    # The wrapper carries totals AND nested suites; summing both would double.
    write_xml(tmp_path, "junit.xml", """
        <testsuites tests="10" failures="2" errors="0" skipped="0">
          <testsuite tests="6" failures="1" errors="0" skipped="0"/>
          <testsuite tests="4" failures="1" errors="0" skipped="0"/>
        </testsuites>""".strip())
    assert junit.pass_rate(str(tmp_path))[0] == 0.8


def test_a_testsuites_wrapper_without_totals_is_summed(tmp_path):
    # jest-junit and go-junit-report both emit a bare wrapper.
    write_xml(tmp_path, "junit.xml", """
        <testsuites>
          <testsuite tests="6" failures="1" errors="0" skipped="0"/>
          <testsuite tests="4" failures="1" errors="0" skipped="0"/>
        </testsuites>""".strip())
    assert junit.pass_rate(str(tmp_path))[0] == 0.8


def test_multiple_reports_are_aggregated(tmp_path):
    write_xml(tmp_path, "reports/junit-unit.xml",
              '<testsuite tests="60" failures="0" errors="0" skipped="0"/>')
    write_xml(tmp_path, "reports/junit-integration.xml",
              '<testsuite tests="40" failures="10" errors="0" skipped="0"/>')
    rate, note = junit.pass_rate(str(tmp_path))
    assert rate == 0.9
    assert "2 report(s)" in note


def test_maven_and_gradle_layouts_are_discovered(tmp_path):
    write_xml(tmp_path, "target/surefire-reports/TEST-com.acme.FooTest.xml",
              '<testsuite tests="4" failures="1" errors="0" skipped="0"/>')
    assert junit.pass_rate(str(tmp_path))[0] == 0.75

    other = tmp_path / "gradle"
    write_xml(other, "build/test-results/test/TEST-Bar.xml",
              '<testsuite tests="4" failures="0" errors="0" skipped="0"/>')
    assert junit.pass_rate(str(other))[0] == 1.0


def test_vendored_directories_are_not_searched(tmp_path):
    # A dependency's own fixtures would otherwise dominate the pass rate.
    write_xml(tmp_path, "node_modules/pkg/junit.xml",
              '<testsuite tests="100" failures="100" errors="0" skipped="0"/>')
    write_xml(tmp_path, "junit.xml",
              '<testsuite tests="10" failures="0" errors="0" skipped="0"/>')
    assert junit.pass_rate(str(tmp_path))[0] == 1.0


def test_no_reports_is_unknown_not_one(tmp_path):
    """The regression this whole package is about: absent tests ≠ passing tests."""
    rate, note = junit.pass_rate(str(tmp_path))
    assert rate is None
    assert "no JUnit XML" in note


def test_unparseable_xml_is_skipped_not_fatal(tmp_path):
    write_xml(tmp_path, "junit.xml", "<testsuite tests=oops")
    write_xml(tmp_path, "junit-good.xml",
              '<testsuite tests="4" failures="0" errors="0" skipped="0"/>')
    assert junit.pass_rate(str(tmp_path))[0] == 1.0

    write_xml(tmp_path, "junit-good.xml", "also broken <<<")
    rate, note = junit.pass_rate(str(tmp_path))
    assert rate is None
    assert "no test counts" in note


def test_a_report_with_more_failures_than_tests_cannot_go_negative(tmp_path):
    write_xml(tmp_path, "junit.xml",
              '<testsuite tests="5" failures="9" errors="0" skipped="0"/>')
    rate, _ = junit.pass_rate(str(tmp_path))
    assert rate == 0.0


# ═══ collect(): the whole assembly ═══════════════════════════════════════════

def test_collect_always_returns_exactly_the_contract(tmp_path):
    out = collect_features(Generic({}), cwd=str(tmp_path))
    assert set(out.features) == set(FEATURES)


def test_unmeasurable_features_stay_none(tmp_path):
    """
    The core regression test. Outside a repo, with no CI API and no test report,
    almost nothing is knowable — and the honest answer is None for each of them.
    Before this package, this same situation produced test_pass_rate=1.0 and
    recent_failure_rate=0.0, i.e. a perfect build conjured from no data.
    """
    out = collect_features(Generic({}), cwd=str(tmp_path))

    assert out.features["test_pass_rate"] is None
    assert out.features["recent_failure_rate"] is None
    assert out.features["days_since_deploy"] is None
    assert out.features["build_time_delta"] is None
    # deployer_exp is derived server-side from build history; the client must not
    # send a value the server would have to decide whether to trust.
    assert out.features["deployer_exp"] is None


def test_clock_features_are_always_available(tmp_path):
    out = collect_features(Generic({}), cwd=str(tmp_path))
    assert out.features["hour_of_day"] in range(24)
    assert out.features["day_of_week"] in range(7)
    assert out.sources["hour_of_day"] == "clock"


def test_collect_records_provenance_for_every_measured_feature(repo):
    out = collect_features(Generic({}), cwd=str(repo))
    for feature in out.measured:
        assert out.sources[feature], f"{feature} has a value but no source"
    assert out.sources["diff_size"] == "git"


def test_collect_reports_measured_and_unknown_separately(tmp_path):
    out = collect_features(Generic({}), cwd=str(tmp_path))
    assert set(out.measured) & set(out.unknown) == set()
    assert set(out.measured) | set(out.unknown) == set(FEATURES)


def test_collect_finds_a_junit_report_in_the_working_tree(repo):
    (repo / "reports").mkdir()
    (repo / "reports" / "junit.xml").write_text(
        '<testsuite tests="8" failures="2" errors="0" skipped="0"/>')
    out = collect_features(Generic({}), cwd=str(repo))
    assert out.features["test_pass_rate"] == 0.75
    assert out.sources["test_pass_rate"] == "junit-xml"


def test_collect_survives_an_adapter_that_throws(tmp_path):
    """
    A third-party adapter bug must not break the customer's build. collect() is
    the last line of defence before the CLI's own catch-all.
    """
    class Exploding(Generic):
        name = "exploding"

        def history(self, limit=10):
            raise RuntimeError("boom")

        def base_commit(self):
            raise RuntimeError("boom")

        def test_pass_rate(self):
            raise RuntimeError("boom")

    out = collect_features(Exploding({}), cwd=str(tmp_path))
    assert set(out.features) == set(FEATURES)
    assert out.features["hour_of_day"] is not None    # the clock still worked


def test_collect_with_a_full_github_environment(monkeypatch, repo):
    ad = github_adapter(monkeypatch)
    (repo / "reports").mkdir()
    (repo / "reports" / "junit.xml").write_text(
        '<testsuite tests="50" failures="1" errors="0" skipped="0"/>')

    out = collect_features(ad, cwd=str(repo))

    # 9 of 10: everything except deployer_exp, which is the server's to decide.
    assert sorted(out.unknown) == ["deployer_exp"]
    assert out.features["recent_failure_rate"] == 0.4
    assert out.features["test_pass_rate"] == 0.98
    assert out.features["diff_size"] == 3
    assert out.meta["triggered_by"] == "hubot"
    assert out.meta["ci_platform"] == "github-actions"
    assert out.meta["job_name"] == "acme/app/CI"


def test_overrides_replace_measurements_and_are_labelled(tmp_path):
    out = collect_features(Generic({}), cwd=str(tmp_path))
    out = apply_overrides(out, {"test_pass_rate": 0.5, "diff_size": 42})
    assert out.features["test_pass_rate"] == 0.5
    assert out.sources["test_pass_rate"] == "override"
    assert out.sources["diff_size"] == "override"


def test_env_overrides_are_typed(monkeypatch):
    monkeypatch.setenv("SAFESHIP_DIFF_SIZE", "250")
    monkeypatch.setenv("SAFESHIP_TEST_PASS_RATE", "0.87")
    monkeypatch.setenv("SAFESHIP_FILES_CHANGED", "not-a-number")
    monkeypatch.setenv("SAFESHIP_IS_HOTFIX", "")

    got = env_overrides()
    assert got["diff_size"] == 250 and isinstance(got["diff_size"], int)
    assert got["test_pass_rate"] == 0.87
    assert "files_changed" not in got, "garbage must be ignored, not crash"
    assert "is_hotfix" not in got, "an empty env var is not a value"


def test_an_override_of_none_does_not_erase_a_measurement(repo):
    out = collect_features(Generic({}), cwd=str(repo))
    measured = out.features["diff_size"]
    out = apply_overrides(out, {"diff_size": None})
    assert out.features["diff_size"] == measured


# ═══ http: retry policy ══════════════════════════════════════════════════════

def test_a_4xx_is_not_retried(monkeypatch):
    # 401 means the API key is wrong. Repeating the request cannot fix that, and
    # hammering a customer's gateway on every build is its own problem.
    router = Router(("/score", http_error(401, b'{"error":"bad key"}'))).install(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(http.SafeShipError) as exc:
        http.post("http://ss.test", "/score", {"a": 1}, retries=3)
    assert len(router.calls) == 1
    assert "401" in str(exc.value)


def test_a_5xx_is_retried_then_gives_up(monkeypatch):
    router = Router(("/score", http_error(503))).install(monkeypatch)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(http.SafeShipError):
        http.post("http://ss.test", "/score", {"a": 1}, retries=2)
    assert len(router.calls) == 3, "1 attempt + 2 retries"


def test_a_transient_failure_recovers(monkeypatch):
    state = {"n": 0}

    def flaky(req, *a, **k):
        state["n"] += 1
        if state["n"] == 1:
            raise urllib.error.URLError("connection reset")
        return _Resp(b'{"score": 12, "verdict": "SAFE"}')

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    assert http.post("http://ss.test", "/score", {}, retries=2)["score"] == 12


def test_non_json_from_a_proxy_is_a_safeship_error(monkeypatch):
    def html(req, *a, **k):
        return _Resp(b"<html>502 Bad Gateway</html>")

    monkeypatch.setattr(urllib.request, "urlopen", html)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(http.SafeShipError) as exc:
        http.post("http://ss.test", "/score", {}, retries=0)
    assert "non-JSON" in str(exc.value)


def test_the_api_key_never_reaches_a_log_line():
    line = http.describe({"tenant_id": "t1", "api_key": "sk_live_secret", "diff_size": 4})
    assert "sk_live_secret" not in line
    assert "[REDACTED]" in line


def test_nulls_are_sent_as_json_null(monkeypatch):
    """
    The server distinguishes "not measured" from "measured as zero". That only
    works if the client actually transmits null instead of dropping the key.
    """
    sent = {}

    def capture(req, *a, **k):
        sent.update(json.loads(req.data.decode()))
        return _Resp(b'{"score": 50, "verdict": "SAFE"}')

    monkeypatch.setattr(urllib.request, "urlopen", capture)
    http.score("http://ss.test", "t1", "k1",
               {"diff_size": 10, "test_pass_rate": None})
    assert sent["test_pass_rate"] is None
    assert "test_pass_rate" in sent


# ═══ the CLI's exit codes — the contract that matters most ═══════════════════

BASE_ARGS = ["--url", "http://ss.test", "--tenant-id", "t1", "--api-key", "k1",
             "--retries", "0", "--platform", "generic", "--allow-insecure-url"]


def _run(monkeypatch, tmp_path, response, extra=()):
    """Run `safeship score` against a canned /score response."""
    monkeypatch.setattr(http, "score", lambda *a, **k: response)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    return main(["score", *BASE_ARGS, "--cwd", str(tmp_path),
                 "--build-id-file", str(tmp_path / "id.txt"), *extra])


def test_a_safe_verdict_exits_zero(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path,
                {"score": 12, "verdict": "SAFE", "build_id": "b1"}) == EXIT_OK


def test_a_blocked_verdict_stops_the_deploy(monkeypatch, tmp_path):
    # The one and only case where safeship_ci is allowed to fail a build.
    assert _run(monkeypatch, tmp_path,
                {"score": 88, "verdict": "BLOCKED", "build_id": "b1"}) == EXIT_BLOCKED


def test_fail_open_reports_blocked_but_lets_the_deploy_through(monkeypatch, tmp_path):
    assert _run(monkeypatch, tmp_path,
                {"score": 88, "verdict": "BLOCKED"}, ["--fail-open"]) == EXIT_OK


def test_an_unreachable_safeship_does_not_break_the_build(monkeypatch, tmp_path, capsys):
    """
    THE CENTRAL PROMISE. A risk gate that halts everyone's deploys during its own
    outage gets deleted in a week (PRODUCTION-PLAN.md §3.3).
    """
    def boom(*a, **k):
        raise http.SafeShipError("cannot reach http://ss.test: timed out")

    monkeypatch.setattr(http, "score", boom)
    assert main(["score", *BASE_ARGS, "--cwd", str(tmp_path)]) == EXIT_OK
    assert "proceeding without a gate" in capsys.readouterr().out


def test_an_unexpected_crash_in_our_own_code_still_exits_zero(monkeypatch, tmp_path, capsys):
    def bug(*a, **k):
        raise ZeroDivisionError("a bug in us")

    monkeypatch.setattr(http, "score", bug)
    assert main(["score", *BASE_ARGS, "--cwd", str(tmp_path)]) == EXIT_OK
    assert "crashed" in capsys.readouterr().out


def test_missing_credentials_are_a_usage_error_not_a_silent_pass(monkeypatch, capsys):
    # Distinct from fail-open: nothing was attempted, and the user must be told
    # rather than left believing the gate is running.
    for var in ("SAFESHIP_URL", "SAFESHIP_TENANT_ID", "SAFESHIP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    assert main(["score", "--platform", "generic"]) == EXIT_USAGE
    assert "missing credentials" in capsys.readouterr().err


def test_a_plain_http_url_warns_about_the_cleartext_api_key(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(http, "score", lambda *a, **k: {"score": 1, "verdict": "SAFE"})
    main(["score", "--url", "http://54.89.160.150", "--tenant-id", "t",
          "--api-key", "k", "--platform", "generic", "--cwd", str(tmp_path)])
    assert "cleartext" in capsys.readouterr().out


def test_imputed_features_are_surfaced_to_the_user(monkeypatch, tmp_path, capsys):
    # If the server had to guess half the input, the user needs to know the score
    # is weaker than it looks — otherwise they trust a number built on medians.
    _run(monkeypatch, tmp_path, {
        "score": 40, "verdict": "SAFE",
        "imputed": ["test_pass_rate", "recent_failure_rate"],
    })
    out = capsys.readouterr().out
    assert "estimated" in out
    assert "test_pass_rate" in out


def test_the_build_id_is_saved_for_the_later_log_step(monkeypatch, tmp_path):
    _run(monkeypatch, tmp_path, {"score": 1, "verdict": "SAFE", "build_id": "bld-7"})
    assert (tmp_path / "id.txt").read_text().strip() == "bld-7"


def test_log_reads_the_build_id_from_the_file(monkeypatch, tmp_path):
    (tmp_path / "id.txt").write_text("bld-9\n")
    seen = {}
    monkeypatch.setattr(http, "log_outcome",
                        lambda url, t, k, build_id, label, **kw:
                        seen.update(build_id=build_id, label=label) or {"ok": True})
    assert main(["log", "1", *BASE_ARGS,
                 "--build-id-file", str(tmp_path / "id.txt")]) == EXIT_OK
    assert seen == {"build_id": "bld-9", "label": 1}


def test_log_without_a_build_id_is_a_warning_not_a_failure(monkeypatch, tmp_path, capsys):
    assert main(["log", "0", *BASE_ARGS,
                 "--build-id-file", str(tmp_path / "missing.txt")]) == EXIT_OK
    assert "nothing to label" in capsys.readouterr().out


def test_a_failed_log_never_fails_the_pipeline(monkeypatch, tmp_path):
    (tmp_path / "id.txt").write_text("bld-9")

    def boom(*a, **k):
        raise http.SafeShipError("down")

    monkeypatch.setattr(http, "log_outcome", boom)
    assert main(["log", "1", *BASE_ARGS,
                 "--build-id-file", str(tmp_path / "id.txt")]) == EXIT_OK


def test_github_actions_outputs_and_summary_are_written(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out.txt"))
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(tmp_path / "sum.md"))

    _run(monkeypatch, tmp_path, {
        "score": 67, "verdict": "REVIEW", "build_id": "b2",
        "top_reasons": [{"label": "Change size", "value_str": "1245 lines"}],
        "imputed": ["build_time_delta"],
    })

    outputs = (tmp_path / "out.txt").read_text()
    assert "score=67" in outputs and "verdict=REVIEW" in outputs
    assert "build-id=b2" in outputs

    summary = (tmp_path / "sum.md").read_text()
    assert "REVIEW" in summary and "67/100" in summary
    assert "1245 lines" in summary
    assert "build_time_delta" in summary


def test_warnings_use_the_actions_annotation_format(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    _run(monkeypatch, tmp_path, {"score": 1, "verdict": "SAFE",
                                 "imputed": ["test_pass_rate"]})
    assert "::warning::" in capsys.readouterr().out


def test_collect_needs_no_credentials_and_touches_no_network(monkeypatch, tmp_path, capsys):
    # `safeship collect` is the debugging entry point: it has to work before the
    # user has an API key, and must never hang on a network call.
    def no_network(*a, **k):
        raise AssertionError("collect must not make network calls")

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    assert main(["collect", "--platform", "generic", "--cwd", str(tmp_path)]) == EXIT_OK
    assert "measured" in capsys.readouterr().out


def test_collect_json_is_machine_readable(monkeypatch, tmp_path, capsys):
    assert main(["collect", "--platform", "generic", "--json",
                 "--cwd", str(tmp_path)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["features"]) == set(FEATURES)
    assert payload["platform"] == "generic"


def test_a_feature_flag_overrides_a_measurement(monkeypatch, tmp_path, capsys):
    main(["collect", "--platform", "generic", "--json", "--cwd", str(tmp_path),
          "--test-pass-rate", "0.42", "--diff-size", "999"])
    payload = json.loads(capsys.readouterr().out)
    assert payload["features"]["test_pass_rate"] == 0.42
    assert payload["features"]["diff_size"] == 999
    assert payload["sources"]["diff_size"] == "override"
