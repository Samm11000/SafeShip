"""
Tests for the structural-features prototype.

This is not wired into the model or the feature contract — it is an experiment
in whether the thing SafeShip is missing can be measured cheaply. The tests
exist because a prototype that quietly returns plausible nonsense is worse than
no prototype: it would get promoted on the strength of numbers nobody checked.

The two that matter most:

  - complexity must not be a restatement of diff size, or it adds nothing to a
    model that already has diff_size
  - a pure reformat must not read as a huge risky change, which is the failure
    found by running this over 40 real commits
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from prototypes.structural.extract import (  # noqa: E402
    FEATURE_NAMES, FileChange, extract, formatting_ratio, is_comment,
    lexical_metrics, parse_diff, python_metrics)


# ── diff parsing ─────────────────────────────────────────────────────────────

SAMPLE_DIFF = """\
diff --git a/app/thing.py b/app/thing.py
index 1111111..2222222 100644
--- a/app/thing.py
+++ b/app/thing.py
@@ -10,0 +11,3 @@ def existing():
+def added(x):
+    if x:
+        return 1
@@ -40,2 +43,0 @@ def other():
-def removed():
-    pass
diff --git a/tests/test_thing.py b/tests/test_thing.py
--- a/tests/test_thing.py
+++ b/tests/test_thing.py
@@ -1,0 +2,1 @@
+def test_added(): pass
"""


def test_a_diff_splits_into_files_and_hunks():
    files = parse_diff(SAMPLE_DIFF)
    assert [f.path for f in files] == ["app/thing.py", "tests/test_thing.py"]
    assert len(files[0].hunks) == 2
    assert files[0].hunks[0] == (11, 3)


def test_added_and_removed_lines_are_separated():
    app = parse_diff(SAMPLE_DIFF)[0]
    assert any("def added" in l for l in app.added)
    assert any("def removed" in l for l in app.removed)
    # The +++/--- headers are metadata and must not be counted as content.
    assert not any(l.startswith("+ b/") or "app/thing.py" in l for l in app.added)


def test_hunk_ranges_become_changed_line_numbers():
    app = parse_diff(SAMPLE_DIFF)[0]
    assert 11 in app.changed_lines() and 13 in app.changed_lines()
    assert 100 not in app.changed_lines()


def test_an_empty_diff_is_not_a_crash():
    assert parse_diff("") == []
    assert parse_diff(None) == []


@pytest.mark.parametrize("path,is_test", [
    ("tests/test_thing.py", True),
    ("app/thing_test.go", True),
    ("src/__tests__/x.js", True),
    ("src/x.spec.ts", True),
    ("app/routes/score.py", False),
    ("app/latest.py", False),          # "test" appears inside a word
])
def test_test_files_are_recognised(path, is_test):
    assert FileChange(path).is_test is is_test


@pytest.mark.parametrize("path,is_config", [
    ("docker-compose.yml", True),
    ("Dockerfile", True),
    ("infra/main.tf", True),
    ("package.json", True),
    ("app/main.py", False),
])
def test_config_files_are_recognised(path, is_config):
    assert FileChange(path).is_config is is_config


# ── python, via ast ──────────────────────────────────────────────────────────

FLAT = """
def simple(a):
    return a + 1
"""

BRANCHY = """
def gnarly(items, flag):
    total = 0
    for item in items:
        if item and flag:
            for sub in item:
                while sub:
                    try:
                        total += sub
                    except ValueError:
                        pass
    return total
"""


def test_complexity_rises_with_branching():
    flat = python_metrics(FLAT)["complexity"]
    branchy = python_metrics(BRANCHY)["complexity"]
    assert flat == 1, "a straight-line function has one path"
    assert branchy > flat * 4, f"branchy scored {branchy}, flat {flat}"


def test_nesting_depth_is_measured():
    assert python_metrics(FLAT)["max_nesting"] <= 1
    assert python_metrics(BRANCHY)["max_nesting"] >= 4


def test_only_the_declarations_a_change_touched_are_counted():
    """
    Whole-file complexity would mostly measure how big the file already was —
    editing one line of a 4000-line module would look enormous.
    """
    source = FLAT + BRANCHY
    everything = python_metrics(source)
    # BRANCHY starts after FLAT's 3 lines; touch only the simple function.
    just_flat = python_metrics(source, changed={2, 3})
    assert just_flat["decls_touched"] == 1
    assert just_flat["complexity"] < everything["complexity"]


def test_unparseable_python_returns_none_rather_than_guessing():
    assert python_metrics("def broken(:\n    pass") is None


def test_boolean_operators_count_as_branches():
    a = python_metrics("def f(x):\n    return x\n")["complexity"]
    b = python_metrics("def f(x, y, z):\n    return x and y and z\n")["complexity"]
    assert b > a


# ── the lexical fallback ─────────────────────────────────────────────────────

def test_the_lexical_path_finds_branches_in_other_languages():
    go = ["func handle(x int) int {", "  if x > 0 {", "    for i := range xs {",
          "      switch v {", "      }", "    }", "  }", "  return x", "}"]
    m = lexical_metrics(go)
    assert m["complexity"] >= 2
    assert m["max_nesting"] >= 2
    assert m["method"] == "lexical"


def test_the_lexical_path_labels_itself_as_an_estimate():
    """
    Mixing exact and estimated numbers as though equivalent is how a metric
    quietly stops meaning anything, so the method travels with the value.
    """
    assert lexical_metrics(["if (a) {}"])["method"] == "lexical"
    assert python_metrics(FLAT)["method"] == "ast"


def test_comments_are_not_counted_as_logic():
    commented = ["# if this were code it would branch", "// if (x) {", "x = 1"]
    assert lexical_metrics(commented)["complexity"] == 0


@pytest.mark.parametrize("line,comment", [
    ("# python", True), ("// js", True), ("-- sql", True),
    ("<!-- html -->", True), ("    # indented", True),
    ("x = 1", False), ("", False),
])
def test_comment_detection(line, comment):
    assert is_comment(line) is comment


# ── against this repository ──────────────────────────────────────────────────

def _git(*args):
    return subprocess.run(("git",) + args, cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def test_it_runs_on_a_real_commit_and_returns_the_full_feature_set():
    result = extract("HEAD~1", "HEAD", cwd=REPO)
    assert result["error"] is None
    assert set(result["features"]) == set(FEATURE_NAMES)


def test_a_bad_revision_is_reported_not_raised():
    """Same contract as safeship_ci: never break the build you are advising on."""
    result = extract("nope-not-a-rev", "HEAD", cwd=REPO)
    assert result["error"] is not None
    assert result["features"] == {}


def test_a_pure_reformat_does_not_look_like_a_risky_change():
    """
    THE REGRESSION THIS PROTOTYPE ALREADY HAD. Commit 0360402 normalised line
    endings: git diff reported 35,445 changed lines, git diff -w reported 15.
    Before the fix it scored higher for structural complexity than almost any
    real change in the repository — a mechanical reformat rated as one of the
    riskiest things the project had ever shipped.
    """
    if not _git("cat-file", "-t", "0360402"):
        pytest.skip("history not available")

    reformat = extract("0360402~1", "0360402", cwd=REPO)["features"]
    assert reformat["formatting_ratio"] > 0.9, "should be almost entirely formatting"
    assert reformat["added_complexity"] == 0, (
        "a reformat added no logic, so it must add no complexity"
    )


def test_a_substantive_commit_is_not_mistaken_for_formatting():
    if not _git("cat-file", "-t", "87a6db6"):
        pytest.skip("history not available")

    real = extract("87a6db6~1", "87a6db6", cwd=REPO)["features"]
    assert real["formatting_ratio"] < 0.1
    assert real["added_complexity"] > 100


def test_formatting_ratio_is_none_when_there_is_nothing_to_compare():
    assert formatting_ratio("HEAD", "HEAD", cwd=REPO) is None


def test_complexity_is_not_just_diff_size_wearing_a_hat():
    """
    The whole point. If structural complexity simply tracked change size, it
    would add nothing to a model that already has diff_size — and the Prime
    Video finding that structure beats volume would not transfer.

    Measured across 30 real commits of this repository, the correlation is
    approximately 0.00.
    """
    import numpy as np

    shas = _git("log", "--format=%h", "-30").split()
    complexity, churn = [], []
    for sha in shas:
        r = extract(f"{sha}~1", sha, cwd=REPO)
        if r["error"] or not r["files"]:
            continue
        stat = _git("diff", "--numstat", f"{sha}~1", sha)
        lines = sum(int(t) for row in stat.splitlines()
                    for t in row.split("\t")[:2] if t.isdigit())
        complexity.append(r["features"]["added_complexity"])
        churn.append(lines)

    assert len(complexity) > 10, "not enough history to judge"
    corr = abs(np.corrcoef(complexity, churn)[0, 1])
    assert corr < 0.75, (
        f"added_complexity correlates {corr:.2f} with diff_size — it is mostly "
        "restating change size rather than measuring structure"
    )


def test_it_stays_inside_the_pipeline_time_budget():
    """safeship_ci budgets under 3s for everything. A build is waiting."""
    import time

    start = time.time()
    extract("HEAD~3", "HEAD", cwd=REPO)
    assert time.time() - start < 3.0
