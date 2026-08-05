"""
Tests for the GitHub composite action.

The action is a YAML file, so nothing type-checks it and nothing imports it — a
typo ships silently and then fails inside a stranger's pipeline. These tests
cover the parts that would break quietly:

  - the file still parses and declares the shape Actions requires
  - no input is interpolated into a shell body (script injection)
  - the modes it dispatches on are the ones the CLI actually implements
  - the two documented prerequisites stay documented

PyYAML is only a transitive dependency here, so the module skips rather than
fails if it is absent.
"""
from __future__ import annotations

import os
import re

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is not a declared dependency")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACTION_DIR = os.path.join(REPO, "action", "gate")
ACTION_YML = os.path.join(ACTION_DIR, "action.yml")


@pytest.fixture(scope="module")
def action():
    with open(ACTION_YML, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def raw():
    with open(ACTION_YML, "r", encoding="utf-8") as fh:
        return fh.read()


def _run_bodies(action):
    return [s.get("run", "") for s in action["runs"]["steps"]]


# ── shape ────────────────────────────────────────────────────────────────────

def test_the_action_declares_what_github_requires(action):
    assert action["name"]
    assert action["description"]
    assert action["runs"]["using"] == "composite"
    assert action["runs"]["steps"], "a composite action needs at least one step"


def test_credentials_are_required_and_everything_else_has_a_default(action):
    required = {k for k, v in action["inputs"].items() if v.get("required")}
    assert required == {"tenant-id", "api-key"}, (
        "only the credentials should be required — anything else forces "
        "boilerplate on every user"
    )
    for name, spec in action["inputs"].items():
        if name not in required:
            assert "default" in spec, f"optional input {name!r} has no default"


def test_every_input_is_documented(action):
    for name, spec in action["inputs"].items():
        assert spec.get("description", "").strip(), f"{name} has no description"


def test_outputs_are_wired_to_the_step_that_produces_them(action):
    step_ids = {s.get("id") for s in action["runs"]["steps"] if s.get("id")}
    assert "gate" in step_ids
    for name, spec in action["outputs"].items():
        value = spec["value"]
        assert "steps.gate.outputs." in value, f"output {name} is not wired to steps.gate"


def test_the_documented_outputs_are_the_ones_the_cli_emits(action):
    """The CLI writes these keys to GITHUB_OUTPUT; the action must expose the same."""
    from safeship_ci import cli

    source = open(cli.__file__, "r", encoding="utf-8").read()
    emitted = set(re.findall(r'_gha_output\(\s*"([a-z\-]+)"', source))
    declared = set(action["outputs"])
    assert declared <= emitted, (
        f"action.yml declares outputs the CLI never writes: {declared - emitted}"
    )


# ── injection ────────────────────────────────────────────────────────────────

def test_no_input_is_interpolated_into_a_shell_body(action):
    """
    ${{ inputs.x }} inside `run:` splices the value into the shell source before
    it executes, so a value containing shell syntax runs as code. Passing through
    `env:` and reading "$VAR" cannot be escaped that way.
    """
    for step, body in zip(action["runs"]["steps"], _run_bodies(action)):
        offenders = re.findall(r"\$\{\{\s*(inputs|github\.event)\.[^}]+\}\}", body)
        assert not offenders, (
            f"step {step['name']!r} interpolates {offenders} into its script — "
            "pass it via env: and read \"$VAR\" instead"
        )


def test_values_used_by_the_script_are_supplied_through_env(action):
    gate = [s for s in action["runs"]["steps"] if s.get("id") == "gate"][0]
    env = gate["env"]
    for var in ("SS_MODE", "SS_FAIL_OPEN", "SS_TIMEOUT", "SS_BUILD_ID_FILE",
                "SAFESHIP_TENANT_ID", "SAFESHIP_API_KEY", "SAFESHIP_URL"):
        assert var in env, f"{var} is not passed through env:"


def test_shell_variables_holding_paths_are_quoted(action):
    gate_body = [s.get("run", "") for s in action["runs"]["steps"]
                 if s.get("id") == "gate"][0]
    # A path with a space would otherwise word-split.
    assert '"$SS_BUILD_ID_FILE"' in gate_body
    assert '"$SS_TIMEOUT"' in gate_body


# ── behaviour ────────────────────────────────────────────────────────────────

def test_the_dispatched_modes_are_the_cli_subcommands(action):
    gate_body = [s.get("run", "") for s in action["runs"]["steps"]
                 if s.get("id") == "gate"][0]
    # Indentation-agnostic: YAML strips the block scalar's common indent, so the
    # depth here is not the depth in the file.
    dispatched = set(re.findall(r"^\s+(score|log|collect|watch)\)\s*$", gate_body, re.M))
    assert dispatched, "the case statement did not parse"

    from safeship_ci.cli import build_parser

    sub = [a for a in build_parser()._actions if hasattr(a, "choices") and a.dest == "cmd"]
    available = set(sub[0].choices)
    assert dispatched <= available, (
        f"action.yml dispatches modes the CLI does not implement: {dispatched - available}"
    )

    documented = set(re.findall(r"\b(score|log|collect)\b",
                               action["inputs"]["mode"]["description"]))
    assert documented == dispatched, (
        f"the mode description documents {documented} but the script handles {dispatched}"
    )


def test_fail_open_defaults_to_advisory(action):
    """
    A gate that starts out blocking deploys on a model the user has not yet
    learned to trust gets removed on day one. Advisory first is deliberate.
    """
    assert action["inputs"]["fail-open"]["default"] == "true"


def test_fail_open_is_documented_as_covering_only_the_verdict(action):
    text = action["inputs"]["fail-open"]["description"].lower()
    assert "blocked" in text
    # The important half: infra failures are unconditionally fail-open.
    assert "outage" in text or "not configurable" in text


def test_the_action_never_fails_the_job_on_a_misconfiguration(action):
    """
    The prerequisite step warns; only a genuinely unknown mode is an error. A
    shallow clone or a missing URL must not be the reason a pipeline goes red.
    """
    pre = action["runs"]["steps"][0]
    body = pre["run"]
    assert "::warning" in body
    assert "exit 1" not in body, "the prerequisite check must not fail the job"


def test_it_locates_the_package_relative_to_the_action_and_degrades_if_missing(action):
    gate_body = [s.get("run", "") for s in action["runs"]["steps"]
                 if s.get("id") == "gate"][0]
    # action/gate/ -> ../.. is the repo root, where safeship_ci lives.
    assert "GITHUB_ACTION_PATH" in gate_body
    assert "PYTHONPATH" in gate_body
    assert "exit 0" in gate_body, "a missing package must skip the gate, not fail it"


def test_no_pip_install_on_the_critical_path(action):
    """safeship_ci is stdlib-only so the action stays a single python3 call."""
    for body in _run_bodies(action):
        assert "pip install" not in body
        assert "pip3 install" not in body


# ── the prerequisites that silently degrade the score ────────────────────────

def test_the_two_invisible_prerequisites_are_documented(raw):
    assert "actions: read" in raw
    assert "fetch-depth" in raw


def test_the_readme_leads_with_a_workflow_that_includes_both(raw):
    with open(os.path.join(ACTION_DIR, "README.md"), "r", encoding="utf-8") as fh:
        readme = fh.read()

    first_block = readme.split("```")[1]
    assert "actions: read" in first_block, (
        "the copy-paste example must include actions: read — most users will "
        "copy it verbatim and never read the prose"
    )
    assert "fetch-depth: 2" in first_block


def test_the_readme_shows_how_outcomes_get_logged(raw):
    with open(os.path.join(ACTION_DIR, "README.md"), "r", encoding="utf-8") as fh:
        readme = fh.read()
    # Without labels the model never learns; this cannot be an afterthought.
    assert "mode:  log" in readme or "mode: log" in readme
    assert "if: always()" in readme
