"""
Bitbucket Pipelines integration instructions.

The one place where the required credential is genuinely non-obvious: unlike
GitHub, Bitbucket hands a pipeline no API token by default, so reading build
history needs a Repository Access Token created by hand.
"""
from __future__ import annotations

ID = "bitbucket"
NAME = "Bitbucket Pipelines"
TAGLINE = "One extra token to create, then the same two steps."
DOCS_URL = "https://support.atlassian.com/bitbucket-cloud/docs/variables-and-secrets/"
SETUP_EFFORT = "~5 minutes"

CLONE_URL = "https://github.com/Samm11000/SafeShip"


def describe(tenant_id, api_key, base_url):
    return {
        "id": ID,
        "name": NAME,
        "tagline": TAGLINE,
        "docs_url": DOCS_URL,
        "setup_effort": SETUP_EFFORT,
        "secrets_location": "Repository settings → Repository variables (tick Secured)",
        "secrets": [
            {"name": "SAFESHIP_TENANT_ID", "value": tenant_id,
             "where": "Repository settings → Repository variables (Secured)"},
            {"name": "SAFESHIP_API_KEY", "value": api_key,
             "where": "Repository settings → Repository variables (Secured)"},
            {"name": "SAFESHIP_URL", "value": base_url,
             "where": "Repository settings → Repository variables"},
        ],
        "prerequisites": [
            {
                "title": "Create a Repository Access Token",
                "why": "Bitbucket gives a pipeline no API token of its own, so build "
                       "history needs one you create. A Repository Access Token is "
                       "the right primitive here: it is scoped to this repository "
                       "and revocable, unlike an account-wide app password.",
                "fix": "Repository settings → Access tokens → Create Repository "
                       "Access Token, scope 'pipeline:read',\n"
                       "then add it as the secured variable "
                       "BITBUCKET_ACCESS_TOKEN.",
            },
            {
                "title": "Raise the clone depth",
                "why": "Bitbucket clones shallow by default, so HEAD~1 is absent and "
                       "diff_size and files_changed cannot be measured.",
                "fix": "clone:\n  depth: 2",
            },
            {
                "title": "Write a JUnit XML report (optional)",
                "why": "Bitbucket has no test-results API, so a report on disk is "
                       "the only source for test_pass_rate — 25% of the score.",
                "fix": "pytest --junitxml=test-results/junit.xml",
            },
        ],
        "history_note": "History is read with your Repository Access Token, from "
                        "inside your pipeline. SafeShip stores no Bitbucket "
                        "credential.",
        "snippet": {
            "filename": "bitbucket-pipelines.yml",
            "language": "yaml",
            "code": _pipelines_yml(),
        },
        "verify_hint": "Push any commit. The risk gate step prints the score and "
                       "this page flips to Connected.",
    }


def _pipelines_yml():
    # Literal replace rather than %-formatting: see the note in github.py.
    return _PIPELINES_YML.replace("__CLONE__", CLONE_URL)


_PIPELINES_YML = """\
image: python:3.11

# Required: the default clone is shallow, so HEAD~1 would not exist.
clone:
  depth: 2

pipelines:
  default:
    - step:
        name: Test
        script:
          - pip install pytest
          - pytest --junitxml=test-results/junit.xml || true
        artifacts:
          - test-results/**

    - step:
        name: SafeShip risk gate
        script:
          # stdlib-only, so there is nothing to install.
          - git clone --depth 1 __CLONE__ /tmp/safeship-ci
          # Exits non-zero ONLY on a BLOCKED verdict. An outage, timeout or 500
          # warns and exits 0. Drop --fail-open to start enforcing.
          - PYTHONPATH=/tmp/safeship-ci python3 -m safeship_ci score --fail-open

    - step:
        name: Deploy
        deployment: production
        script:
          - ./deploy.sh
        after-script:
          # The learning signal: BITBUCKET_EXIT_CODE is 0 when the step passed.
          - git clone --depth 1 __CLONE__ /tmp/safeship-ci 2>/dev/null || true
          - PYTHONPATH=/tmp/safeship-ci python3 -m safeship_ci log
            $([ "$BITBUCKET_EXIT_CODE" = "0" ] && echo 0 || echo 1) || true
"""
