"""
GitHub Actions integration instructions.

The easiest of the three: the runner already holds a token that can read build
history, so there is no credential to create. It just has to be *granted* —
`permissions: actions: read` — which is the single most common thing people miss.
"""
from __future__ import annotations

ID = "github-actions"
NAME = "GitHub Actions"
TAGLINE = "Two secrets and one step. No token to create."
DOCS_URL = "https://docs.github.com/en/actions/security-for-github-actions/security-guides/using-secrets-in-github-actions"
SETUP_EFFORT = "~2 minutes"

#: Where the composite action lives. Referenced by path inside the SafeShip repo,
#: so there is no second repository to keep in sync.
ACTION_REF = "Samm11000/SafeShip/action/gate@main"


def describe(tenant_id, api_key, base_url):
    return {
        "id": ID,
        "name": NAME,
        "tagline": TAGLINE,
        "docs_url": DOCS_URL,
        "setup_effort": SETUP_EFFORT,
        "secrets_location": "Repository → Settings → Secrets and variables → Actions → New repository secret",
        "secrets": [
            {"name": "SAFESHIP_TENANT_ID", "value": tenant_id,
             "where": "Settings → Secrets and variables → Actions"},
            {"name": "SAFESHIP_API_KEY", "value": api_key,
             "where": "Settings → Secrets and variables → Actions"},
            {"name": "SAFESHIP_URL", "value": base_url,
             "where": "Settings → Secrets and variables → Actions"},
        ],
        "prerequisites": [
            {
                "title": "Grant `actions: read`",
                "why": "Reading your own workflow's run history is what supplies "
                       "recent_failure_rate, days_since_deploy and build_time_delta "
                       "— the largest single block of the model's decision weight. "
                       "GITHUB_TOKEN does not include it by default on repositories "
                       "created after February 2023, so the API returns 403 and all "
                       "three are estimated instead of measured.",
                "fix": "permissions:\n  contents: read\n  actions: read",
            },
            {
                "title": "Check out at least 2 commits",
                "why": "actions/checkout defaults to fetch-depth: 1, so HEAD~1 does "
                       "not exist locally and `git diff HEAD~1` fails — taking "
                       "diff_size and files_changed with it.",
                "fix": "- uses: actions/checkout@v4\n  with:\n    fetch-depth: 2"
                       "   # use 0 for accurate pull-request bases",
            },
            {
                "title": "Write a JUnit XML report (optional)",
                "why": "GitHub has no test-results API, so a report on disk is the "
                       "only portable source for test_pass_rate — 25% of the score. "
                       "Without one it is estimated from your history.",
                "fix": "pytest --junitxml=reports/junit.xml"
                       "     # jest --reporters=jest-junit, etc.",
            },
        ],
        "history_note": "Build history comes from your own GITHUB_TOKEN. SafeShip "
                        "never stores a GitHub credential.",
        "snippet": {
            "filename": ".github/workflows/deploy.yml",
            "language": "yaml",
            "code": _workflow(),
        },
        "verify_hint": "Push any commit, or run the workflow manually. The gate "
                       "prints the score and this page flips to Connected.",
    }


def _workflow():
    # str.replace, not %-formatting or .format(): these snippets are full of
    # literal % ("25% of the score") and braces (${{ secrets.X }}), both of which
    # those two mechanisms try to interpret. A plain replace has no syntax.
    return _WORKFLOW.replace("__ACTION__", ACTION_REF)


_WORKFLOW = """\
# Required: reading your own run history supplies the three history features.
permissions:
  contents: read
  actions: read

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2          # required: HEAD~1 must exist for the diff

      # Optional but worth 25% of the score.
      - run: pytest --junitxml=reports/junit.xml
        continue-on-error: true

      - id: safeship
        uses: __ACTION__
        with:
          tenant-id: ${{ secrets.SAFESHIP_TENANT_ID }}
          api-key:   ${{ secrets.SAFESHIP_API_KEY }}
          url:       ${{ secrets.SAFESHIP_URL }}
          # Advisory to begin with: a BLOCKED verdict is reported but does not
          # stop the deploy. Set to 'false' once you trust the scores.
          fail-open: 'true'

      - run: ./deploy.sh

      # Tell SafeShip what actually happened. Without this the model never
      # learns from your deploys — it is the most valuable step here.
      - if: always()
        uses: __ACTION__
        with:
          tenant-id: ${{ secrets.SAFESHIP_TENANT_ID }}
          api-key:   ${{ secrets.SAFESHIP_API_KEY }}
          url:       ${{ secrets.SAFESHIP_URL }}
          mode:  log
          label: ${{ job.status == 'success' && '0' || '1' }}
"""
