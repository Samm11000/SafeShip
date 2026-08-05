# SafeShip Deployment Risk Gate — GitHub Action

Measures the build you are about to deploy and asks SafeShip how risky it looks.

```yaml
permissions:
  contents: read
  actions: read                      # required — see below

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2             # required — see below

      - run: pytest --junitxml=reports/junit.xml   # optional but worth 25% of the score

      - id: safeship
        uses: Samm11000/SafeShip/action/gate@main
        with:
          tenant-id: ${{ secrets.SAFESHIP_TENANT_ID }}
          api-key:   ${{ secrets.SAFESHIP_API_KEY }}
          url:       ${{ secrets.SAFESHIP_URL }}

      - run: ./deploy.sh

      # Tell SafeShip what actually happened. Without this the model never
      # learns from reality — it is the single most valuable step here.
      - if: always()
        uses: Samm11000/SafeShip/action/gate@main
        with:
          tenant-id: ${{ secrets.SAFESHIP_TENANT_ID }}
          api-key:   ${{ secrets.SAFESHIP_API_KEY }}
          url:       ${{ secrets.SAFESHIP_URL }}
          mode:  log
          label: ${{ job.status == 'success' && '0' || '1' }}
```

## The two things you must configure

Both are invisible failures: without them the score still appears, but it rests
on estimates instead of measurements.

### 1. `actions: read`

```yaml
permissions:
  contents: read
  actions: read
```

`GITHUB_TOKEN` defaults to contents+metadata read only on repositories created
after February 2023. Without `actions: read` the workflow-runs API returns 403,
and `recent_failure_rate`, `days_since_deploy` and `build_time_delta` are all
lost — together the largest single block of the model's decision weight. The
action prints the exact fix if this happens.

### 2. `fetch-depth: 2`

```yaml
- uses: actions/checkout@v4
  with:
    fetch-depth: 2      # use 0 if you want accurate pull-request bases
```

`actions/checkout` defaults to `fetch-depth: 1`, so `HEAD~1` does not exist
locally and `git diff HEAD~1` fails — taking `diff_size` and `files_changed`
with it. The action warns when it detects a shallow clone.

On `pull_request` events the base is read from the event payload
(`pull_request.base.sha`) rather than `HEAD~1`, which is more accurate — but that
commit still has to be present, so `fetch-depth: 0` is best for PR workflows.

## Start in advisory mode

`fail-open` defaults to `true`: a BLOCKED verdict is reported loudly and the
workflow continues. Run it that way until you trust the scores, then set
`fail-open: false` to actually enforce.

```yaml
with:
  fail-open: false      # now a BLOCKED verdict fails the job
```

This controls **only** the BLOCKED verdict. A SafeShip outage, a timeout, a 500,
or a bug in the action itself always fails open and is not configurable — a risk
gate that halts everyone's deploys during its own outage gets deleted in a week.

## Inputs

| Input | Default | Notes |
|---|---|---|
| `tenant-id` | *required* | Store as a secret |
| `api-key` | *required* | Store as a secret |
| `url` | — | Your SafeShip base URL |
| `mode` | `score` | `score`, `log`, or `collect` |
| `label` | `0` | `mode: log` only. `0` = fine, `1` = it broke |
| `fail-open` | `true` | `false` makes a BLOCKED verdict fail the job |
| `github-token` | `${{ github.token }}` | Needs `actions: read` |
| `working-directory` | `.` | Repository root to inspect |
| `timeout` | `10` | Seconds before giving up and proceeding |
| `build-id-file` | `safeship_build_id.txt` | How `score` hands the build to `log` |
| `extra-args` | — | Extra CLI flags, e.g. `--no-history` |

## Outputs

| Output | Notes |
|---|---|
| `score` | 0–100. Empty when SafeShip was unreachable |
| `verdict` | `SAFE`, `REVIEW`, `BLOCKED`, or `UNAVAILABLE` |
| `build-id` | Pass to a later `mode: log` step |

```yaml
- run: echo "risk was ${{ steps.safeship.outputs.score }}"
```

## What gets measured

| Source | Features | Needs |
|---|---|---|
| git | `diff_size`, `files_changed`, `is_hotfix` | `fetch-depth: 2` |
| Actions API | `recent_failure_rate`, `days_since_deploy`, `build_time_delta` | `actions: read` |
| JUnit XML | `test_pass_rate` | a test report on disk |
| clock | `hour_of_day`, `day_of_week` | — |
| server | `deployer_exp` | derived from your history, never sent |

Anything that cannot be measured is sent as `null`, and SafeShip imputes it from
your own history and tells you it did. It is never replaced with a plausible
constant — a guess dressed as a measurement is worse than an admitted gap.

To see what your pipeline can measure without scoring anything:

```yaml
- uses: Samm11000/SafeShip/action/gate@main
  with:
    tenant-id: unused
    api-key:   unused
    mode:      collect
```

## Debugging

`mode: collect` prints every feature with its source and the reason for each gap.
It makes no network calls and needs no credentials.

Run it locally the same way:

```bash
git clone https://github.com/Samm11000/SafeShip
PYTHONPATH=SafeShip python3 -m safeship_ci collect
```

stdlib only — no `pip install`, on any Python 3.9+.
