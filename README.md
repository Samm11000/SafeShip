# 🚢 SafeShip

## AI-Powered Deployment Risk Intelligence Platform

SafeShip predicts whether a software release is likely to fail **before it reaches production**, watches the service **after** it deploys, and learns from what actually happened.

**It is CI-agnostic.** The API is plain HTTP + JSON, so anything that can make a POST request can gate its deploys — GitHub Actions, Jenkins, GitLab CI, CircleCI, Buildkite, Argo, or a shell script. There is no Jenkins dependency anywhere in the scoring path; Jenkins is simply one adapter, and an optional one.

---

## Table of Contents

1. Problem Statement
2. How SafeShip Works
3. System Architecture
4. Core Features
4a. Integrations — GitHub Actions, Jenkins, Bitbucket, anything
4b. Sentinel — the post-deploy safety net
5. Machine Learning Model
6. Why Random Forest (Current Choice)
7. Future Upgrade: XGBoost
8. Training Dataset Strategy
9. Model Retraining Lifecycle
10. Drift Detection
11. AWS Infrastructure
12. Why ECR is Used
13. Deployment Options on AWS
14. API Endpoints
15. Dashboard Features
16. Security & Multi-Tenancy
17. Performance & Metrics
18. Roadmap
19. Quick Start

---

## 1. Problem Statement

Traditional CI/CD tools can verify:

* Build success
* Test execution
* Artifact creation
* Deployment automation

But they usually cannot answer:

> **Should this release be deployed right now?**

Many incidents happen after successful builds due to:

* Hidden bugs
* Weak tests
* Large risky changes
* Hotfix pressure
* Unstable recent release history
* Bad deployment timing
* Config mistakes

SafeShip adds an AI decision layer before deployment.

---

## 2. How SafeShip Works

The full loop — pre-deploy gate, post-deploy watch, and an automatic learning signal:

1. **Score** — your pipeline POSTs build signals to `/score`
2. **Predict** — the model returns risk `0–100` with the top reasons behind it
3. **Gate** — pipeline allows / warns / blocks on `SAFE | WARNING | BLOCKED`
4. **Deploy** — your normal deploy runs, untouched
5. **Watch** — **Sentinel** probes the service, compares error rate and latency
   against a baseline, alerts and can **trigger a rollback** if it regresses
6. **Learn** — the outcome is POSTed to `/log`. Sentinel's exit code supplies the
   label automatically, so nobody has to remember to report what happened
7. **Improve** — nightly retraining produces a model personalised to your team

Step 6 is what makes the loop close by itself. A "learns from outcomes" system
that depends on humans reporting outcomes does not learn.

---

## 3. System Architecture

* **Flask API** (gunicorn): scoring, logging, auth
* **Frontend Dashboard**: tenant analytics, setup guide, charts, live demo
* **Sentinel**: standalone post-deploy health watchdog + rollback trigger
* **S3**: per-build records, datasets, model files, backups
* **DynamoDB**: tenant metadata, hashed API keys, thresholds
* **Lambda**: scheduled retraining and drift detection
* **ECR**: retraining container image (ML deps exceed Lambda's zip limit)
* **Your CI**: any system that can POST — see Integrations below

---

## 4. Core Features

* Real-time deployment risk scoring
* Explainable risk reasons
* Personalized tenant models
* Automatic daily retraining
* Historical charts and metrics
* API key based access
* Jenkins-ready setup snippets
* Multi-tenant SaaS design

---

## 4a. Integrations

**Start at `/setup`.** It asks which CI you use and hands you the exact secret
names, the prerequisites for that platform, and a snippet to paste — then polls
until your first real build arrives and shows you its score. Everything below is
what that wizard generates, if you would rather read it than click it.

### `safeship_ci` — the thing that does the measuring

SafeShip scores ten features, and seven of them cannot be obtained from a bare
HTTP call: they need your repository and your CI's own API.
[`safeship_ci/`](safeship_ci/) is a small extractor that runs *inside* your
pipeline and collects them:

| Source | Features | Needs |
|---|---|---|
| git | `diff_size`, `files_changed`, `is_hotfix` | 2 commits of history |
| your CI's API | `recent_failure_rate`, `days_since_deploy`, `build_time_delta` | your own CI token |
| JUnit XML | `test_pass_rate` | a test report on disk |
| the clock | `hour_of_day`, `day_of_week` | — |
| SafeShip | `deployer_exp` | derived server-side, never sent |

It is **stdlib-only** — no `pip install`, no dependency tree to conflict with
your build's — and it uses **your** CI's native token, so SafeShip stores no
third-party credentials and needs no network path into your infrastructure. That
is the only design that works for a firewalled or on-prem Jenkins.

Anything it cannot measure is sent as `null`, never as a plausible-looking
constant. The server imputes it from your own history and the response says which
values were guessed — so partial adoption is safe and honest, and a score built
mostly on medians does not look identical to one built on facts.

To see exactly what your pipeline can measure, before scoring anything:

```bash
python3 -m safeship_ci collect      # no credentials, no network calls
```

### GitHub Actions

```yaml
permissions:
  contents: read
  actions: read                      # required — see below

steps:
  - uses: actions/checkout@v4
    with: { fetch-depth: 2 }         # required — see below

  - uses: Samm11000/SafeShip/action/gate@main
    with:
      tenant-id: ${{ secrets.SAFESHIP_TENANT_ID }}
      api-key:   ${{ secrets.SAFESHIP_API_KEY }}
      url:       ${{ secrets.SAFESHIP_URL }}

  - run: ./deploy.sh

  - if: always()                      # the learning loop
    uses: Samm11000/SafeShip/action/gate@main
    with:
      tenant-id: ${{ secrets.SAFESHIP_TENANT_ID }}
      api-key:   ${{ secrets.SAFESHIP_API_KEY }}
      url:       ${{ secrets.SAFESHIP_URL }}
      mode:  log
      label: ${{ job.status == 'success' && '0' || '1' }}
```

Two prerequisites, both of which fail **silently** — you still get a score, it is
just built on estimates:

- **`actions: read`.** `GITHUB_TOKEN` does not grant it by default on repositories
  created after February 2023, so the workflow-runs API returns 403 and the three
  history features are lost.
- **`fetch-depth: 2`.** `actions/checkout` defaults to 1, so `HEAD~1` does not
  exist and `git diff HEAD~1` fails, taking `diff_size` and `files_changed` with
  it. Use `0` for accurate pull-request bases.

Full reference: [`action/gate/README.md`](action/gate/README.md).

### Jenkins

Jenkins is the only one of the three that exposes test results through its own
API, so `test_pass_rate` can come from Jenkins rather than from a file.

```groovy
environment {
  SAFESHIP_URL       = credentials('SAFESHIP_URL')
  SAFESHIP_TENANT_ID = credentials('SAFESHIP_TENANT_ID')
  SAFESHIP_API_KEY   = credentials('SAFESHIP_API_KEY')
  JENKINS_USER       = credentials('JENKINS_USER')   // needed for history
  JENKINS_TOKEN      = credentials('JENKINS_TOKEN')  // an API token, not a password
}

stage('SafeShip risk gate') {
  steps {
    sh '''
      [ -d /tmp/safeship-ci ] || git clone --depth 1 https://github.com/Samm11000/SafeShip /tmp/safeship-ci
      PYTHONPATH=/tmp/safeship-ci python3 -m safeship_ci score --fail-open
    '''
  }
}

post {
  success { sh 'PYTHONPATH=/tmp/safeship-ci python3 -m safeship_ci log 0 || true' }
  failure { sh 'PYTHONPATH=/tmp/safeship-ci python3 -m safeship_ci log 1 || true' }
}
```

Needs `python3` on the agent, an API token for history, no shallow clone, and
`junit 'reports/junit.xml'` if you want test results read natively.

### Bitbucket Pipelines

```yaml
clone:
  depth: 2                            # required: the default clone is shallow

pipelines:
  default:
    - step:
        name: SafeShip risk gate
        script:
          - git clone --depth 1 https://github.com/Samm11000/SafeShip /tmp/safeship-ci
          - PYTHONPATH=/tmp/safeship-ci python3 -m safeship_ci score --fail-open
```

Bitbucket hands a pipeline no API token, so build history needs a **Repository
Access Token** (scoped to the repo and revocable, unlike an account-wide app
password) exposed as the secured variable `BITBUCKET_ACCESS_TOKEN`.

### Anything else

`generic` mode measures the git and clock features and reports the rest as
unknown, which the server then imputes:

```bash
PYTHONPATH=/path/to/SafeShip python3 -m safeship_ci score \
  --url "$SAFESHIP_URL" --tenant-id "$TENANT" --api-key "$KEY"
```

You can supply anything it could not detect with `--test-pass-rate 0.94`, or via
`SAFESHIP_TEST_PASS_RATE`. Adding a platform properly is one adapter file in
[`safeship_ci/adapters/`](safeship_ci/adapters/) plus one entry in
[`app/integrations/`](app/integrations/).

### Fail-open, deliberately

Exit `1` means **BLOCKED** — the gate worked and the model says this deploy is
risky. An outage, a timeout, a 500, or a bug in the extractor **warns and exits
0**. Conflating those two is the difference between a useful gate and an outage
amplifier, and a gate that halts everyone's deploys during its own incident gets
deleted within a week.

`--fail-open` additionally downgrades a real BLOCKED verdict to a warning. Start
there, and turn it off when you trust the scores.

---

## 4b. Sentinel — the post-deploy safety net

Scoring guesses before the fact. Sentinel checks reality after it.

It probes your service for a short window, compares error rate and p95 latency
against a quick baseline, and if things regress it alerts and can run a rollback
command. It needs **nothing but a URL** — no Datadog, no Prometheus, no
Kubernetes.

```bash
python sentinel/safeship_sentinel.py \
    --url https://myservice/health \
    --window 120 --interval 5 \
    --error-rate 0.20 --latency-mult 2.0 \
    --rollback-cmd "kubectl rollout undo deploy/myservice" \
    --slack-webhook https://hooks.slack.com/services/...
```

Exit code `0` = HEALTHY, `1` = DEGRADED — so any CI can gate or roll back on it.

### It labels the build itself

Given credentials (`SAFESHIP_URL`, `SAFESHIP_TENANT_ID`, `SAFESHIP_API_KEY`) it
posts the outcome to `/log` when the window closes, against the build the risk gate
scored. So the learning loop needs no extra pipeline step — and an optional step
that teaches the model is exactly the step people don't add.

This matters beyond convenience, because **Sentinel's verdict is better evidence
than a pipeline's exit code:**

- A pipeline that fails *before* the deploy step never deployed. Labelling that as
  a deploy failure is simply wrong — it is a fact about your tests, not about
  deployment risk.
- A pipeline that goes green while production breaks is the most valuable row in
  the dataset, and pipeline status will never reveal it.

So `sentinel_healthy` / `sentinel_degraded` are weighted 1.0 during retraining
while `ci_success` / `ci_failure` are weighted lower, and **an observation is never
overwritten by an inference** — if Sentinel has already reported what it saw, a
later pipeline-status fallback is rejected and says why. See
[`app/labels.py`](app/labels.py) for the full taxonomy.

Both outcomes are reported, not just failures: successes are the majority class,
and a model that has never seen a normal deploy cannot recognise an abnormal one.
Labelling failures never changes the exit code — losing a label costs one row,
while failing a pipeline costs a deploy.

---

## 5. Machine Learning Model

Current production model uses `RandomForestClassifier`.

### Input Features

The order below is the contract. It lives in exactly one place,
[`ml/features.py`](ml/features.py), because a positional model input that
silently reorders mispredicts without erroring — `tests/test_features.py` fails if
any copy of the list drifts.

| # | Feature | Importance | Collected by |
|---|---|---|---|
| 1 | `diff_size` | 0.176 | git |
| 2 | `files_changed` | low | git |
| 3 | `hour_of_day` | low | the server's clock |
| 4 | `day_of_week` | low | the server's clock |
| 5 | `recent_failure_rate` | **0.278** | your CI's API |
| 6 | `test_pass_rate` | **0.250** | JUnit XML, or Jenkins' API |
| 7 | `is_hotfix` | low | git branch name |
| 8 | `deployer_exp` | low | **derived server-side** — see below |
| 9 | `days_since_deploy` | low | your CI's API |
| 10 | `build_time_delta` | low | your CI's API |

Two things follow from those importances and are worth knowing:

- **`recent_failure_rate` and `test_pass_rate` are 52.8% of the decision.** Any
  integration that does not really measure them is not really scoring the build.
  This is why `safeship_ci` exists: the earlier integrations sent `0.0` and `1.0`
  for these — the most reassuring values in each range — so everything came back
  SAFE.
- **`deployer_exp` is never taken from the request.** It is counted from the
  tenant's own build history, keyed on the actor the CI reports. A client hint is
  accepted only for an actor with no history, and only if it is *lower* than what
  would have been imputed: claiming inexperience is self-penalising and therefore
  credible, while claiming to be a veteran is unverifiable and is exactly the
  attack.

### Output

* Probability of risky deployment
* Converted to score 0–100
* Verdicts: SAFE / WARNING / BLOCKED

---

## 6. Why Random Forest (Current Choice)

Random Forest was selected because it is the best current tradeoff for an MVP / early production system.

### Advantages

* Strong performance on tabular data
* Handles nonlinear feature interactions
* Resistant to overfitting vs single tree
* Works well on small/medium datasets
* Fast inference (<200ms target)
* Native feature importance for explainability
* CPU-friendly retraining

### Example Interaction It Learns

Large diff + low tests + hotfix + recent failures = risky

### Key Hyperparameters

* n_estimators=100
* max_depth=8
* class_weight=balanced
* min_samples_leaf=3
* random_state=42
* n_jobs=-1

---

## 7. Future Upgrade: XGBoost

XGBoost is a future candidate when dataset volume grows.

### Why Later?

* More tuning complexity today
* Random Forest already strong on current scale
* Need larger labelled data to justify migration

### Benefits of Future XGBoost

* Often higher accuracy on mature tabular datasets
* Better handling of subtle interactions
* Strong regularization

Planned benchmark path:
Random Forest vs XGBoost vs LightGBM using tenant cohorts.

---

## 8. Training Dataset Strategy

No public labelled deployment-failure dataset exists, so the cold start is solved
with synthetic data. It is worth being precise about what that means, because it
is easy to overstate.

### Phase 1: Synthetic bootstrap — a rule, not a discovery

`ml/generate_synthetic.py` invents 3000 builds and labels them with a
hand-written weighted rule encoding ordinary DevOps beliefs: large changes are
riskier, hotfixes are riskier, low test pass rates are riskier, unstable
pipelines are riskier. `ml/train_base_model.py` then fits a Random Forest to
those labels.

So **the base model is that rule, compressed into a forest.** A 19-line
reimplementation of the label generator scores AUC 0.968 on the same data against
the forest's 0.974 — the model buys 0.006 AUC over the rules it was trained to
imitate, and its output correlates 0.89 with the generator's raw risk score.

That is fine for a cold start — a sensible prior beats no opinion, and it is
transparent and cheap. But it means the base model's metrics measure *how well
the forest recovered a known function*, not how well it predicts deployment
failures. There is no real-world signal in it, and none of the usual ML
validation applies, because the labels were not observations.

**Treat `base` as a heuristic prior, not a trained model.**

### Phase 2: Real outcomes — where the actual learning happens

The bottleneck is labels, not model architecture. A label arrives when something
calls `/log`, which is what the `log` step in every integration snippet above is
for — and it is the single most valuable line in those snippets.

Better still, label from **Sentinel** rather than from pipeline status. "Did
production degrade after this deploy" is the real target; "did the pipeline go
red" is a proxy that misses the important case, a green pipeline that broke
production. Sentinel's exit code does this automatically.

One structural limitation worth stating: **a gate cannot learn from the deploys it
blocks.** If SafeShip blocks a risky deploy, nobody ever finds out whether it
would have failed, so the model only ever learns from deploys it allowed. This is
the selective-labels problem, it is inherent to gates rather than a bug, and it is
why "deploy anyway" should be an easy, instrumented button: overrides on BLOCKED
builds are the only counterfactual labels that exist.

---

## 9. Model Retraining Lifecycle

### Schedule

Daily scheduled retraining (e.g. 2 AM UTC).

### Steps

1. Trigger Lambda
2. Pull the tenant's labelled builds from S3 (90-day rolling window)
3. Require at least 200 labelled builds, else skip
4. Impute missing features from medians — **the same way `/score` does**, so the
   model is not trained on a different distribution than it is asked to judge
5. Split **by time**, holding out the most recent 20%
6. Apply SMOTE to the training set only
7. Train the candidate
8. Validate: enough held-out rows, enough held-out failures, precision, AUC, and
   no regression against the incumbent measured on the same window
9. If every check passes, swap the model in S3 (candidate → live)
10. API hot-reloads within 5 minutes

Any failed check keeps the existing model and alerts Slack. Skipping is always
preferred to promoting something unmeasurable.

### Time Required

* Small tenants: seconds
* Medium tenants: under a minute
* Multi-tenant batch: few minutes

---

## 10. Drift Detection

Models degrade when team behavior changes.

### Examples

* New release frequency
* Better testing culture
* New team members
* Microservices migration

### Detection Signals

* Falling precision
* Rising false negatives
* Score distribution shifts
* Feature distribution changes
* More manual overrides

### Mitigation

Daily retraining + metric monitoring.

---

## 11. AWS Infrastructure

### EC2

Hosts Flask scoring API.

### S3

Stores:

* tenant datasets
* model.pkl files
* backups
* archives

### DynamoDB

Stores:

* tenant_id
* api_key
* thresholds
* model_phase
* precision
* build counts

### Lambda

Runs retraining jobs on schedule.

---

## 12. Why ECR is Used

Lambda retraining dependencies became too large for zip package limits.

ECR stores a Docker image containing:

* Python runtime
* pandas
* scikit-learn
* imbalanced-learn
* boto3
* retrain scripts

### Benefits

* Larger package support
* Versioned deployments
* Reproducible environment
* Easier ML dependency management

---

## 13. Deployment Options on AWS

### Current

* Flask API on EC2
* Lambda retrain worker
* S3 + DynamoDB backend

### Alternative Production Paths

1. ECS Fargate containers
2. EKS Kubernetes
3. API Gateway + Lambda scoring
4. Elastic Beanstalk
5. Multi-AZ EC2 Auto Scaling

---

## 14. API Endpoints

Two endpoints are the entire integration surface — there is no SDK or plugin to
install.

### POST /score

Scores a build. Every feature is optional: send `null`, or omit it, and SafeShip
imputes it from your history and tells you it did.

```json
{
  "score": 74, "verdict": "BLOCKED", "color": "red", "model_phase": "base",
  "build_id": "dg-a1b2c3d4-e5f6a7b8",
  "top_reasons": [
    {"feature": "recent_failure_rate", "label": "Recent failure rate",
     "importance": 0.278, "value": 0.4, "imputed": false, "source": "provided",
     "value_str": "40% of last 10 builds failed"}
  ],
  "imputed": ["build_time_delta"],
  "feature_sources": {"diff_size": "provided", "build_time_delta": "tenant_median"}
}
```

`imputed` and each reason's `imputed`/`source` are the honest part: a verdict built
mostly on medians should not look identical to one built on measurements, and a
`value_str` for an estimated feature is suffixed `— estimated` so it cannot be
mistaken for a fact.

`deployer_exp` is ignored if you send it. It is derived from your own build history
so it cannot be inflated to lower your score.

### POST /log

Records what actually happened — `label: 0` fine, `1` broke. This is the training
signal; without it the model never learns from your deploys.

### Supporting endpoints

| Endpoint | Purpose |
|---|---|
| `GET /setup` | Onboarding wizard — pick a platform, get its snippet, verify |
| `GET /api/setup/status` | Whether a first build has arrived yet, and its score |
| `GET /dashboard` | Tenant analytics UI |
| `GET /health` | Liveness — always cheap, never touches AWS |
| `GET /ready` | Readiness — probes S3, DynamoDB and the model |
| `GET /metrics` | Request, latency and error counters |

---

## 15. Dashboard Features

* Build counts
* Labelled build progress
* Model phase badge
* Risk charts
* Feature importance chart
* Jenkins setup cards
* Copy-paste integration snippets

---

## 16. Security & Multi-Tenancy

* Unique tenant IDs
* API keys per tenant
* Per-tenant datasets
* Per-tenant models
* Isolated analytics

---

## 17. Performance & Metrics

**Serving:** p50 latency under 200ms on a warm model cache; `/metrics` exposes
live request, latency and error counters, and `/ready` probes S3, DynamoDB and the
model itself.

**Model quality — read section 8 first.** The base model reports precision 0.851,
recall 0.864 and AUC-ROC 0.935, and those numbers are *not* evidence that SafeShip
predicts deployment failures. They are measured against synthetic labels produced
by a hand-written rule, so they quantify how faithfully the forest reproduced that
rule. A 19-line reimplementation of the rule scores AUC 0.968 on the same data.

Honest accuracy figures require real labelled outcomes, and per-tenant models are
only promoted once there are enough of them to measure on:

* 200 labelled builds minimum
* a **time-ordered** held-out set of at least 40 builds, containing at least 5
  failures — a random split would train on builds that happened after the ones it
  is scored on, which reports a number the deployed model cannot achieve
* precision at least **1.5× your own deploy-failure rate** — not a fixed number.
  Precision is bounded below by the base rate, so a single threshold means
  different things for different teams: at a 76% failure rate a model that flags
  *every* build scores 0.760 and would have been promoted, while a team at the
  DORA-elite rate under 5% could almost never clear a fixed 0.75 however good
  its model was. That is backwards — SafeShip is worth least to the first team
  and most to the second. Measured leave-one-project-out on ApacheJIT, lift
  ranged 1.20× to 3.26× across 14 projects (median 1.99×)
* AUC ≥ 0.70, which is base-rate independent and so stays absolute
* and no regression against the model already in production

Below those floors the retrain job **skips rather than promotes**. Reporting
precision measured on two rows would be worse than reporting nothing.

---

## 18. Roadmap

Shipped:

* ✅ `safeship_ci` — real feature extraction for GitHub Actions, Jenkins and
  Bitbucket Pipelines, replacing hardcoded constants
* ✅ `action/gate` composite action for GitHub Actions
* ✅ Platform-aware onboarding wizard at `/setup`, with first-build verification
* ✅ Unmeasured features imputed from history and reported as estimates, rather
  than silently defaulted to their most reassuring values
* ✅ `deployer_exp` derived server-side, so it cannot be spoofed
* ✅ Slack alerts (per-tenant incoming webhook)
* ✅ Post-deploy watchdog + rollback trigger (Sentinel)
* ✅ Append-only per-build storage (dashboard cost no longer grows with history)
* ✅ Structured JSON logging, `/metrics`, `/ready`

Next, roughly in order of how much it matters:

* **HTTPS + a domain.** The dashboard currently hands out a bare-IP `http://` URL,
  so every build sends its API key in cleartext. This gates everything else.
* **Automatic labelling from Sentinel**, so the model gets real outcomes without
  anyone remembering to add a step (see section 8)
* **Model versioning with one-step rollback** — more urgent now that better
  features make retraining actually change behaviour
* GitLab CI / CircleCI adapters (one file each)
* XGBoost benchmarking against Random Forest
* SHAP explainability
* Canary deployment signals
* Teams alerts

See `PRODUCTION-PLAN.md` for the path to production readiness.

---

## 19. Quick Start

### Run it locally — no AWS account needed

`run_local.py` starts the real app against an in-process mock of S3 + DynamoDB
(moto), seeds a demo tenant and sample builds, and prints working credentials.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python run_local.py          # → http://127.0.0.1:5000
```

Open the printed dashboard link, or try `/demo` — an interactive scorer that
needs no API key.

> On macOS, AirPlay Receiver occupies port 5000. Use `PORT=5001 python run_local.py`.

### Run the tests

```bash
pip install pytest moto
pytest tests/ -q                # 33 tests, fully mocked, no cloud calls
```

### Run against real AWS

```bash
python bootstrap_aws.py          # create buckets + DynamoDB table
python ml/train_base_model.py    # train the cold-start baseline
python ml/upload_base_model.py   # publish it to S3
gunicorn --workers 2 --bind 0.0.0.0:5000 app.main:app
```

Your CI then calls `/score` — see Integrations above.

---

## Final Vision

SafeShip transforms CI/CD from:

Build -> Test -> Deploy

into:

Build -> Test -> Predict Risk -> Safer Deploy -> Learn -> Improve

---

## Author

Built by Swyam Yadav with guidance from Dr. Naween Kumar.
