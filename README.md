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
4a. Integrations — GitHub Actions, Jenkins, anything
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

Because `/score` and `/log` are ordinary JSON endpoints, integration is a request
— not a plugin.

### GitHub Actions (reference implementation)

`safeship-demo-app` runs the complete loop on GitHub Actions:

```yaml
- name: Pre-deploy risk gate (SafeShip /score)
  run: python3 safeship_gate.py          # exits non-zero if BLOCKED

- name: Deploy
  run: ./deploy.sh

- name: Post-deploy watch (SafeShip Sentinel)
  id: watch
  run: python3 safeship_sentinel.py --url $URL/health --window 120

- name: Log outcome (learning loop)
  if: always()
  run: python3 safeship_log.py ${{ steps.watch.outcome == 'success' && '0' || '1' }}
```

### Jenkins

```groovy
stage('SafeShip Gate') {
    steps { sh 'python3 safeship_gate.py' }
}
```

### Anything else

```bash
curl -sX POST "$SAFESHIP_URL/score" -H 'Content-Type: application/json' \
  -d '{"tenant_id":"...","api_key":"...","diff_size":120,"files_changed":4, ...}'
```

Only two components are Jenkins-specific, and neither is required:
`ml/feature_extractor.py` (optionally enriches signals from the Jenkins API) and
`jenkins/outcome_logger.py`.

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

Exit code `0` = HEALTHY, `1` = DEGRADED — so any CI can gate or roll back on it,
and that same exit code is what auto-labels the build for retraining.

---

## 5. Machine Learning Model

Current production model uses `RandomForestClassifier`.

### Input Features

1. diff_size
2. files_changed
3. hour_of_day
4. day_of_week
5. recent_failure_rate
6. test_pass_rate
7. is_hotfix
8. deployer_exp
9. days_since_deploy
10. build_time_delta

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

No public labelled deployment-failure dataset exists.

### Phase 1: Synthetic Bootstrap Data

Used to solve cold start.
Generated scenarios encode:

* large changes riskier
* hotfixes riskier
* low test pass rate riskier
* unstable pipelines riskier

### Phase 2: Real Tenant Data

Once users integrate Jenkins:

* real builds collected
* real outcomes labelled
* tenant models become more accurate

---

## 9. Model Retraining Lifecycle

### Schedule

Daily scheduled retraining (e.g. 2 AM UTC).

### Steps

1. Trigger Lambda
2. Pull tenant CSV data from S3
3. Check minimum labelled rows
4. Split train/test
5. Apply SMOTE to training set only
6. Train new model
7. Validate metrics
8. If passed, replace production model in S3
9. API hot reloads latest model

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

These four endpoints are the entire integration surface — there is no SDK or
plugin to install.

### POST /score

Returns deployment risk score.

### POST /log

Logs final deployment outcome.

### GET /dashboard

Tenant analytics UI.

### GET /health

Health check.

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

Example internal metrics:

* Precision ~85%
* Recall ~86%
* AUC-ROC ~0.93
* Latency <200ms warm cache

---

## 18. Roadmap

Shipped:

* ✅ GitHub Actions integration (reference pipeline in `safeship-demo-app`)
* ✅ Slack alerts (per-tenant incoming webhook)
* ✅ Post-deploy watchdog + rollback trigger (Sentinel)
* ✅ Append-only per-build storage (dashboard cost no longer grows with history)

Next:

* GitLab CI / CircleCI reference pipelines
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
