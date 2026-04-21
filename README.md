# 🚢 SafeShip

## AI-Powered Deployment Risk Intelligence Platform

SafeShip predicts whether a software release is likely to fail **before it reaches production**. It integrates with CI/CD pipelines such as Jenkins, scores deployments in real time, continuously learns from outcomes, and retrains personalized models for each team.

---

## Table of Contents

1. Problem Statement
2. How SafeShip Works
3. System Architecture
4. Core Features
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

1. Jenkins pipeline calls `/score`
2. SafeShip extracts deployment signals
3. ML model predicts risk probability
4. Returns score (0–100)
5. Pipeline can allow / warn / block
6. After deployment, `/log` records real outcome
7. Nightly retraining improves future predictions

---

## 3. System Architecture

* **Frontend Dashboard**: tenant analytics, setup guide, charts
* **Flask API**: scoring + logging + auth
* **S3**: datasets, model files, backups
* **DynamoDB**: tenant metadata
* **Lambda**: scheduled retraining jobs
* **ECR**: stores retraining container image
* **Jenkins**: CI/CD integration source

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
  n- hotfixes riskier
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

* GitHub Actions integration
* GitLab CI support
* Slack / Teams alerts
* XGBoost benchmarking
* SHAP explainability
* Canary deployment signals
* Kubernetes rollback detection

---

## 19. Quick Start

```bash
# Run API
python main.py

# Retrain manually
python retrain.py
```

Jenkins users can paste the generated SafeShip stage into existing Jenkinsfiles.

---

## Final Vision

SafeShip transforms CI/CD from:

Build -> Test -> Deploy

into:

Build -> Test -> Predict Risk -> Safer Deploy -> Learn -> Improve

---

## Author

Built by Swyam Yadav with guidance from Dr. Naween Kumar.
