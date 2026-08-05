# SafeShip — Path to Production

An honest assessment of what stands between the current codebase and something
you would let a paying team route their deploys through, plus the order to fix it
in.

**Current state, fairly stated:** a working product with a real ML loop, 33
passing tests, a genuinely good interactive demo, and a novel self-closing
learning signal. It is a strong *prototype*. It is not yet a service you can put
someone else's release pipeline behind, for reasons that are almost entirely
operational rather than functional.

The gap is smaller than it looks. Most of Phase 0 is a weekend.

---

## Phase 0 — Stop the bleeding (do before anything else)

Small, concrete, and each one is a thing that will bite you.

### 0.1 Rotate `SECRET_KEY` — this is the urgent one

`app/main.py` fell back to a hardcoded default value for `SECRET_KEY` (the
literal string is deliberately not reproduced here — it is still recoverable from
git history, which is exactly the problem). That key signs Flask session cookies,
so anyone who reads the repository can forge a session for any tenant.

**Status: partially fixed.** The app now refuses to boot when
`SAFESHIP_ENV=production` and `SECRET_KEY` is unset, and generates a random
per-process key outside production. But:

- **The old value remains in git history and cannot be un-published.** Rotation
  is mandatory, not optional.
- Generate one and set it in the EC2/container environment:
  `python -c "import secrets; print(secrets.token_urlsafe(48))"`
- Treat every existing session as compromised. Rotating invalidates them, which
  is the desired outcome.

### 0.2 Commit the work that only exists on one laptop — ✅ DONE

`tests/` and `sentinel/` were untracked. Now committed across nine reviewable
commits, along with the per-build S3 migration.

### 0.3 Make CI run the tests

`.github/workflows/deploy.yml` is checkout → deploy to EC2. **Nothing runs
pytest.** A broken commit ships straight to production.

You built a product whose premise is *"don't deploy blind"* and its own pipeline
deploys blind. Fix it, then go further: have SafeShip's pipeline call SafeShip's
own `/score`. That is simultaneously a real safety gate and the most persuasive
demo you could put in a README.

### 0.4 Stop leaking hashes into logs — ✅ DONE

Failed authentication printed the first 8 characters of both the stored and the
supplied hash. Removed: a truncated credential hash is still a credential hash.
Failed auth is now a single WARNING with the tenant id and nothing else.

### 0.5 Constant-time key comparison

`if stored_hash != given_hash` → `hmac.compare_digest(...)`. The practical timing
signal over a network is tiny, but this is table stakes and any security review
will flag it.

### 0.6 Delete the dead duplicates — ✅ DONE

`app/dynamo_client.py` went from 454 to 146 lines. It carried a 314-line
commented-out copy of itself; a search-and-replace over the live code silently
matched the dead copy first, which is precisely the hazard.

### 0.7 Add `.env.example` and split dev deps — ◐ PARTIAL

`requirements-dev.txt` added, so a clean clone can now run the suite.
**Still outstanding:** `.env.example` documenting every variable.

### 0.8 Pin the Python version

Docker builds on **3.11**, your venv is **3.9.6** (upstream security support
ended October 2025). Dev/prod skew on a language runtime is how "works on my
machine" happens. Add `.python-version`, match Docker, drop 3.9.

**A concrete blocker to clear first.** 28 `.py` files carry a stale header line:

```
Path: C:\deploy-gate\app\routes\score.py
```

`\d` is an invalid escape sequence. Today that is a `DeprecationWarning`; on
**3.12 it becomes a `SyntaxWarning`, and it is slated to become a `SyntaxError`**.
So the upgrade trips over 28 files at once — and the warnings are currently
invisible because `.pyc` caching only emits them when a file is recompiled. The
paths are also simply wrong; this is not a Windows repo.

`app/`, `ml/`, and their vendored copies under `ansible/files/` are all affected.
The fix is deleting the lines — mechanical and docstring-only — but it should land
as its own commit so the diff stays reviewable.

```bash
grep -rln 'Path: C:\\' --include='*.py' .
```

---

## Phase 1 — Make it operable — ✅ DONE

You cannot run a service you cannot see. All six items shipped; verified against
a running instance.

### 1.1 Real logging — ✅ DONE

The "150 print() calls" figure was wrong: most sat inside `__main__` CLI blocks
(correct as prints) or commented-out code. **39** were real, and all are
converted to `app/observability.py` — JSON lines, levels, credential scrubbing,
and a `request_id` that threads an inbound `X-Request-ID` from CI through
scorer → slack → access log.

### 1.2 Error tracking — ✅ DONE

Sentry wired via `SENTRY_DSN`; a no-op when unset, and degrades to a warning if
the package is absent. Unhandled exceptions log a stack and return a generic
body carrying the request id.

### 1.3 A `/health` that means something — ✅ DONE

It currently returns `{"status":"ok"}` unconditionally — it will report healthy
while S3 and DynamoDB are both unreachable. Split it:

- `/health` — liveness, stays trivial
- `/ready` — actually checks S3 + DynamoDB + model loaded

This matters doubly here: your own Sentinel gates on health endpoints, so a
meaningless one undermines your own product's core claim.

### 1.4 Metrics — ✅ DONE

`/metrics` in Prometheus text format: request counters by endpoint and status,
plus a latency histogram. Counters are per-worker under gunicorn — documented,
with the access log as the source of truth for exact totals. Move to
`prometheus_client` multiprocess mode when a real Prometheus arrives.

### 1.5 Rate limiting — ✅ DONE

`flask-limiter`, keyed on a hash of the API key rather than the IP: every build
from one CI provider shares an egress IP, so IP-keying would throttle unrelated
tenants together. `swallow_errors=True` — a customer's build must never fail
because our limiter had a bad day. In-memory storage is per-worker; point
`RATE_LIMIT_STORAGE_URI` at Redis for a shared ceiling.

### 1.6 Security headers + CORS — ◐ MOSTLY DONE

`nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy`, COOP, `Permissions-Policy`
and HSTS-in-production are set; cookies are HttpOnly + SameSite, Secure in
production.

**Still outstanding:** a Content-Security-Policy. The dashboard uses inline
`<script>` for its charts, so a strict CSP needs nonces first — there is a
`TODO(csp)` at the call site. An explicit CORS policy is also still open.

---

## Phase 2 — Make it trustworthy (weeks 3–4)

This is the ML-specific work, and it is what separates "a model in prod" from
"an ML product".

### 2.1 Version the models

Every model writes to a single mutable key:

```
base/model.pkl
tenant_<id>/model.pkl
```

A bad retrain silently overwrites the good model and **there is no way back**.
This is the most serious design issue after `SECRET_KEY`.

- Write immutable, versioned keys: `tenant_<id>/models/2026-08-05T02-00Z.pkl`
- Keep a `current` pointer (S3 object or DynamoDB attribute)
- Enable S3 versioning on the bucket regardless
- Make rollback one pointer update

### 2.2 A real model registry

Store per version: training rows, feature list, hyperparameters, precision,
recall, AUC, the data window, and who/what triggered it. Today
`base_metadata.json` is a single mutable file — you cannot answer "why did
scoring change last Tuesday?"

### 2.3 Fix the train/predict feature skew — ✅ DONE

Training passed a named DataFrame; prediction passed a bare array, so **column
order was the only thing keeping them aligned**, and reordering one feature would
have produced confidently wrong predictions with no error.

`ml/features.py` is now the single `FEATURES` contract, prediction builds a named
DataFrame via `to_frame()`, and `tests/test_features.py` fails if any copy drifts —
including the deliberately isolated ones in the Lambda handlers and
`safeship_ci/contract.py`.

Investigating this surfaced a far larger version of the same problem, which the
work after it addressed: neither reference integration actually *collected* the
features. `recent_failure_rate` and `test_pass_rate` — 52.8% of the model's weight
— were sent as `0.0` and `1.0`, the most reassuring value in each range. The
alignment was fine; the inputs were fiction. See `safeship_ci/` and
`app/imputation.py`.

Still open in the same family:

- `lambda/drift/handler.py` should build its input through `to_frame()` rather
  than its own array.
- Adoption itself causes drift: a tenant whose pipeline starts reporting a real
  `test_pass_rate` after weeks of imputed medians changes distribution under the
  model. That belongs to 2.5.

### 2.4 Shadow mode before enforcing

New tenants should get *scores without gating* for their first N builds, so a
model calibrated on synthetic data cannot block a real release on day one.
Promote to enforcing once tenant-specific precision clears a bar. You already
track `model_phase` — make it mean this.

### 2.5 Close the drift loop

`lambda/drift/handler.py` exists but detection needs to *do* something: alert,
and refuse to promote a retrained model that fails validation. Retraining that
can only ever overwrite is not a safety mechanism.

### 2.6 Calibration, not just accuracy

For a *risk* product the score must mean something. 85% precision says nothing
about whether "70/100" corresponds to a real 70% failure rate. Add a reliability
curve and Brier score; consider isotonic calibration. This is what makes a
customer trust the number enough to block a deploy on it.

---

## Phase 3 — Make it survivable (weeks 5–6)

### 3.1 Infrastructure as code

Ansible configures a box that was created by hand. There is no `terraform/`
(your `.gitignore` optimistically anticipates one). Today the infrastructure
exists only as steps someone remembers.

Terraform for: VPC, EC2/ECS, S3 (versioned, encrypted, lifecycle), DynamoDB
(PITR on), Lambda, ECR, IAM (least privilege), CloudWatch alarms.

### 3.2 Remove the single point of failure

One `t2.micro`, 2 gunicorn workers, in one AZ. If it dies, every customer's
pipeline either blocks or fails open — and *which* of those it does is currently
implicit. ECS Fargate behind an ALB across two AZs is the smallest honest step.

### 3.3 Decide the failure mode, explicitly

**The most important product decision on this page.** If SafeShip is down or slow,
does the customer's pipeline block or proceed?

Answer must be: **fail open, with a timeout**, and say so loudly in the docs. A
risk gate that halts everyone's deploys when *it* breaks will be removed within a
week. `safeship_gate.py` needs a hard timeout and a documented default.

### 3.4 Backup and restore

S3 versioning + DynamoDB PITR, and an actual restore rehearsal. Untested backups
are folklore.

### 3.5 Load test

Establish what 2 workers actually hold. You publish a latency claim; know the
number at which it stops being true.

---

## Phase 4 — Make it a product (weeks 7+)

- **Onboarding**: signup → API key → copy-paste snippet → first score, without
  talking to you
- **Docs site**: quickstart per CI system, API reference, self-hosting guide
- **GitHub App** instead of raw API keys for the GitHub Actions path — far lower
  friction and how competitors do it
- **Audit log**: who scored what, which model version, what verdict. Enterprise
  buyers ask on day one
- **Data policy**: retention, deletion, and a statement that you never receive
  source code — only metrics about it. This is the objection every security team
  will raise, and your architecture already answers it well
- **Billing** if commercial: metered on scores
- Then, and only then: XGBoost, SHAP, GitLab/CircleCI adapters, canary signals

---

## Suggested order

| When | Focus | Why |
|---|---|---|
| ~~Today~~ | ~~0.2~~ ✅ | Work is committed |
| **Now** | **0.1 — rotate `SECRET_KEY` in the live environment** | The old value is still in git history |
| ~~Weeks 1–2~~ | ~~Phase 1~~ ✅ | Done — logging, Sentry, /ready, /metrics, limits, headers |
| This week | 0.3, 0.5, 0.7, 0.8 | CI test gate, compare_digest, .env.example, Python pin |
| Weeks 3–4 | Phase 2 | Model versioning + feature skew are latent data-corruption bugs |
| Weeks 5–6 | Phase 3 | Needed before anyone else depends on it |
| Weeks 7+ | Phase 4 | Growth work, once the base is sound |

---

## Three things not to lose sight of

1. **The self-closing learning loop is the genuinely novel part.** Sentinel's exit
   code auto-labels the training data. Most "learns from outcomes" systems quietly
   depend on humans reporting outcomes, and therefore never learn. Lead with this.

2. **Zero-code-access is a feature, not a footnote.** You only ever see metrics
   *about* a diff, never the diff. That clears the biggest procurement objection
   in this category — say it on the landing page.

3. **The `/demo` page is the best sales asset you have.** Live sliders, instant
   rescoring, no API key. Keep it working through every refactor.

---

## What is genuinely good already

Worth stating, because the list above is all problems:

- Gunicorn, multi-stage Dockerfile, healthcheck in compose, sensible worker count
- API keys hashed (sha256 over 128 bits of entropy — correct choice for keys)
- AWS creds via IAM instance role, never in env or code — done right
- Slack webhooks stored per tenant, so there is no platform-wide Slack secret
- 33 tests fully mocked with `moto` — no cloud calls, 4-second suite
- ECR-for-Lambda was the right call for the 250MB limit, and the reasoning is
  written down
- Append-only per-build storage instead of rewriting a whole CSV — good instinct,
  and the tests prove it
- `run_local.py` gives a zero-credential local environment. Many funded products
  lack this
