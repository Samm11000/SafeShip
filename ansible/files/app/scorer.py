"""
scorer.py
Path: C:\deploy-gate\app\scorer.py

PURPOSE:
  Loads the correct model for a tenant from S3 (or cache).
  Runs prediction and returns score 0-100 + top 3 risk reasons.
  Hot-reloads model every 5 minutes without restarting Flask.

HOW TO TEST:
  cd C:\deploy-gate
  python app\scorer.py
"""

import os
import time
import joblib
import boto3
import tempfile
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
S3_BUCKET       = os.getenv("S3_MODELS_BUCKET", "deploy-gate-models")
AWS_REGION      = os.getenv("AWS_REGION",        "ap-south-1")
CACHE_TTL_SECS  = 300   # reload model from S3 every 5 minutes

# Must match exact order from validator.py to_model_input() and train_base_model.py
FEATURE_COLUMNS = [
    "diff_size",
    "files_changed",
    "hour_of_day",
    "day_of_week",
    "recent_failure_rate",
    "test_pass_rate",
    "is_hotfix",
    "deployer_exp",
    "days_since_deploy",
    "build_time_delta",
]

# Human-readable explanation for each feature (shown in Slack alert)
FEATURE_LABELS = {
    "diff_size":            "Diff size (lines changed)",
    "files_changed":        "Files changed",
    "hour_of_day":          "Time of deploy",
    "day_of_week":          "Day of week",
    "recent_failure_rate":  "Recent failure rate",
    "test_pass_rate":       "Test pass rate",
    "is_hotfix":            "Hotfix branch",
    "deployer_exp":         "Deployer experience",
    "days_since_deploy":    "Days since last deploy",
    "build_time_delta":     "Build time delta",
}

# Human-readable value formatting for Slack messages
def _format_value(feature, value):
    if feature == "diff_size":
        if value > 500: return f"{value} lines (very large)"
        if value > 200: return f"{value} lines (large)"
        return f"{value} lines"
    if feature == "hour_of_day":
        suffix = "AM" if value < 12 else "PM"
        hour12 = value if value <= 12 else value - 12
        if hour12 == 0: hour12 = 12
        risk = " (end of day)" if 16 <= value <= 19 else ""
        return f"{hour12}:00 {suffix}{risk}"
    if feature == "day_of_week":
        days = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        name = days[int(value)] if 0 <= int(value) <= 6 else str(value)
        risk = " (risky!)" if int(value) >= 4 else ""
        return f"{name}{risk}"
    if feature == "recent_failure_rate":
        pct = round(value * 100)
        if pct == 0: return "0% (all recent builds passed)"
        return f"{pct}% of last 10 builds failed"
    if feature == "test_pass_rate":
        return f"{round(value * 100)}% tests passing"
    if feature == "is_hotfix":
        return "Yes (hotfix branch)" if value == 1 else "No"
    if feature == "deployer_exp":
        return f"{value} past deploys"
    if feature == "days_since_deploy":
        return f"{value} days"
    if feature == "build_time_delta":
        pct = round(value * 100)
        if pct > 0:  return f"+{pct}% slower than average"
        if pct < 0:  return f"{pct}% faster than average"
        return "normal"
    return str(value)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL CACHE
# Keeps models in memory so we don't hit S3 on every single request.
# Refreshes every 5 minutes to pick up newly trained tenant models.
# ─────────────────────────────────────────────────────────────────────────────

class ModelCache:
    def __init__(self):
        self._models     = {}   # tenant_id -> model object
        self._timestamps = {}   # tenant_id -> last loaded time
        self._s3         = boto3.client("s3", region_name=AWS_REGION)

    def _s3_key_for(self, tenant_id):
        """Returns the S3 key to load for a given tenant."""
        if tenant_id == "base" or tenant_id == "demo":
            return "base/model.pkl"
        return f"tenant_{tenant_id}/model.pkl"

    def _base_key(self):
        return "base/model.pkl"

    def _model_exists_in_s3(self, key):
        """Check if a model file exists in S3 without downloading it."""
        try:
            self._s3.head_object(Bucket=S3_BUCKET, Key=key)
            return True
        except Exception:
            return False

    def _download_model(self, s3_key):
        """Downloads model from S3 to a temp file and loads it."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            self._s3.download_file(S3_BUCKET, s3_key, tmp_path)
            model = joblib.load(tmp_path)
            return model
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_model(self, tenant_id):
        """
        Returns the best model for this tenant.
        Logic:
          1. If tenant has their own model in S3 -> use it
          2. Otherwise -> use base model
          3. Cache in memory for CACHE_TTL_SECS seconds
        """
        now     = time.time()
        cache_key = tenant_id

        # Check if cached model is still fresh
        last_loaded = self._timestamps.get(cache_key, 0)
        if cache_key in self._models and (now - last_loaded) < CACHE_TTL_SECS:
            return self._models[cache_key], self._get_phase(tenant_id)

        # Cache miss or expired — reload from S3
        print(f"[scorer] Loading model for tenant: {tenant_id}")

        tenant_key = self._s3_key_for(tenant_id)
        base_key   = self._base_key()

        # Try tenant model first, fall back to base
        if tenant_id not in ("base", "demo") and self._model_exists_in_s3(tenant_key):
            model = self._download_model(tenant_key)
            phase = "tenant"
            print(f"[scorer]   Loaded tenant model from s3://{S3_BUCKET}/{tenant_key}")
        else:
            model = self._download_model(base_key)
            phase = "base"
            print(f"[scorer]   Loaded base model from s3://{S3_BUCKET}/{base_key}")

        self._models[cache_key]     = model
        self._timestamps[cache_key] = now
        self._phase = {tenant_id: phase}

        return model, phase

    def _get_phase(self, tenant_id):
        return getattr(self, "_phase", {}).get(tenant_id, "base")

    def invalidate(self, tenant_id):
        """Force reload on next request (called after retrain)."""
        self._models.pop(tenant_id, None)
        self._timestamps.pop(tenant_id, None)


# Singleton cache — shared across all Flask requests
_cache = ModelCache()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def score_build(features_list, tenant_id="base"):
    """
    Takes a list of 10 feature values and returns a score dict.

    Args:
        features_list : list of 10 values in FEATURE_COLUMNS order
        tenant_id     : used to load correct model from cache

    Returns:
        {
            "score":       74,          # 0-100
            "verdict":     "BLOCKED",   # SAFE / WARNING / BLOCKED
            "color":       "red",       # green / yellow / red
            "model_phase": "tenant",    # base / tenant
            "top_reasons": [
                {"feature": "recent_failure_rate", "label": "Recent failure rate",
                 "importance": 0.278, "value": 0.4, "value_str": "40% of last 10 builds failed"},
                ...
            ]
        }
    """
    model, phase = _cache.get_model(tenant_id)

    # Build input array — shape (1, 10)
    X = np.array(features_list).reshape(1, -1)

    # predict_proba returns [[P(safe), P(risky)]]
    proba       = model.predict_proba(X)[0]
    risky_proba = float(proba[1])
    score       = int(round(risky_proba * 100))

    # Verdict based on thresholds
    if score <= 40:
        verdict = "SAFE"
        color   = "green"
    elif score <= 70:
        verdict = "WARNING"
        color   = "yellow"
    else:
        verdict = "BLOCKED"
        color   = "red"

    # Top 3 risk reasons using feature importances
    importances = model.feature_importances_
    reasons = []
    for i, (feat, imp) in enumerate(zip(FEATURE_COLUMNS, importances)):
        val = features_list[i]
        reasons.append({
            "feature":   feat,
            "label":     FEATURE_LABELS[feat],
            "importance": round(float(imp), 3),
            "value":     val,
            "value_str": _format_value(feat, val),
        })

    # Sort by importance descending, take top 3
    top_reasons = sorted(reasons, key=lambda x: x["importance"], reverse=True)[:3]

    return {
        "score":       score,
        "verdict":     verdict,
        "color":       color,
        "model_phase": phase,
        "top_reasons": top_reasons,
    }


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING scorer.py")
    print("=" * 60)

    # Test 1: Risky deploy (Friday 5pm, large diff, high failure rate)
    print("\nTest 1: Classic risky deploy")
    print("  Friday 5pm, 847 lines changed, 40% recent failure rate")
    risky_features = [
        847,   # diff_size
        12,    # files_changed
        17,    # hour_of_day  (5pm)
        4,     # day_of_week  (Friday)
        0.4,   # recent_failure_rate (40% failures)
        0.85,  # test_pass_rate
        0,     # is_hotfix
        10,    # deployer_exp
        3.0,   # days_since_deploy
        0.1,   # build_time_delta
    ]
    result1 = score_build(risky_features, tenant_id="base")
    print(f"  Score   : {result1['score']} / 100")
    print(f"  Verdict : {result1['verdict']}")
    print(f"  Color   : {result1['color']}")
    print(f"  Phase   : {result1['model_phase']}")
    print(f"  Top reasons:")
    for r in result1["top_reasons"]:
        print(f"    {r['label']:30s} = {r['value_str']}")

    # Test 2: Safe deploy (Tuesday morning, small diff, all tests passing)
    print("\nTest 2: Safe deploy")
    print("  Tuesday 10am, 45 lines changed, 0% recent failure rate")
    safe_features = [
        45,    # diff_size
        3,     # files_changed
        10,    # hour_of_day  (10am)
        1,     # day_of_week  (Tuesday)
        0.0,   # recent_failure_rate (no failures)
        1.0,   # test_pass_rate (all passing)
        0,     # is_hotfix
        50,    # deployer_exp (experienced)
        1.0,   # days_since_deploy
        -0.05, # build_time_delta (slightly faster)
    ]
    result2 = score_build(safe_features, tenant_id="base")
    print(f"  Score   : {result2['score']} / 100")
    print(f"  Verdict : {result2['verdict']}")
    print(f"  Color   : {result2['color']}")

    # Test 3: Hotfix deploy
    print("\nTest 3: Hotfix deploy")
    hotfix_features = [
        120,   # diff_size
        5,     # files_changed
        15,    # hour_of_day (3pm)
        2,     # day_of_week (Wednesday)
        0.2,   # recent_failure_rate
        0.90,  # test_pass_rate
        1,     # is_hotfix = YES
        25,    # deployer_exp
        0.5,   # days_since_deploy
        0.0,   # build_time_delta
    ]
    result3 = score_build(hotfix_features, tenant_id="base")
    print(f"  Score   : {result3['score']} / 100")
    print(f"  Verdict : {result3['verdict']}")

    # Verify scoring logic makes sense
    print("\n--- SANITY CHECK ---")
    assert result1["score"] > result2["score"], \
        f"FAIL: Risky deploy ({result1['score']}) should score higher than safe ({result2['score']})"
    print(f"  [OK] Risky score ({result1['score']}) > Safe score ({result2['score']})")

    assert result1["verdict"] in ["WARNING", "BLOCKED"], \
        "FAIL: Risky deploy should be WARNING or BLOCKED"
    print(f"  [OK] Risky deploy verdict is {result1['verdict']}")

    assert result2["verdict"] == "SAFE", \
        f"FAIL: Safe deploy should be SAFE, got {result2['verdict']}"
    print(f"  [OK] Safe deploy verdict is SAFE")

    assert len(result1["top_reasons"]) == 3, "FAIL: Should return exactly 3 reasons"
    print(f"  [OK] Returns exactly 3 top reasons")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - scorer.py is ready")
    print("Next: evaluate.py (File 3 of Phase 2)")
    print("=" * 60)