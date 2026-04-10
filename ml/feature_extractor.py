"""
feature_extractor.py
Path: C:\deploy-gate\ml\feature_extractor.py

PURPOSE:
  Connects to your Jenkins API and extracts all 10 features for a given build.
  Every feature has a fallback value — so if Jenkins doesn't have the data,
  we use a safe default instead of crashing.

HOW TO RUN (for testing):
  python feature_extractor.py

WHAT IT DOES:
  - Talks to Jenkins REST API
  - Extracts 10 features (diff size, failure rate, time, etc.)
  - Returns a clean dictionary ready to feed into the ML model
"""

import os
import datetime
import requests
from requests.auth import HTTPBasicAuth


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit these 3 lines to match your Jenkins setup
# ─────────────────────────────────────────────────────────────────────────────

JENKINS_URL      = os.getenv("JENKINS_URL",      "http://localhost:8080")
JENKINS_USER     = os.getenv("JENKINS_USER",     "admin")
JENKINS_TOKEN    = os.getenv("JENKINS_TOKEN",    "your-jenkins-api-token")

# ─────────────────────────────────────────────────────────────────────────────


def _jenkins_get(path, fallback=None):
    """
    Makes a GET request to Jenkins REST API.
    Returns parsed JSON if successful, or fallback value if anything goes wrong.
    We NEVER let a network error crash the scoring — we always use fallbacks.
    """
    try:
        url  = f"{JENKINS_URL}/{path}"
        auth = HTTPBasicAuth(JENKINS_USER, JENKINS_TOKEN)
        resp = requests.get(url, auth=auth, timeout=5)

        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"[feature_extractor] Jenkins returned {resp.status_code} for {path}")
            return fallback

    except Exception as e:
        print(f"[feature_extractor] Could not reach Jenkins: {e}")
        return fallback


def get_diff_size(git_diff_output=None, files_changed=5):
    """
    FEATURE 1: diff_size
    How many lines of code changed in this commit.
    
    Best case:  We parse the output of `git diff --stat`
    Fallback:   files_changed * 20 (rough estimate)
    
    In your Jenkins pipeline, add this step BEFORE calling the scoring API:
        env.GIT_DIFF_STAT = sh(script: 'git diff --stat HEAD~1 HEAD | tail -1', returnStdout: true).trim()
    Then pass that string here.
    
    Example git diff output: "5 files changed, 847 insertions(+), 12 deletions(-)"
    """
    if git_diff_output:
        try:
            # Parse "X insertions(+), Y deletions(-)" from git diff --stat output
            parts   = git_diff_output.replace(",", "").split()
            numbers = [int(p) for p in parts if p.isdigit()]
            if len(numbers) >= 2:
                # numbers[0] = files changed, numbers[1] = insertions, numbers[2] = deletions
                total = sum(numbers[1:])  # insertions + deletions
                return max(1, total)
        except Exception:
            pass

    # Fallback: estimate from files changed
    return max(1, files_changed * 20)


def get_files_changed(git_diff_output=None):
    """
    FEATURE 2: files_changed
    How many files were touched in this commit.
    
    Fallback: 5 (reasonable default for a small commit)
    """
    if git_diff_output:
        try:
            parts   = git_diff_output.replace(",", "").split()
            numbers = [int(p) for p in parts if p.isdigit()]
            if numbers:
                return max(1, numbers[0])  # first number is always files changed
        except Exception:
            pass

    return 5  # fallback


def get_time_features():
    """
    FEATURES 3 & 4: hour_of_day and day_of_week
    What time is it right now when the build was triggered.
    
    These NEVER have a fallback problem — time is always available.
    
    Returns:
      hour_of_day  : 0–23  (17 = 5 PM)
      day_of_week  : 0–6   (0=Monday, 4=Friday, 6=Sunday)
    """
    now = datetime.datetime.now()
    return {
        "hour_of_day": now.hour,
        "day_of_week": now.weekday()   # 0=Monday, 6=Sunday
    }


def get_recent_failure_rate(job_name):
    """
    FEATURE 5: recent_failure_rate
    What fraction of the last 10 builds for this job failed.
    
    0.0 = all 10 passed
    1.0 = all 10 failed
    0.4 = 4 of last 10 failed (HIGH RISK signal)
    
    Calls Jenkins API: /job/{job_name}/api/json?tree=builds[result]{0,10}
    Fallback: 0.0 (assume no failures if we can't reach Jenkins)
    """
    data = _jenkins_get(
        f"job/{job_name}/api/json?tree=builds[result]{{0,10}}",
        fallback=None
    )

    if not data or "builds" not in data:
        print(f"[feature_extractor] Could not get build history for {job_name}, using fallback 0.0")
        return 0.0

    builds  = data["builds"]
    if not builds:
        return 0.0

    failures = sum(1 for b in builds if b.get("result") == "FAILURE")
    return round(failures / len(builds), 3)


def get_test_pass_rate(job_name):
    """
    FEATURE 6: test_pass_rate
    What fraction of tests passed in the last build.
    
    1.0  = all tests passing
    0.0  = all tests failing
    0.85 = 85% passing
    
    Calls Jenkins API: /job/{job_name}/lastBuild/testReport/api/json
    Fallback: 1.0 (if no test framework configured, assume tests are fine)
    """
    data = _jenkins_get(
        f"job/{job_name}/lastBuild/testReport/api/json",
        fallback=None
    )

    if not data:
        # No test report = no test framework configured. Default to 1.0 (not penalise them)
        return 1.0

    pass_count  = data.get("passCount",  0)
    fail_count  = data.get("failCount",  0)
    skip_count  = data.get("skipCount",  0)
    total       = pass_count + fail_count + skip_count

    if total == 0:
        return 1.0

    return round(pass_count / total, 3)


def get_is_hotfix(branch_name=""):
    """
    FEATURE 7: is_hotfix
    Is this a hotfix or emergency fix branch?
    
    1 = YES (hotfix, fix, patch, revert branch)
    0 = NO  (normal feature or main branch)
    
    In Jenkins, branch name is available as: env.BRANCH_NAME or env.GIT_BRANCH
    Fallback: 0 (assume not a hotfix)
    """
    if not branch_name:
        return 0

    branch_lower    = branch_name.lower()
    hotfix_keywords = ["hotfix", "hotfix/", "fix/", "patch/", "revert/", "emergency"]

    for keyword in hotfix_keywords:
        if keyword in branch_lower:
            return 1

    return 0


def get_deployer_experience(triggered_by, tenant_data_path=None):
    """
    FEATURE 8: deployer_exp
    How many times has this person deployed before?
    
    High number = experienced deployer (lower risk)
    Low number  = new deployer (slightly higher risk)
    
    We count rows in the tenant's S3 data CSV where triggered_by matches.
    
    Fallback: 1 (assume new deployer if we can't check history)
    """
    if not triggered_by or not tenant_data_path:
        return 1

    try:
        import pandas as pd
        if os.path.exists(tenant_data_path):
            df = pd.read_csv(tenant_data_path)
            if "triggered_by" in df.columns:
                count = len(df[df["triggered_by"] == triggered_by])
                return max(1, count)
    except Exception as e:
        print(f"[feature_extractor] Could not get deployer experience: {e}")

    return 1  # fallback


def get_days_since_deploy(job_name):
    """
    FEATURE 9: days_since_deploy
    How many days since the last SUCCESSFUL build?
    
    0.5 = deployed yesterday (low risk from this feature)
    14  = two weeks since last deploy (higher risk — code drift)
    60  = very long gap (high risk)
    
    Calls Jenkins API: /job/{job_name}/lastSuccessfulBuild/api/json
    Fallback: 7.0 (one week — neutral-ish default)
    """
    data = _jenkins_get(
        f"job/{job_name}/lastSuccessfulBuild/api/json",
        fallback=None
    )

    if not data or "timestamp" not in data:
        return 7.0

    try:
        last_success_ms = data["timestamp"]                          # Jenkins gives milliseconds
        last_success_dt = datetime.datetime.fromtimestamp(last_success_ms / 1000)
        delta           = datetime.datetime.now() - last_success_dt
        days            = round(delta.total_seconds() / 86400, 1)   # convert to days
        return max(0.0, days)
    except Exception:
        return 7.0


def get_build_time_delta(job_name):
    """
    FEATURE 10: build_time_delta
    Is this build taking longer than usual?
    
    0.0   = same as average (normal)
    0.5   = 50% longer than average (something is slow)
    -0.3  = 30% faster than average (might be skipping steps)
    
    Compares current build duration against 14-day average.
    Fallback: 0.0 (assume normal build time)
    """
    # Get last 20 builds to calculate average
    data = _jenkins_get(
        f"job/{job_name}/api/json?tree=builds[duration,timestamp]{{0,20}}",
        fallback=None
    )

    if not data or "builds" not in data:
        return 0.0

    builds = data["builds"]
    if len(builds) < 3:
        return 0.0  # not enough history to calculate average

    try:
        durations = [b["duration"] for b in builds if b.get("duration", 0) > 0]
        if not durations:
            return 0.0

        avg_duration     = sum(durations[1:]) / len(durations[1:])  # exclude current build
        current_duration = durations[0]

        if avg_duration == 0:
            return 0.0

        delta = (current_duration - avg_duration) / avg_duration
        return round(delta, 3)

    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION — this is what gets called from the scoring API
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    job_name,
    branch_name     = "",
    git_diff_output = None,
    triggered_by    = "",
    tenant_data_path= None
):
    """
    Extracts ALL 10 features for a build and returns them as a dictionary.
    
    This is the ONLY function you need to call from outside this file.
    
    Args:
        job_name         : Jenkins job name (e.g. "payments-service")
        branch_name      : Git branch (e.g. "main", "hotfix/payment-fix")
        git_diff_output  : Output of `git diff --stat HEAD~1 HEAD | tail -1`
        triggered_by     : Jenkins username who triggered the build
        tenant_data_path : Local path to tenant data CSV (for deployer_exp)
    
    Returns:
        dict with all 10 features, ready to feed into the ML model
    """
    print(f"\n[feature_extractor] Extracting features for job: {job_name}")

    time_features    = get_time_features()
    files_changed    = get_files_changed(git_diff_output)
    diff_size        = get_diff_size(git_diff_output, files_changed)

    features = {
        "diff_size":            diff_size,
        "files_changed":        files_changed,
        "hour_of_day":          time_features["hour_of_day"],
        "day_of_week":          time_features["day_of_week"],
        "recent_failure_rate":  get_recent_failure_rate(job_name),
        "test_pass_rate":       get_test_pass_rate(job_name),
        "is_hotfix":            get_is_hotfix(branch_name),
        "deployer_exp":         get_deployer_experience(triggered_by, tenant_data_path),
        "days_since_deploy":    get_days_since_deploy(job_name),
        "build_time_delta":     get_build_time_delta(job_name),
    }

    print(f"[feature_extractor] Features extracted successfully:")
    for key, value in features.items():
        print(f"  {key:28s} = {value}")

    return features


# ─────────────────────────────────────────────────────────────────────────────
# TEST — run this file directly to see if it works
# python feature_extractor.py
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING feature_extractor.py")
    print("=" * 60)

    print("\nTest 1: Extracting features WITHOUT Jenkins (all fallbacks)")
    features = extract_features(
        job_name         = "test-job",
        branch_name      = "main",
        git_diff_output  = "5 files changed, 847 insertions(+), 12 deletions(-)",
        triggered_by     = "test-user",
        tenant_data_path = None
    )

    print("\nTest 2: Hotfix branch detection")
    is_hf = get_is_hotfix("hotfix/payment-crash-fix")
    print(f"  hotfix/payment-crash-fix  → is_hotfix = {is_hf}  (expected: 1)")

    is_hf2 = get_is_hotfix("feature/new-login")
    print(f"  feature/new-login         → is_hotfix = {is_hf2}  (expected: 0)")

    print("\nTest 3: Diff size parsing")
    diff1 = get_diff_size("5 files changed, 847 insertions(+), 12 deletions(-)")
    print(f"  847 ins + 12 del          → diff_size = {diff1}  (expected: 859)")

    diff2 = get_diff_size("1 file changed, 3 insertions(+), 1 deletion(-)")
    print(f"  3 ins + 1 del             → diff_size = {diff2}  (expected: 4)")

    print("\nTest 4: Time features (will show current time)")
    tf = get_time_features()
    print(f"  hour_of_day = {tf['hour_of_day']}  (0-23)")
    print(f"  day_of_week = {tf['day_of_week']}  (0=Mon, 4=Fri, 6=Sun)")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("If you see numbers above with no errors — Phase 1, File 1 is DONE.")
    print("=" * 60)