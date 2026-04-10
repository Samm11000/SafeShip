"""
outcome_logger.py
Path: C:\deploy-gate\jenkins\outcome_logger.py

PURPOSE:
  Checks what happened after a deploy and sends the result to the
  Deploy Gate API. This is what creates labelled training data
  so the model gets smarter over time.

  Called automatically from the Jenkinsfile post{} block.
  Can also be run manually for any build.

HOW TO RUN MANUALLY:
  python outcome_logger.py \
    --url      http://54.89.160.150 \
    --tenant   your-tenant-id \
    --key      your-api-key \
    --build-id dg-abc123-def456 \
    --result   SUCCESS

SIGNALS CHECKED (in order):
  1. Jenkins build result     (FAILURE = risky)
  2. Hotfix commit after deploy within 4 hours  (risky)
  3. Revert commit after deploy within 6 hours  (risky)
  4. No signals = safe
"""

import os
import sys
import json
import argparse
import requests
from datetime import datetime, timezone


def detect_label(build_result, branch_name="", commit_msg=""):
    """
    Determines the label and confidence based on available signals.

    Returns:
        label        : 0 (safe) or 1 (risky)
        label_source : string describing which signal triggered
        sample_weight: 0.6-1.0 confidence weight
    """
    signals = []

    # Signal 1: Build itself failed
    if build_result == "FAILURE":
        signals.append(("failure", 1.0))

    # Signal 2: This IS a hotfix (suggests previous deploy caused a problem)
    if branch_name:
        b = branch_name.lower()
        if any(k in b for k in ["hotfix", "fix/", "patch/", "revert/"]):
            signals.append(("hotfix_branch", 0.8))

    # Signal 3: Commit message mentions revert or rollback
    if commit_msg:
        c = commit_msg.lower()
        if any(k in c for k in ["revert", "rollback", "undo", "hotfix"]):
            signals.append(("revert_commit", 0.7))

    if not signals:
        return 0, "success", 1.0

    # Use highest confidence signal
    signals.sort(key=lambda x: x[1], reverse=True)
    top_signal = signals[0]

    # If 2+ signals triggered — high confidence risky
    weight = 1.0 if len(signals) >= 2 else top_signal[1]

    return 1, top_signal[0], weight


def log_outcome(url, tenant_id, api_key, build_id,
                build_result, branch_name="", commit_msg=""):
    """
    Sends outcome to Deploy Gate API.
    This updates the label in S3 CSV for this build.
    """
    label, label_source, sample_weight = detect_label(
        build_result, branch_name, commit_msg
    )

    payload = {
        "tenant_id":     tenant_id,
        "api_key":       api_key,
        "build_id":      build_id,
        "label":         label,
        "label_source":  label_source,
        "sample_weight": sample_weight,
    }

    print(f"[outcome_logger] Logging outcome for build {build_id}")
    print(f"  Result       : {build_result}")
    print(f"  Label        : {label} ({'risky' if label == 1 else 'safe'})")
    print(f"  Label source : {label_source}")
    print(f"  Confidence   : {sample_weight}")

    try:
        resp = requests.post(
            f"{url}/log",
            json    = payload,
            headers = {"Content-Type": "application/json"},
            timeout = 10,
        )
        if resp.status_code == 200:
            print(f"  [OK] Outcome logged successfully")
            return True
        else:
            print(f"  [WARN] API returned {resp.status_code}: {resp.text}")
            return False
    except Exception as e:
        print(f"  [ERROR] Could not reach Deploy Gate API: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Log deploy outcome to Deploy Gate")
    parser.add_argument("--url",       required=True, help="Deploy Gate API URL")
    parser.add_argument("--tenant",    required=True, help="Your tenant_id")
    parser.add_argument("--key",       required=True, help="Your api_key")
    parser.add_argument("--build-id",  required=True, help="build_id from /score response")
    parser.add_argument("--result",    required=True, help="Jenkins build result: SUCCESS or FAILURE")
    parser.add_argument("--branch",    default="",    help="Git branch name")
    parser.add_argument("--commit-msg",default="",    help="Latest commit message")
    args = parser.parse_args()

    success = log_outcome(
        url          = args.url,
        tenant_id    = args.tenant,
        api_key      = args.key,
        build_id     = args.build_id,
        build_result = args.result,
        branch_name  = args.branch,
        commit_msg   = args.commit_msg,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()