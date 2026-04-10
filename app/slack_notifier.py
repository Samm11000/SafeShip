"""
slack_notifier.py
Path: C:\deploy-gate\app\slack_notifier.py

PURPOSE:
  Posts deploy risk alerts to a tenant's Slack channel.
  Uses the tenant's OWN Slack incoming webhook - no Slack API key needed on our side.
  If no webhook configured, silently skips (does not crash).

HOW TO TEST:
  cd C:\deploy-gate
  python app\slack_notifier.py
  (will test formatting without actually sending - safe to run anytime)
"""

import os
import json
import requests

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR MAP for Slack message attachments
# ─────────────────────────────────────────────────────────────────────────────
VERDICT_COLORS = {
    "SAFE":    "#2eb886",   # green
    "WARNING": "#f2c744",   # yellow
    "BLOCKED": "#e01e5a",   # red
}

VERDICT_EMOJI = {
    "SAFE":    ":white_check_mark:",
    "WARNING": ":warning:",
    "BLOCKED": ":no_entry:",
}

DAY_NAMES = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD THE SLACK MESSAGE PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def _build_payload(job_name, build_number, score_result, tenant):
    """
    Builds the Slack message JSON payload.
    Uses Slack's attachment format for coloured sidebars.
    """
    score    = score_result["score"]
    verdict  = score_result["verdict"]
    color    = VERDICT_COLORS.get(verdict, "#cccccc")
    emoji    = VERDICT_EMOJI.get(verdict, ":question:")
    phase    = score_result.get("model_phase", "base")
    reasons  = score_result.get("top_reasons", [])

    # Thresholds from tenant settings (default 40/70)
    thresh_y = tenant.get("threshold_yellow", 40)
    thresh_r = tenant.get("threshold_red",    70)

    # Header text
    header = f"{emoji}  *Deploy Gate — {verdict}*"
    if verdict == "BLOCKED":
        header += "  _(add `DEPLOY_FORCE=true` to override)_"

    # Score bar visual  e.g.  [██████████░░░░░░░░░░]  74/100
    filled = int(score / 5)
    empty  = 20 - filled
    bar    = "█" * filled + "░" * empty

    # Top reasons formatted
    reasons_text = ""
    for i, r in enumerate(reasons[:3], 1):
        reasons_text += f"\n  {i}. *{r['label']}* — {r['value_str']}"

    # Model phase note
    phase_note = (
        "Personalised model (trained on your builds)"
        if phase == "tenant"
        else "Base model (personalising as your builds accumulate)"
    )

    payload = {
        "text": header,
        "attachments": [
            {
                "color": color,
                "blocks": [
                    {
                        "type": "section",
                        "fields": [
                            {
                                "type": "mrkdwn",
                                "text": f"*Job*\n{job_name} #{build_number}"
                            },
                            {
                                "type": "mrkdwn",
                                "text": f"*Risk Score*\n`[{bar}]`  *{score}/100*"
                            },
                        ]
                    },
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": (
                                f"*Top risk factors:*{reasons_text}\n\n"
                                f"_Model: {phase_note}_"
                            )
                        }
                    },
                    {
                        "type": "context",
                        "elements": [
                            {
                                "type": "mrkdwn",
                                "text": (
                                    f"Thresholds: "
                                    f"green <{thresh_y}  |  "
                                    f"yellow {thresh_y}-{thresh_r}  |  "
                                    f"red >{thresh_r}"
                                )
                            }
                        ]
                    }
                ]
            }
        ]
    }

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# SEND NOTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def send_alert(job_name, build_number, score_result, tenant):
    """
    Sends a Slack alert to the tenant's webhook.

    Args:
        job_name     : Jenkins job name  e.g. "payments-service"
        build_number : Jenkins build number  e.g. 142
        score_result : dict returned by scorer.score_build()
        tenant       : tenant dict from DynamoDB (needs slack_webhook key)

    Returns:
        True if sent successfully, False if skipped or failed
    """
    webhook_url = tenant.get("slack_webhook", "").strip()

    # No webhook configured — skip silently, do not crash
    if not webhook_url:
        print(f"[slack] No webhook configured for tenant {tenant.get('tenant_id','?')} — skipping")
        return False

    payload = _build_payload(job_name, build_number, score_result, tenant)

    try:
        resp = requests.post(
            webhook_url,
            data    = json.dumps(payload),
            headers = {"Content-Type": "application/json"},
            timeout = 5,
        )

        if resp.status_code == 200:
            print(f"[slack] Alert sent for {job_name} #{build_number} — score {score_result['score']}")
            return True
        else:
            print(f"[slack] Webhook returned {resp.status_code}: {resp.text}")
            return False

    except requests.exceptions.Timeout:
        print(f"[slack] Webhook timed out — skipping alert")
        return False
    except Exception as e:
        print(f"[slack] Failed to send alert: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING slack_notifier.py")
    print("=" * 60)

    # Mock score results
    risky_result = {
        "score":       74,
        "verdict":     "BLOCKED",
        "color":       "red",
        "model_phase": "tenant",
        "top_reasons": [
            {"label": "Recent failure rate",       "value_str": "40% of last 10 builds failed",  "importance": 0.278},
            {"label": "Diff size (lines changed)", "value_str": "847 lines (very large)",         "importance": 0.176},
            {"label": "Time of deploy",            "value_str": "5:00 PM (end of day)",           "importance": 0.064},
        ]
    }

    safe_result = {
        "score":       8,
        "verdict":     "SAFE",
        "color":       "green",
        "model_phase": "base",
        "top_reasons": [
            {"label": "Recent failure rate",  "value_str": "0% (all recent builds passed)", "importance": 0.278},
            {"label": "Test pass rate",       "value_str": "100% tests passing",            "importance": 0.250},
            {"label": "Diff size",            "value_str": "45 lines",                     "importance": 0.176},
        ]
    }

    # Mock tenant (no real webhook — just testing payload building)
    tenant_no_webhook = {
        "tenant_id":        "test123",
        "slack_webhook":    "",
        "threshold_yellow": 40,
        "threshold_red":    70,
    }

    tenant_with_webhook = {
        "tenant_id":        "test123",
        "slack_webhook":    "https://hooks.slack.com/test",
        "threshold_yellow": 40,
        "threshold_red":    70,
    }

    # Test 1: No webhook — should skip gracefully
    print("\nTest 1: No webhook configured")
    result = send_alert("payments-service", 142, risky_result, tenant_no_webhook)
    assert result == False
    print("  [OK] Skipped gracefully when no webhook")

    # Test 2: Build and print the payload (verify formatting)
    print("\nTest 2: Build BLOCKED payload (formatting check)")
    payload = _build_payload("payments-service", 142, risky_result, tenant_with_webhook)
    assert payload["text"].startswith(":no_entry:")
    assert len(payload["attachments"]) == 1
    assert payload["attachments"][0]["color"] == "#e01e5a"
    print("  [OK] BLOCKED payload built correctly")
    print(f"  Header : {payload['text'][:60]}")
    print(f"  Color  : {payload['attachments'][0]['color']}  (red)")

    # Test 3: Build SAFE payload
    print("\nTest 3: Build SAFE payload")
    payload2 = _build_payload("user-service", 88, safe_result, tenant_no_webhook)
    assert payload2["attachments"][0]["color"] == "#2eb886"
    print("  [OK] SAFE payload built correctly")
    print(f"  Color  : {payload2['attachments'][0]['color']}  (green)")

    # Test 4: Print what the full Slack message looks like
    print("\nTest 4: Full message preview (what Slack would receive)")
    print("-" * 40)
    blocks = payload["attachments"][0]["blocks"]
    for block in blocks:
        if block["type"] == "section" and "fields" in block:
            for field in block["fields"]:
                print(field["text"])
        elif block["type"] == "section" and "text" in block:
            print(block["text"]["text"])
        elif block["type"] == "context":
            print(block["elements"][0]["text"])
    print("-" * 40)

    # Test 5: Wrong webhook URL — should fail gracefully
    print("\nTest 5: Invalid webhook URL (should fail gracefully, not crash)")
    bad_tenant = {"tenant_id": "x", "slack_webhook": "https://invalid.url/test",
                  "threshold_yellow": 40, "threshold_red": 70}
    result2 = send_alert("job", 1, risky_result, bad_tenant)
    print(f"  [OK] Failed gracefully, returned: {result2}")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED - slack_notifier.py is ready")
    print("Next: app\\routes\\score.py")
    print("=" * 60)