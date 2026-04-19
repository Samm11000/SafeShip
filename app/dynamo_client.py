# """
# dynamo_client.py
# Path: C:\deploy-gate\app\dynamo_client.py

# PURPOSE:
#   All DynamoDB reads and writes in one place.
#   Handles tenant creation, lookup, and build count updates.

# HOW TO TEST:
#   cd C:\deploy-gate
#   python app\dynamo_client.py
# """

# import os
# import uuid
# import hashlib
# import boto3
# from datetime import datetime, timezone
# from boto3.dynamodb.conditions import Key
# from decimal import Decimal

# # ─────────────────────────────────────────────────────────────────────────────
# # CONFIG
# # ─────────────────────────────────────────────────────────────────────────────
# AWS_REGION   = os.getenv("AWS_REGION",    "ap-south-1")
# DYNAMO_TABLE = os.getenv("DYNAMO_TABLE",  "tenants")


# def _table():
#     """Returns DynamoDB table resource."""
#     dynamo = boto3.resource("dynamodb", region_name=AWS_REGION)
#     return dynamo.Table(DYNAMO_TABLE)


# def _hash_key(api_key: str) -> str:
#     """SHA-256 hash the API key before storing."""
#     return hashlib.sha256(api_key.encode()).hexdigest()


# def _now_iso() -> str:
#     return datetime.now(timezone.utc).isoformat()


# # ─────────────────────────────────────────────────────────────────────────────
# # CREATE TENANT
# # ─────────────────────────────────────────────────────────────────────────────

# def create_tenant(email: str = "") -> dict:
#     """
#     Creates a new tenant record in DynamoDB.
#     Returns the tenant_id and plain-text api_key (only time we return plain key).
#     """
#     tenant_id   = str(uuid.uuid4()).replace("-", "")[:16]  # short UUID
#     api_key     = str(uuid.uuid4()).replace("-", "")       # full UUID as key
#     hashed_key  = _hash_key(api_key)

#     item = {
#         "tenant_id":        tenant_id,
#         "api_key_hash":     hashed_key,
#         "email":            email,
#         "created_at":       _now_iso(),
#         "build_count":      0,
#         "labelled_count":   0,
#         "model_phase":      "base",
#         "last_retrain":     "",
#         "model_precision":  Decimal("0.851"),  # base model precision
#         "slack_webhook":    "",
#         "drift_alert_sent": False,
#         "threshold_red":    70,
#         "threshold_yellow": 40,
#     }

#     _table().put_item(Item=item)
#     print(f"[dynamo] Created tenant: {tenant_id}")

#     return {
#         "tenant_id": tenant_id,
#         "api_key":   api_key,   # return plain key ONCE only
#     }


# # ─────────────────────────────────────────────────────────────────────────────
# # VALIDATE API KEY
# # ─────────────────────────────────────────────────────────────────────────────

# def validate_tenant(tenant_id: str, api_key: str) -> dict | None:
#     """
#     Checks tenant exists and API key is correct.
#     Returns tenant record if valid, None if invalid.
#     """
#     # Special case: demo tenant always valid
#     if tenant_id == "demo":
#         return {
#             "tenant_id":      "demo",
#             "model_phase":    "base",
#             "threshold_red":  70,
#             "threshold_yellow": 40,
#             "slack_webhook":  "",
#         }

#     try:
#         resp = _table().get_item(Key={"tenant_id": tenant_id})
#         item = resp.get("Item")

#         if not item:
#             print(f"[dynamo] Tenant not found: {tenant_id}")
#             return None

#         # Verify hashed key
#         if item.get("api_key_hash") != _hash_key(api_key):
#             print(f"[dynamo] Invalid API key for tenant: {tenant_id}")
#             return None

#         return item

#     except Exception as e:
#         print(f"[dynamo] Error validating tenant: {e}")
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # GET TENANT
# # ─────────────────────────────────────────────────────────────────────────────

# def get_tenant(tenant_id: str) -> dict | None:
#     """Gets tenant record by ID (no key check — internal use only)."""
#     try:
#         resp = _table().get_item(Key={"tenant_id": tenant_id})
#         return resp.get("Item")
#     except Exception as e:
#         print(f"[dynamo] Error getting tenant {tenant_id}: {e}")
#         return None


# # ─────────────────────────────────────────────────────────────────────────────
# # INCREMENT BUILD COUNT
# # ─────────────────────────────────────────────────────────────────────────────

# def increment_build_count(tenant_id: str) -> int:
#     """
#     Atomically increments build_count by 1.
#     Returns the new count.
#     """
#     if tenant_id == "demo":
#         return 0

#     try:
#         resp = _table().update_item(
#             Key={"tenant_id": tenant_id},
#             UpdateExpression="SET build_count = build_count + :inc",
#             ExpressionAttributeValues={":inc": 1},
#             ReturnValues="UPDATED_NEW",
#         )
#         new_count = int(resp["Attributes"]["build_count"])
#         return new_count
#     except Exception as e:
#         print(f"[dynamo] Error incrementing build count: {e}")
#         return 0


# # ─────────────────────────────────────────────────────────────────────────────
# # INCREMENT LABELLED COUNT
# # ─────────────────────────────────────────────────────────────────────────────

# def increment_labelled_count(tenant_id: str) -> int:
#     """Increments labelled_count when outcome logger assigns a label."""
#     if tenant_id == "demo":
#         return 0
#     try:
#         resp = _table().update_item(
#             Key={"tenant_id": tenant_id},
#             UpdateExpression="SET labelled_count = labelled_count + :inc",
#             ExpressionAttributeValues={":inc": 1},
#             ReturnValues="UPDATED_NEW",
#         )
#         return int(resp["Attributes"]["labelled_count"])
#     except Exception as e:
#         print(f"[dynamo] Error incrementing labelled count: {e}")
#         return 0


# # ─────────────────────────────────────────────────────────────────────────────
# # UPDATE SLACK WEBHOOK
# # ─────────────────────────────────────────────────────────────────────────────

# def update_slack_webhook(tenant_id: str, webhook_url: str):
#     """Saves tenant's Slack webhook URL."""
#     try:
#         _table().update_item(
#             Key={"tenant_id": tenant_id},
#             UpdateExpression="SET slack_webhook = :w",
#             ExpressionAttributeValues={":w": webhook_url},
#         )
#         print(f"[dynamo] Slack webhook updated for {tenant_id}")
#     except Exception as e:
#         print(f"[dynamo] Error updating webhook: {e}")


# # ─────────────────────────────────────────────────────────────────────────────
# # UPDATE MODEL METADATA (called by Lambda after retrain)
# # ─────────────────────────────────────────────────────────────────────────────

# def update_model_metadata(tenant_id: str, phase: str, precision: float):
#     """Updates model_phase, model_precision, last_retrain after successful swap."""
#     try:
#         _table().update_item(
#             Key={"tenant_id": tenant_id},
#             UpdateExpression=(
#                 "SET model_phase = :phase, "
#                 "model_precision = :prec, "
#                 "last_retrain = :ts"
#             ),
#             ExpressionAttributeValues={
#                 ":phase": phase,
#                 ":prec":  Decimal(str(round(float(precision), 4))),
#                 ":ts":    _now_iso(),
#             },
#         )
#         print(f"[dynamo] Model metadata updated for {tenant_id}: {phase} / {precision:.3f}")
#     except Exception as e:
#         print(f"[dynamo] Error updating model metadata: {e}")


# # ─────────────────────────────────────────────────────────────────────────────
# # UPDATE THRESHOLDS
# # ─────────────────────────────────────────────────────────────────────────────

# def update_thresholds(tenant_id: str, yellow: int, red: int):
#     """Saves custom score thresholds from dashboard settings."""
#     try:
#         _table().update_item(
#             Key={"tenant_id": tenant_id},
#             UpdateExpression=(
#                 "SET threshold_yellow = :y, threshold_red = :r"
#             ),
#             ExpressionAttributeValues={":y": yellow, ":r": red},
#         )
#         print(f"[dynamo] Thresholds updated for {tenant_id}: yellow={yellow}, red={red}")
#     except Exception as e:
#         print(f"[dynamo] Error updating thresholds: {e}")


# # ─────────────────────────────────────────────────────────────────────────────
# # TEST
# # ─────────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     print("=" * 60)
#     print("TESTING dynamo_client.py")
#     print("=" * 60)

#     # Test 1: Create a tenant
#     print("\nTest 1: Create tenant")
#     result = create_tenant(email="test@example.com")
#     tenant_id = result["tenant_id"]
#     api_key   = result["api_key"]
#     print(f"  [OK] tenant_id = {tenant_id}")
#     print(f"  [OK] api_key   = {api_key[:8]}...  (truncated for security)")

#     # Test 2: Validate with correct key
#     print("\nTest 2: Validate with correct API key")
#     tenant = validate_tenant(tenant_id, api_key)
#     assert tenant is not None, "FAIL: Should find tenant"
#     assert tenant["tenant_id"] == tenant_id
#     print(f"  [OK] Tenant validated successfully")
#     print(f"  [OK] model_phase   = {tenant['model_phase']}")
#     print(f"  [OK] build_count   = {tenant['build_count']}")

#     # Test 3: Validate with wrong key
#     print("\nTest 3: Validate with WRONG API key")
#     bad = validate_tenant(tenant_id, "wrongkey123")
#     assert bad is None, "FAIL: Should reject wrong key"
#     print(f"  [OK] Correctly rejected wrong API key")

#     # Test 4: Increment build count
#     print("\nTest 4: Increment build count")
#     count1 = increment_build_count(tenant_id)
#     count2 = increment_build_count(tenant_id)
#     count3 = increment_build_count(tenant_id)
#     assert count3 == 3, f"FAIL: Expected 3, got {count3}"
#     print(f"  [OK] Build count incremented to {count3}")

#     # Test 5: Update Slack webhook
#     print("\nTest 5: Update Slack webhook")
#     update_slack_webhook(tenant_id, "https://hooks.slack.com/test/webhook")
#     updated = get_tenant(tenant_id)
#     assert updated["slack_webhook"] == "https://hooks.slack.com/test/webhook"
#     print(f"  [OK] Slack webhook saved")

#     # Test 6: Update model metadata
#     print("\nTest 6: Update model metadata (simulate retrain)")
#     update_model_metadata(tenant_id, "tenant", 0.823)
#     updated2 = get_tenant(tenant_id)
#     assert updated2["model_phase"] == "tenant"
#     assert float(updated2["model_precision"]) == 0.823
#     print(f"  [OK] model_phase updated to: {updated2['model_phase']}")
#     print(f"  [OK] model_precision updated to: {updated2['model_precision']}")

#     # Test 7: Demo tenant always valid
#     print("\nTest 7: Demo tenant")
#     demo = validate_tenant("demo", "any-key")
#     assert demo is not None
#     print(f"  [OK] Demo tenant always valid")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED - dynamo_client.py is ready")
#     print("Next: slack_notifier.py")
#     print("=" * 60)


"""
dynamo_client.py - Fixed version with proper auth + email lookup
"""
import os, uuid, hashlib, boto3
from datetime import datetime, timezone
from decimal import Decimal

AWS_REGION   = os.getenv("AWS_REGION",   "ap-south-1")
DYNAMO_TABLE = os.getenv("DYNAMO_TABLE", "tenants")

def _table():
    return boto3.resource("dynamodb", region_name=AWS_REGION).Table(DYNAMO_TABLE)

def _hash(key): return hashlib.sha256(key.encode()).hexdigest()
def _now():     return datetime.now(timezone.utc).isoformat()

def create_tenant(email=""):
    tenant_id = str(uuid.uuid4()).replace("-","")[:16]
    api_key   = str(uuid.uuid4()).replace("-","")
    item = {
        "tenant_id":        tenant_id,
        "api_key_hash":     _hash(api_key),
        "api_key_preview":  api_key[:8] + "...",  # show first 8 chars for reference
        "email":            email.strip().lower(),
        "created_at":       _now(),
        "build_count":      0,
        "labelled_count":   0,
        "model_phase":      "base",
        "last_retrain":     "",
        "model_precision":  Decimal("0.851"),
        "slack_webhook":    "",
        "drift_alert_sent": False,
        "threshold_red":    70,
        "threshold_yellow": 40,
    }
    _table().put_item(Item=item)
    print(f"[dynamo] Created tenant: {tenant_id}")
    return {"tenant_id": tenant_id, "api_key": api_key}

def validate_tenant(tenant_id, api_key):
    """Validates tenant_id + api_key combo."""
    if tenant_id == "demo":
        return {"tenant_id":"demo","model_phase":"base",
                "threshold_red":70,"threshold_yellow":40,"slack_webhook":""}
    if not tenant_id or not api_key:
        return None
    try:
        resp = _table().get_item(Key={"tenant_id": tenant_id})
        item = resp.get("Item")
        if not item:
            print(f"[dynamo] Tenant not found: {tenant_id}")
            return None
        stored_hash = item.get("api_key_hash","")
        given_hash  = _hash(api_key.strip())
        if stored_hash != given_hash:
            print(f"[dynamo] Wrong key for tenant: {tenant_id}")
            print(f"[dynamo] Expected hash prefix: {stored_hash[:8]}")
            print(f"[dynamo] Given hash prefix:    {given_hash[:8]}")
            return None
        return item
    except Exception as e:
        print(f"[dynamo] validate_tenant error: {e}")
        return None

def lookup_by_email(email):
    """Find tenant by email — for login flow."""
    if not email:
        return None
    try:
        resp = _table().scan(
            FilterExpression="email = :e",
            ExpressionAttributeValues={":e": email.strip().lower()}
        )
        items = resp.get("Items", [])
        return items[0] if items else None
    except Exception as e:
        print(f"[dynamo] lookup_by_email error: {e}")
        return None

def get_tenant(tenant_id):
    try:
        resp = _table().get_item(Key={"tenant_id": tenant_id})
        return resp.get("Item")
    except Exception as e:
        print(f"[dynamo] get_tenant error: {e}")
        return None

def increment_build_count(tenant_id):
    if tenant_id == "demo": return 0
    try:
        resp = _table().update_item(
            Key={"tenant_id": tenant_id},
            UpdateExpression="SET build_count = build_count + :i",
            ExpressionAttributeValues={":i": 1},
            ReturnValues="UPDATED_NEW"
        )
        return int(resp["Attributes"]["build_count"])
    except Exception as e:
        print(f"[dynamo] increment error: {e}")
        return 0

def increment_labelled_count(tenant_id):
    if tenant_id == "demo": return 0
    try:
        resp = _table().update_item(
            Key={"tenant_id": tenant_id},
            UpdateExpression="SET labelled_count = labelled_count + :i",
            ExpressionAttributeValues={":i": 1},
            ReturnValues="UPDATED_NEW"
        )
        return int(resp["Attributes"]["labelled_count"])
    except Exception as e:
        return 0

def update_slack_webhook(tenant_id, url):
    try:
        _table().update_item(Key={"tenant_id":tenant_id},
            UpdateExpression="SET slack_webhook = :w",
            ExpressionAttributeValues={":w": url})
    except Exception as e:
        print(f"[dynamo] webhook error: {e}")

def update_thresholds(tenant_id, yellow, red):
    try:
        _table().update_item(Key={"tenant_id":tenant_id},
            UpdateExpression="SET threshold_yellow=:y, threshold_red=:r",
            ExpressionAttributeValues={":y":yellow,":r":red})
    except Exception as e:
        print(f"[dynamo] threshold error: {e}")

def update_model_metadata(tenant_id, phase, precision):
    try:
        _table().update_item(Key={"tenant_id":tenant_id},
            UpdateExpression="SET model_phase=:p, model_precision=:pr, last_retrain=:t",
            ExpressionAttributeValues={
                ":p": phase,
                ":pr": Decimal(str(round(float(precision),4))),
                ":t": _now()
            })
    except Exception as e:
        print(f"[dynamo] model_meta error: {e}")