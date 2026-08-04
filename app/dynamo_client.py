"""
dynamo_client.py - Fixed version with proper auth + email lookup
"""
import os, uuid, hashlib, boto3
from datetime import datetime, timezone
from decimal import Decimal

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from observability import get_logger

log = get_logger("dynamo")

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
    log.info("tenant created", extra={"tenant_id": tenant_id})
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
            log.warning("auth failed: tenant not found", extra={"tenant_id": tenant_id})
            return None
        stored_hash = item.get("api_key_hash","")
        given_hash  = _hash(api_key.strip())
        if stored_hash != given_hash:
            # Deliberately does NOT log the hashes, not even truncated.
            log.warning("auth failed: key mismatch", extra={"tenant_id": tenant_id})
            return None
        return item
    except Exception as e:
        log.error("validate_tenant failed", extra={"tenant_id": tenant_id, "err": str(e)})
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
        log.error("lookup_by_email failed", extra={"err": str(e)})
        return None

def get_tenant(tenant_id):
    try:
        resp = _table().get_item(Key={"tenant_id": tenant_id})
        return resp.get("Item")
    except Exception as e:
        log.error("get_tenant failed", extra={"err": str(e)})
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
        log.error("counter increment failed", extra={"err": str(e)})
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
        log.error("slack webhook update failed", extra={"err": str(e)})

def update_thresholds(tenant_id, yellow, red):
    try:
        _table().update_item(Key={"tenant_id":tenant_id},
            UpdateExpression="SET threshold_yellow=:y, threshold_red=:r",
            ExpressionAttributeValues={":y":yellow,":r":red})
    except Exception as e:
        log.error("threshold update failed", extra={"err": str(e)})

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
        log.error("model metadata update failed", extra={"err": str(e)})
