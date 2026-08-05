"""
http.py — talking to SafeShip, and the fail-open contract.

THE RULE
    SafeShip must never break a build it was only asked to advise on. A risk gate
    that halts everyone's deploys when *it* has an outage gets deleted within a
    week (PRODUCTION-PLAN.md §3.3).

    So there are two distinct kinds of "no":

      BLOCKED  — the gate worked and the model says this deploy is risky.
                 Exit 1. That is the product doing its job.

      ERROR    — SafeShip was unreachable, slow, or returned nonsense.
                 Warn loudly and exit 0. The pipeline proceeds.

    Conflating those two is the difference between a useful gate and an outage
    amplifier.

stdlib urllib only — this runs in customers' pipelines and must not drag a
dependency tree behind it.
"""
from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

DEFAULT_TIMEOUT = float(os.getenv("SAFESHIP_TIMEOUT", "10"))
DEFAULT_RETRIES = int(os.getenv("SAFESHIP_RETRIES", "2"))
USER_AGENT = "safeship-ci/1.0"


class SafeShipError(Exception):
    """Transport or protocol failure. Callers treat this as fail-open."""


def _redact(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    if "api_key" in out:
        out["api_key"] = "[REDACTED]"
    return out


def post(url: str, path: str, payload: Dict[str, Any],
         timeout: float = DEFAULT_TIMEOUT,
         retries: int = DEFAULT_RETRIES,
         request_id: Optional[str] = None,
         insecure: bool = False) -> Dict[str, Any]:
    """
    POST JSON and return the decoded response.

    Retries only on transport errors and 5xx — never on a 4xx, which means our
    request was wrong and repeating it will not help. Raises SafeShipError after
    the final attempt; it never raises anything else.
    """
    endpoint = url.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }
    if request_id:
        # Lets one build be traced through the server's structured logs.
        headers["X-Request-ID"] = request_id

    context = None
    if endpoint.startswith("https://") and insecure:
        # Only for self-signed staging endpoints, and only when asked.
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    last = "unknown error"
    for attempt in range(retries + 1):
        req = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=context) as resp:
                raw = resp.read().decode("utf-8", "replace")
                try:
                    return json.loads(raw)
                except json.JSONDecodeError:
                    raise SafeShipError(
                        f"{endpoint} returned non-JSON ({resp.status}): {raw[:200]}"
                    )

        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:300]
            except Exception:
                pass
            # 4xx is our fault (bad key, invalid payload) — retrying is pointless.
            if 400 <= exc.code < 500:
                raise SafeShipError(f"HTTP {exc.code} from {endpoint}: {detail}")
            last = f"HTTP {exc.code} from {endpoint}: {detail}"

        except urllib.error.URLError as exc:
            last = f"cannot reach {endpoint}: {exc.reason}"
        except (TimeoutError, OSError) as exc:
            last = f"{type(exc).__name__} talking to {endpoint}: {exc}"

        if attempt < retries:
            # Short linear backoff: a build is waiting on this.
            time.sleep(0.6 * (attempt + 1))

    raise SafeShipError(last)


def score(url: str, tenant_id: str, api_key: str, features: Dict[str, Any],
          meta: Optional[Dict[str, Any]] = None, **kw) -> Dict[str, Any]:
    """POST /score. `features` may hold None — the server imputes and says so."""
    payload: Dict[str, Any] = {"tenant_id": tenant_id, "api_key": api_key}
    payload.update(features)
    if meta:
        payload.update(meta)
    return post(url, "/score", payload, **kw)


def log_outcome(url: str, tenant_id: str, api_key: str, build_id: str,
                label: int, source: Optional[str] = None, **kw) -> Dict[str, Any]:
    """
    POST /log. label 0 = deploy was fine, 1 = it broke.

    This is the training signal. Without it the model never learns from reality.

    The default source says where the label came from: CI status, which is a
    weaker signal than an actual production check. It used to send "failure" /
    "success", and the server's weighting list contained "failure" and "safe" —
    so every successful deploy fell through to a lower sample_weight because of a
    string mismatch nothing would ever surface. `ci_failure` / `ci_success` are
    the canonical names (app/labels.py); Sentinel sends `sentinel_*` instead,
    which is trusted more because it observed production rather than inferring
    from a pipeline.
    """
    return post(url, "/log", {
        "tenant_id": tenant_id,
        "api_key": api_key,
        "build_id": build_id,
        "label": int(label),
        "label_source": source or ("ci_failure" if int(label) == 1 else "ci_success"),
    }, **kw)


def describe(payload: Dict[str, Any]) -> str:
    """Loggable form of a request — never leaks the key."""
    return json.dumps(_redact(payload), sort_keys=True)
