"""Tiny shared HTTP helper for adapters. stdlib only, never raises."""
from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple


def get_json(url: str, headers: Optional[Dict[str, str]] = None,
             timeout: float = 6.0) -> Tuple[Optional[Any], Optional[str]]:
    """Returns (data, error_message). Exactly one is non-None."""
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "safeship-ci/1.0",
        **(headers or {}),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace")), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return None, f"unreachable: {exc.reason}"
    except json.JSONDecodeError:
        return None, "non-JSON response"
    except (TimeoutError, OSError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


def basic_auth(user: str, token: str) -> Dict[str, str]:
    raw = f"{user}:{token}".encode("utf-8")
    return {"Authorization": "Basic " + base64.b64encode(raw).decode("ascii")}
