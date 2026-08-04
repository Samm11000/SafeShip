"""
observability.py — structured logging, request correlation, and metrics.

WHY THIS EXISTS
    The app previously logged with ~150 bare `print()` calls. That is invisible
    in production: no levels, no timestamps, nothing to filter on, and no way to
    follow a single request through score -> log -> retrain.

WHAT YOU GET
    log = get_logger(__name__)
    log.info("scored", extra={"tenant_id": tid, "score": 91})

    Every line comes out as one JSON object with a `request_id` attached, so
    CloudWatch Logs Insights (or jq, or anything) can filter and correlate:

        {"ts":"2026-08-05T02:14:33Z","level":"INFO","logger":"routes.score",
         "msg":"scored","request_id":"a3f1...","tenant_id":"74f1...","score":91}

    Set LOG_FORMAT=text for human-readable local development.

NEVER LOG
    API keys, session cookies, or any part of a credential hash — not even
    truncated. See scrub() below.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict

# --------------------------------------------------------------------------- #
# Request correlation
# --------------------------------------------------------------------------- #

# A ContextVar rather than flask.g: this also works in the Lambda retrain and
# drift handlers, and in Sentinel, none of which have a Flask request context.
_request_id: ContextVar[str] = ContextVar("request_id", default="-")
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:16]


def set_request_id(value: str | None) -> str:
    rid = (value or new_request_id()).strip()[:64]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


def set_tenant_id(value: str | None) -> None:
    _tenant_id.set((value or "-")[:64])


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #

# Substring match, lowercased — anything whose key looks like a credential is
# replaced wholesale rather than truncated. A truncated hash is still a hash.
_SENSITIVE = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "passwd",
    "token",
    "authorization",
    "cookie",
    "hash",
    "webhook",
)


def scrub(value: Any) -> Any:
    """Recursively redact credential-shaped values before they reach a log."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if any(s in str(k).lower() for s in _SENSITIVE):
                out[k] = "[REDACTED]"
            else:
                out[k] = scrub(v)
        return out
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


# --------------------------------------------------------------------------- #
# Formatters
# --------------------------------------------------------------------------- #

# LogRecord attributes that are built in — anything else the caller passed via
# `extra=` is treated as a structured field and merged into the JSON output.
_STD_ATTRS = {
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "module", "msecs",
    "message", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": get_request_id(),
        }
        tid = _tenant_id.get()
        if tid and tid != "-":
            payload["tenant_id"] = tid

        for key, value in record.__dict__.items():
            if key not in _STD_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)

        payload = scrub(payload)
        # default=str so a stray datetime or Decimal can never break logging.
        return json.dumps(payload, default=str, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Readable local output. Same fields, easier on human eyes."""

    def format(self, record: logging.LogRecord) -> str:
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in _STD_ATTRS and not k.startswith("_")
        }
        extras = scrub(extras)
        tail = " ".join(f"{k}={v}" for k, v in extras.items())
        rid = get_request_id()
        base = (
            f"{time.strftime('%H:%M:%S', time.localtime(record.created))} "
            f"{record.levelname:<5} [{rid}] {record.name}: {record.getMessage()}"
        )
        if tail:
            base = f"{base}  {tail}"
        if record.exc_info:
            base = f"{base}\n{self.formatException(record.exc_info)}"
        return base


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

_configured = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Idempotent. Safe to call from app factory, Lambda handler, or a script."""
    global _configured
    if _configured:
        return

    level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt = (fmt or os.getenv("LOG_FORMAT", "json")).lower()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())

    root = logging.getLogger()
    # Replace, don't append: gunicorn and Flask both install handlers, and we do
    # not want every line emitted two or three times.
    root.handlers = [handler]
    root.setLevel(level)

    # Werkzeug's own access log duplicates ours, with less information.
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    # boto3/botocore are extremely chatty at INFO.
    for noisy in ("boto3", "botocore", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
#
# A deliberately small in-process registry exposed at /metrics in Prometheus
# text format.
#
# HONEST LIMITATION: these counters are per-process. Under gunicorn with 2
# workers a scrape hits one worker, so absolute counts are a sample, not a
# total. That is fine for latency distribution and relative rates, and it needs
# no shared volume or extra dependency.
#
# The structured access log is the source of truth for exact counts — derive
# them with a CloudWatch metric filter or Logs Insights query. Move to
# prometheus_client's multiprocess mode when a real Prometheus is introduced.

_counters: Dict[str, float] = {}
_hist_buckets = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
_hists: Dict[str, Dict[str, float]] = {}


def _key(name: str, labels: Dict[str, Any] | None) -> str:
    if not labels:
        return name
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


def incr(name: str, labels: Dict[str, Any] | None = None, by: float = 1.0) -> None:
    _counters[_key(name, labels)] = _counters.get(_key(name, labels), 0.0) + by


def observe(name: str, seconds: float, labels: Dict[str, Any] | None = None) -> None:
    k = _key(name, labels)
    h = _hists.setdefault(k, {"count": 0.0, "sum": 0.0})
    h["count"] += 1
    h["sum"] += seconds
    for b in _hist_buckets:
        if seconds <= b:
            h[f"le_{b}"] = h.get(f"le_{b}", 0.0) + 1


def render_metrics() -> str:
    """Prometheus text exposition format."""
    lines = []
    for k, v in sorted(_counters.items()):
        lines.append(f"{k} {v}")
    for k, h in sorted(_hists.items()):
        if "{" in k:
            base, labels = k.split("{", 1)
            labels = "," + labels[:-1]
        else:
            base, labels = k, ""
        cumulative = 0.0
        for b in _hist_buckets:
            cumulative += h.get(f"le_{b}", 0.0)
            lines.append(f'{base}_bucket{{le="{b}"{labels}}} {cumulative}')
        lines.append(f'{base}_bucket{{le="+Inf"{labels}}} {h["count"]}')
        lines.append(f"{base}_count{{{labels[1:]}}} {h['count']}" if labels else f"{base}_count {h['count']}")
        lines.append(f"{base}_sum{{{labels[1:]}}} {h['sum']}" if labels else f"{base}_sum {h['sum']}")
    return "\n".join(lines) + "\n"
