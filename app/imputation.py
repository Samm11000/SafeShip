"""
imputation.py — what to do when a feature is missing.

WHY THIS EXISTS
    Every optional feature used to fall back to an optimistic constant:

        recent_failure_rate -> 0.0     ("nothing has ever failed")
        test_pass_rate      -> 1.0     ("every test passes")

    Those two carry 52.8% of the model's decision weight, and the training data
    says the *typical* build has recent_failure_rate 0.261 and test_pass_rate
    0.819. So "I could not measure this" was being scored as "this is a perfect
    build" — backwards for a risk product, and the reason a pipeline that sends
    partial data always comes back SAFE.

    Missing data now means *typical*, not *perfect*:

      1. the median of this tenant's own recent builds, when they have history
      2. otherwise the median of the training set (constants below)

    The /score response reports which features were imputed, so a user can see
    what was guessed rather than believing a confident number built on defaults.

    This is what makes partial adoption safe: a team can start with the four
    features git gives away for free and improve from there, without the score
    quietly lying to them.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from typing import Any, Iterable, Mapping

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ml_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml")
if _ml_dir not in sys.path:
    sys.path.insert(0, _ml_dir)

from features import FEATURES  # noqa: E402
from observability import get_logger  # noqa: E402

log = get_logger("imputation")


# --------------------------------------------------------------------------- #
# Fallback: medians of the training set
# --------------------------------------------------------------------------- #
#
# Measured from ml/data/synthetic_builds.csv (3000 rows) — NOT invented. Recompute
# with tools/recompute_medians.py if the base dataset is ever regenerated, because
# an imputed value drawn from a different distribution than the model was trained
# on is its own quiet bug.
TRAINING_MEDIANS: dict[str, float] = {
    "diff_size": 105.0,
    "files_changed": 4.0,
    "hour_of_day": 13.0,
    "day_of_week": 3.0,
    "recent_failure_rate": 0.261,
    "test_pass_rate": 0.819,
    "is_hotfix": 0.0,
    "deployer_exp": 27.0,
    "days_since_deploy": 2.7,
    "build_time_delta": 0.011,
}

# Integer-valued features, so an imputed median does not arrive as 4.0 files.
_INT_FEATURES = {"diff_size", "files_changed", "hour_of_day", "day_of_week",
                 "is_hotfix", "deployer_exp"}

# Features better answered by the server's own clock than by any median.
_CLOCK_FEATURES = {"hour_of_day", "day_of_week"}

# How many of a tenant's recent builds to draw medians from, and how long to
# cache them. Recomputing per request would add an S3 listing to the hot path.
_HISTORY_SIZE = 50
_CACHE_TTL_SECS = 300
_MIN_HISTORY = 8   # below this, a tenant median is noise; prefer the training set


# --------------------------------------------------------------------------- #
# Tenant medians, cached
# --------------------------------------------------------------------------- #

class _MedianCache:
    def __init__(self) -> None:
        self._values: dict[str, dict[str, float]] = {}
        self._stamps: dict[str, float] = {}

    def get(self, tenant_id: str) -> dict[str, float]:
        now = time.time()
        if tenant_id in self._values and (now - self._stamps.get(tenant_id, 0)) < _CACHE_TTL_SECS:
            return self._values[tenant_id]

        medians: dict[str, float] = {}
        try:
            # Reuse the existing reader rather than adding another S3 code path.
            from routes.dashboard import _load_builds

            rows = _load_builds(tenant_id, limit=_HISTORY_SIZE) or []
            if len(rows) >= _MIN_HISTORY:
                medians = _medians_from(rows)
        except Exception as exc:
            # Never let imputation break scoring — fall through to training medians.
            log.warning("tenant median computation failed",
                        extra={"tenant_id": tenant_id, "err": str(exc)})

        self._values[tenant_id] = medians
        self._stamps[tenant_id] = now
        if medians:
            log.info("tenant medians refreshed",
                     extra={"tenant_id": tenant_id, "features": len(medians)})
        return medians

    def invalidate(self, tenant_id: str) -> None:
        self._values.pop(tenant_id, None)
        self._stamps.pop(tenant_id, None)


def _medians_from(rows: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    rows = list(rows)
    for feat in FEATURES:
        vals = []
        for r in rows:
            v = r.get(feat)
            if v is None or v == "":
                continue
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                continue
        if len(vals) >= _MIN_HISTORY:
            out[feat] = float(statistics.median(vals))
    return out


_cache = _MedianCache()


def invalidate(tenant_id: str) -> None:
    """Called after a build is written, so the next score sees fresh history."""
    _cache.invalidate(tenant_id)


# --------------------------------------------------------------------------- #
# The public entry point
# --------------------------------------------------------------------------- #

def impute(values: Mapping[str, Any], tenant_id: str = "base",
           now: time.struct_time | None = None) -> tuple[dict[str, Any], list[str], dict[str, str]]:
    """
    Fill missing features and report exactly what was filled.

    `values` may hold None for any feature. Returns:
        (complete values, list of imputed feature names, source per feature)

    `source` is one of "provided", "tenant_median", "training_median", "clock" —
    surfaced in the API response so a caller can tell a measurement from a guess.
    """
    tenant_medians = _cache.get(tenant_id) if tenant_id else {}
    stamp = now or time.gmtime()

    filled: dict[str, Any] = {}
    imputed: list[str] = []
    source: dict[str, str] = {}

    for feat in FEATURES:
        v = values.get(feat)
        if v is not None:
            filled[feat] = v
            source[feat] = "provided"
            continue

        imputed.append(feat)

        if feat in _CLOCK_FEATURES:
            # The server's clock beats any median, and beats trusting a client
            # that can claim it is always 2pm on a Tuesday.
            filled[feat] = stamp.tm_hour if feat == "hour_of_day" else stamp.tm_wday
            source[feat] = "clock"
            continue

        if feat in tenant_medians:
            filled[feat] = tenant_medians[feat]
            source[feat] = "tenant_median"
        else:
            filled[feat] = TRAINING_MEDIANS[feat]
            source[feat] = "training_median"

        if feat in _INT_FEATURES:
            filled[feat] = int(round(float(filled[feat])))

    return filled, imputed, source
