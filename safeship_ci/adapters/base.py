"""
The adapter contract.

Every adapter answers the questions git cannot: what happened on *previous*
builds, how long they took, and who triggered this one. Those four features —
recent_failure_rate, build_time_delta, days_since_deploy and test_pass_rate —
are 30%+ of the model's weight and need the CI provider's own API.

RULES FOR AN ADAPTER
  1. Never raise. Return None for anything it cannot determine. None means
     "unknown", the server imputes from the tenant's history, and the response
     reports that it did. A guess dressed as a measurement is the bug this whole
     package exists to fix.
  2. Never exceed its time budget. A build is waiting.
  3. Use the CI's own native token. safeship_ci holds no credentials of ours and
     stores nothing.
"""
from __future__ import annotations

import os
from typing import Dict, List, NamedTuple, Optional


class BuildRecord(NamedTuple):
    """One historical build, normalised across platforms."""
    succeeded: Optional[bool]        # None when still running / cancelled
    duration_secs: Optional[float]
    finished_at: Optional[float]     # epoch seconds


class Adapter:
    """Base adapter. Subclasses override what their platform can answer."""

    name = "generic"
    #: Total wall-clock budget for all network calls this adapter makes.
    budget_secs = 8.0

    def __init__(self, env: Optional[Dict[str, str]] = None) -> None:
        self.env = dict(os.environ if env is None else env)
        self.notes: List[str] = []

    # ── detection ────────────────────────────────────────────────────────────

    @classmethod
    def detect(cls, env: Optional[Dict[str, str]] = None) -> bool:
        """True when this adapter's platform is the one we are running on."""
        return False

    # ── identity ─────────────────────────────────────────────────────────────

    def actor(self) -> Optional[str]:
        """Who triggered this build. Keys the server's deployer_exp counter."""
        return None

    def job_name(self) -> Optional[str]:
        return None

    def branch(self) -> Optional[str]:
        return None

    def commit(self) -> Optional[str]:
        return None

    def base_commit(self) -> Optional[str]:
        """
        The commit to diff against, when the platform knows better than HEAD~1 —
        a pull request's merge base, or a push event's `before`.
        """
        return None

    def build_number(self) -> Optional[str]:
        return None

    # ── history (the expensive part) ─────────────────────────────────────────

    def history(self, limit: int = 10) -> Optional[List[BuildRecord]]:
        """
        Recent completed builds of *this* job, newest first, excluding the current
        one. None when unavailable (no token, no permission, API down).
        """
        return None

    def current_duration_secs(self) -> Optional[float]:
        """How long the current build has taken so far, if the platform says."""
        return None

    def test_pass_rate(self) -> Optional[float]:
        """Only if the platform exposes test results natively (Jenkins does)."""
        return None

    # ── derived features, shared by every adapter ────────────────────────────

    def note(self, msg: str) -> None:
        self.notes.append(msg)

    def features_from_history(self, limit: int = 10) -> Dict[str, Optional[float]]:
        """
        Turn build history into three features. Shared here so every adapter
        computes them identically — the arithmetic is not platform-specific, only
        the data source is.
        """
        out: Dict[str, Optional[float]] = {
            "recent_failure_rate": None,
            "days_since_deploy": None,
            "build_time_delta": None,
        }

        records = self.history(limit)
        if not records:
            self.note("no build history available")
            return out

        # recent_failure_rate — over builds with a definite outcome. A cancelled
        # build is not a failure.
        decided = [r for r in records if r.succeeded is not None]
        if decided:
            failed = sum(1 for r in decided if not r.succeeded)
            out["recent_failure_rate"] = round(failed / len(decided), 4)
            self.note(f"recent_failure_rate from {len(decided)} build(s)")

        # days_since_deploy — since the last SUCCESSFUL build. A long gap and a
        # very short gap are both riskier than a steady cadence.
        import time as _time

        successes = [r for r in decided if r.succeeded and r.finished_at]
        if successes:
            newest = max(r.finished_at for r in successes)  # type: ignore[type-var]
            days = (_time.time() - float(newest)) / 86400.0
            out["days_since_deploy"] = round(max(0.0, days), 3)

        # build_time_delta — this build's duration vs the median of recent ones,
        # as a signed fraction. +0.5 means it took 50% longer than usual, which
        # often means something unexpected happened.
        durations = sorted(r.duration_secs for r in records
                           if r.duration_secs and r.duration_secs > 0)
        current = self.current_duration_secs()
        if durations and current and current > 0:
            mid = len(durations) // 2
            median = (durations[mid] if len(durations) % 2
                      else (durations[mid - 1] + durations[mid]) / 2.0)
            if median > 0:
                out["build_time_delta"] = round((current - median) / median, 4)

        return out
