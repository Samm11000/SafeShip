"""
collect.py — assemble the 10 features from whatever this environment can prove.

The whole point of safeship_ci. Three sources, in order of trustworthiness:

    git        diff_size, files_changed, is_hotfix      — free, no token
    the CI API recent_failure_rate, days_since_deploy,
               build_time_delta, test_pass_rate         — needs the CI's token
    the clock  hour_of_day, day_of_week                 — free

Anything that cannot be measured is reported as **None**, never as a plausible
constant. The server then imputes from the tenant's own history and says so in
the response. That distinction — measured vs guessed — is the reason this package
exists: the old integrations sent hardcoded values and the score silently stopped
depending on the build.

deployer_exp is deliberately NOT collected: the server derives it from build
history so it cannot be spoofed.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from . import gitinfo, junit
from .adapters import Adapter, detect
from .contract import FEATURES


class Collection:
    """The result of a collection run: values, provenance, and warnings."""

    def __init__(self) -> None:
        self.features: Dict[str, Optional[Any]] = {f: None for f in FEATURES}
        self.meta: Dict[str, Any] = {}
        self.sources: Dict[str, str] = {}
        self.notes: List[str] = []
        self.platform: str = "unknown"

    # ── reporting ────────────────────────────────────────────────────────────

    @property
    def measured(self) -> List[str]:
        return [f for f, v in self.features.items() if v is not None]

    @property
    def unknown(self) -> List[str]:
        return [f for f, v in self.features.items() if v is None]

    def set(self, feature: str, value: Any, source: str) -> None:
        if value is None:
            return
        self.features[feature] = value
        self.sources[feature] = source

    def note(self, msg: str) -> None:
        if msg and msg not in self.notes:
            self.notes.append(msg)

    def summary(self) -> str:
        lines = [f"platform: {self.platform}",
                 f"measured {len(self.measured)}/{len(FEATURES)} features"]
        for f in FEATURES:
            v = self.features[f]
            if v is None:
                lines.append(f"  {f:22} —        (unknown, server will impute)")
            else:
                lines.append(f"  {f:22} {str(v):<8} [{self.sources.get(f, '?')}]")
        if self.notes:
            lines.append("notes:")
            lines.extend(f"  - {n}" for n in self.notes)
        return "\n".join(lines)


def collect(adapter: Optional[Adapter] = None,
            cwd: Optional[str] = None,
            want_history: bool = True,
            want_tests: bool = True,
            junit_globs: Optional[tuple] = None) -> Collection:
    """
    Gather everything available. Never raises — a collection failure must not
    break the build it is advising on.
    """
    out = Collection()
    ad = adapter or detect()
    out.platform = ad.name

    # ── the clock ────────────────────────────────────────────────────────────
    # Sent for completeness, though the server prefers its own clock over any
    # client-supplied time (a client could claim it is always mid-afternoon).
    now = time.localtime()
    out.set("hour_of_day", now.tm_hour, "clock")
    out.set("day_of_week", now.tm_wday, "clock")

    # ── git ──────────────────────────────────────────────────────────────────
    branch = ad.branch() or gitinfo.branch_name(cwd)
    try:
        base = ad.base_commit()
        size, files, note = gitinfo.diff_stats(base, cwd)
        out.set("diff_size", size, "git")
        out.set("files_changed", files, "git")
        if size is None:
            out.note(f"diff unavailable — {note}")
            if gitinfo.is_shallow(cwd):
                out.note(gitinfo.shallow_hint())
        else:
            out.note(f"diff {note}")
    except Exception as exc:                                   # pragma: no cover
        out.note(f"git inspection failed: {type(exc).__name__}")

    try:
        out.set("is_hotfix", gitinfo.is_hotfix(branch, cwd), "git")
    except Exception:                                          # pragma: no cover
        pass

    # ── the CI API ───────────────────────────────────────────────────────────
    if want_history:
        try:
            for feature, value in ad.features_from_history().items():
                out.set(feature, value, f"{ad.name}-api")
        except Exception as exc:                               # pragma: no cover
            out.note(f"history lookup failed: {type(exc).__name__}")

    # ── tests ────────────────────────────────────────────────────────────────
    if want_tests:
        rate = None
        try:
            rate = ad.test_pass_rate()          # Jenkins answers this natively
            if rate is not None:
                out.set("test_pass_rate", rate, f"{ad.name}-api")
        except Exception:                                      # pragma: no cover
            rate = None

        if rate is None:
            # Everywhere else: JUnit XML, which most runners already emit.
            try:
                rate, note = junit.pass_rate(cwd or ".", junit_globs)
                if rate is None:
                    out.note(f"test_pass_rate unavailable — {note}. Emit JUnit XML "
                             "(e.g. pytest --junitxml=reports/junit.xml) to supply it.")
                else:
                    out.set("test_pass_rate", rate, "junit-xml")
                    out.note(f"test_pass_rate from JUnit XML: {note}")
            except Exception as exc:                           # pragma: no cover
                out.note(f"JUnit parsing failed: {type(exc).__name__}")

    # ── metadata (not features; used for logging and attribution) ────────────
    out.meta = {
        "job_name": ad.job_name() or os.path.basename(os.getcwd()),
        "branch_name": branch or "",
        "triggered_by": ad.actor() or "unknown",
        "build_number": ad.build_number() or "",
        "commit": ad.commit() or gitinfo.head_sha(cwd) or "",
        "ci_platform": ad.name,
    }

    for n in getattr(ad, "notes", []):
        out.note(n)

    return out


def apply_overrides(out: Collection, overrides: Dict[str, Any]) -> Collection:
    """
    Let a user supply anything we could not measure (or correct what we did),
    via CLI flags or SAFESHIP_<FEATURE> env vars.
    """
    for feature, value in overrides.items():
        if feature in out.features and value is not None:
            out.features[feature] = value
            out.sources[feature] = "override"
    return out


def env_overrides() -> Dict[str, Any]:
    """Reads SAFESHIP_DIFF_SIZE, SAFESHIP_TEST_PASS_RATE, and so on."""
    found: Dict[str, Any] = {}
    for feature in FEATURES:
        raw = os.getenv("SAFESHIP_" + feature.upper())
        if raw is None or raw.strip() == "":
            continue
        try:
            found[feature] = int(raw) if feature in (
                "diff_size", "files_changed", "hour_of_day", "day_of_week",
                "is_hotfix", "deployer_exp") else float(raw)
        except ValueError:
            continue
    return found
