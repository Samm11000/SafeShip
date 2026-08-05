"""
Bitbucket Pipelines adapter.

Bitbucket has no native test-results API, so test_pass_rate comes from JUnit XML
(see junit.py) rather than from here.

AUTH IS NOT AUTOMATIC
    Unlike GITHUB_TOKEN, Bitbucket does not hand a build an API token by default.
    A Repository Access Token must be added as a secured repository variable —
    that is the right primitive here because it is repo-scoped and revocable,
    unlike an account-wide app password.

        BITBUCKET_ACCESS_TOKEN   (recommended: Repository Access Token)

    Without it, history degrades to unknown and the reason is reported.

SORTING QUIRK
    `sort=-completed_on` returns invalid-sort-attribute. `-created_on` is the
    supported descending sort.
"""
from __future__ import annotations

import calendar
import os
import time
from typing import Dict, List, Optional

from ._net import get_json
from .base import Adapter, BuildRecord

_API = "https://api.bitbucket.org/2.0"
_TIMEOUT = 6


class BitbucketPipelines(Adapter):
    name = "bitbucket"

    @classmethod
    def detect(cls, env: Optional[Dict[str, str]] = None) -> bool:
        e = os.environ if env is None else env
        return bool(e.get("BITBUCKET_BUILD_NUMBER") or e.get("BITBUCKET_PIPELINE_UUID"))

    # ── identity ─────────────────────────────────────────────────────────────

    def actor(self) -> Optional[str]:
        # Bitbucket exposes a UUID rather than a username. That is fine: the
        # server only needs a stable key to count builds per person.
        return (self.env.get("BITBUCKET_STEP_TRIGGERER_UUID")
                or self.env.get("BITBUCKET_WORKSPACE") or None)

    def job_name(self) -> Optional[str]:
        repo = self.env.get("BITBUCKET_REPO_FULL_NAME")
        return f"{repo}/pipelines" if repo else None

    def branch(self) -> Optional[str]:
        return (self.env.get("BITBUCKET_PR_BRANCH")
                or self.env.get("BITBUCKET_BRANCH") or None)

    def commit(self) -> Optional[str]:
        return self.env.get("BITBUCKET_COMMIT") or None

    def base_commit(self) -> Optional[str]:
        return self.env.get("BITBUCKET_PR_DESTINATION_COMMIT") or None

    def build_number(self) -> Optional[str]:
        return self.env.get("BITBUCKET_BUILD_NUMBER") or None

    # ── API ──────────────────────────────────────────────────────────────────

    def _repo(self) -> Optional[str]:
        full = self.env.get("BITBUCKET_REPO_FULL_NAME")
        if full and "/" in full:
            return full
        ws = self.env.get("BITBUCKET_WORKSPACE")
        slug = self.env.get("BITBUCKET_REPO_SLUG")
        return f"{ws}/{slug}" if ws and slug else None

    def _headers(self) -> Optional[Dict[str, str]]:
        token = (self.env.get("BITBUCKET_ACCESS_TOKEN")
                 or self.env.get("BITBUCKET_TOKEN")
                 or self.env.get("BITBUCKET_STEP_OAUTH_TOKEN"))
        if not token:
            self.note("no BITBUCKET_ACCESS_TOKEN — history unavailable. Add a "
                      "Repository Access Token as a secured variable.")
            return None
        return {"Authorization": f"Bearer {token}"}

    def history(self, limit: int = 10) -> Optional[List[BuildRecord]]:
        repo = self._repo()
        if not repo:
            self.note("BITBUCKET_REPO_FULL_NAME / WORKSPACE+REPO_SLUG missing")
            return None
        headers = self._headers()
        if headers is None:
            return None

        url = (f"{_API}/repositories/{repo}/pipelines"
               f"?sort=-created_on&pagelen={max(1, min(limit + 1, 100))}")
        data, err = get_json(url, headers, _TIMEOUT)
        if err:
            if err in ("HTTP 401", "HTTP 403"):
                self.note(f"Bitbucket {err}: the access token needs read access "
                          "to pipelines")
            else:
                self.note(f"Bitbucket API: {err}")
            return None
        if not isinstance(data, dict):
            return None

        current = self.build_number()
        records: List[BuildRecord] = []
        for run in data.get("values", []) or []:
            if current and str(run.get("build_number")) == str(current):
                continue

            result = (((run.get("state") or {}).get("result") or {})
                      .get("name") or "").upper()
            if result == "SUCCESSFUL":
                ok: Optional[bool] = True
            elif result in ("FAILED", "ERROR"):
                ok = False
            else:
                ok = None   # STOPPED / PAUSED / IN_PROGRESS

            secs = run.get("duration_in_seconds")
            secs = float(secs) if secs else None
            finished = _epoch(run.get("completed_on") or run.get("created_on"))

            records.append(BuildRecord(ok, secs, finished))
            if len(records) >= limit:
                break

        if records:
            self.note(f"{len(records)} previous pipeline(s) from the Bitbucket API")
        return records or None

    def current_duration_secs(self) -> Optional[float]:
        repo, uuid = self._repo(), self.env.get("BITBUCKET_PIPELINE_UUID")
        headers = self._headers()
        if not (repo and uuid and headers):
            return None
        # The UUID arrives brace-wrapped: {abc-123}
        data, err = get_json(f"{_API}/repositories/{repo}/pipelines/{uuid}",
                             headers, _TIMEOUT)
        if err or not isinstance(data, dict):
            return None
        started = _epoch(data.get("created_on"))
        return max(0.0, time.time() - started) if started else None


def _epoch(value: Optional[str]) -> Optional[float]:
    """
    Bitbucket timestamps look like 2026-08-06T09:12:33.123456+00:00.

    calendar.timegm rather than time.mktime — see the note in github.py::_epoch;
    mktime would misread the offset in any DST-observing timezone.
    """
    if not value or not isinstance(value, str):
        return None
    txt = value.strip().replace("Z", "+0000")
    # Normalise +00:00 -> +0000 for %z on older Pythons.
    if len(txt) > 6 and txt[-3] == ":" and txt[-6] in "+-":
        txt = txt[:-3] + txt[-2:]
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = time.strptime(txt, fmt)
        except (ValueError, OverflowError):
            continue
        return calendar.timegm(parsed) - (parsed.tm_gmtoff or 0)
    return None
