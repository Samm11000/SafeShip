"""
Jenkins adapter.

Jenkins is the richest of the three: it is the only platform that exposes test
results through its own API, so test_pass_rate needs no JUnit XML here.

Ports the working logic from ml/feature_extractor.py, but on stdlib urllib rather
than `requests` — safeship_ci must not drag a dependency tree into someone's
build agent. That server-side module keeps its own copy; this one ships to
customers.

Auth: JENKINS_USER + JENKINS_TOKEN (an API token, not a password). Without them
an authenticated Jenkins returns 403 and history degrades to unknown.
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

from ._net import basic_auth, get_json
from .base import Adapter, BuildRecord

_TIMEOUT = 6


def _job_path(job_name: str) -> str:
    """
    JOB_NAME uses '/' for folders and multibranch ("team/service/main"), but the
    URL form needs every segment prefixed with 'job/'.
    """
    segments = [s for s in job_name.split("/") if s]
    return "/".join("job/" + urllib_quote(s) for s in segments)


def urllib_quote(value: str) -> str:
    from urllib.parse import quote

    return quote(value, safe="")


class Jenkins(Adapter):
    name = "jenkins"

    @classmethod
    def detect(cls, env: Optional[Dict[str, str]] = None) -> bool:
        e = os.environ if env is None else env
        return bool(e.get("JENKINS_URL") or e.get("JENKINS_HOME"))

    # ── identity ─────────────────────────────────────────────────────────────

    def actor(self) -> Optional[str]:
        # BUILD_USER_ID needs the build-user-vars plugin; CHANGE_AUTHOR is set on
        # multibranch PR builds. Fall back to the commit author.
        for var in ("BUILD_USER_ID", "CHANGE_AUTHOR", "GIT_AUTHOR_NAME"):
            val = self.env.get(var, "").strip()
            if val:
                return val
        return None

    def job_name(self) -> Optional[str]:
        return self.env.get("JOB_NAME") or None

    def branch(self) -> Optional[str]:
        return (self.env.get("CHANGE_BRANCH") or self.env.get("BRANCH_NAME")
                or self.env.get("GIT_BRANCH") or None)

    def commit(self) -> Optional[str]:
        return self.env.get("GIT_COMMIT") or None

    def base_commit(self) -> Optional[str]:
        # On a PR build Jenkins exposes the previous successful commit.
        return (self.env.get("CHANGE_TARGET")
                or self.env.get("GIT_PREVIOUS_SUCCESSFUL_COMMIT")
                or self.env.get("GIT_PREVIOUS_COMMIT") or None)

    def build_number(self) -> Optional[str]:
        return self.env.get("BUILD_NUMBER") or None

    # ── API ──────────────────────────────────────────────────────────────────

    def _headers(self) -> Dict[str, str]:
        user = self.env.get("JENKINS_USER") or self.env.get("JENKINS_USER_ID")
        token = self.env.get("JENKINS_TOKEN") or self.env.get("JENKINS_API_TOKEN")
        if user and token:
            return basic_auth(user, token)
        return {}

    def history(self, limit: int = 10) -> Optional[List[BuildRecord]]:
        base = (self.env.get("JENKINS_URL") or "").rstrip("/")
        job = self.job_name()
        if not base or not job:
            self.note("JENKINS_URL or JOB_NAME missing")
            return None

        # The tree parameter asks for exactly the fields needed — a plain
        # ?depth=1 would pull megabytes of unrelated build detail.
        url = (f"{base}/{_job_path(job)}/api/json"
               f"?tree=builds[number,result,duration,timestamp]{{0,{limit + 1}}}")
        data, err = get_json(url, self._headers(), _TIMEOUT)
        if err:
            if err == "HTTP 403":
                self.note("Jenkins 403: set JENKINS_USER and JENKINS_TOKEN "
                          "(an API token) for build history")
            else:
                self.note(f"Jenkins API: {err}")
            return None
        if not isinstance(data, dict):
            return None

        current = self.build_number()
        records: List[BuildRecord] = []
        for build in data.get("builds", []) or []:
            if current and str(build.get("number")) == str(current):
                continue
            result = (build.get("result") or "").upper()
            if result == "SUCCESS":
                ok: Optional[bool] = True
            elif result in ("FAILURE", "UNSTABLE"):
                ok = False
            else:
                ok = None   # ABORTED / NOT_BUILT / still running

            duration = build.get("duration")
            secs = float(duration) / 1000.0 if duration else None
            ts = build.get("timestamp")
            finished = (float(ts) / 1000.0 + (secs or 0)) if ts else None

            records.append(BuildRecord(ok, secs, finished))
            if len(records) >= limit:
                break

        if records:
            self.note(f"{len(records)} previous build(s) from the Jenkins API")
        return records or None

    def current_duration_secs(self) -> Optional[float]:
        # Jenkins does not export the current build's start time as an env var,
        # so derive it from the API record for this build number.
        base = (self.env.get("JENKINS_URL") or "").rstrip("/")
        job, num = self.job_name(), self.build_number()
        if not (base and job and num):
            return None
        url = f"{base}/{_job_path(job)}/{num}/api/json?tree=timestamp"
        data, err = get_json(url, self._headers(), _TIMEOUT)
        if err or not isinstance(data, dict):
            return None
        ts = data.get("timestamp")
        return max(0.0, time.time() - float(ts) / 1000.0) if ts else None

    def test_pass_rate(self) -> Optional[float]:
        """Jenkins is the one platform with a native test-results API."""
        build_url = (self.env.get("BUILD_URL") or "").rstrip("/")
        if not build_url:
            return None
        data, err = get_json(f"{build_url}/testReport/api/json"
                             "?tree=passCount,failCount,skipCount",
                             self._headers(), _TIMEOUT)
        if err or not isinstance(data, dict):
            if err and err != "HTTP 404":     # 404 just means no tests published
                self.note(f"Jenkins testReport: {err}")
            return None
        passed = int(data.get("passCount") or 0)
        failed = int(data.get("failCount") or 0)
        considered = passed + failed          # skips excluded, as in junit.py
        if considered <= 0:
            return None
        self.note(f"test_pass_rate from Jenkins testReport ({passed}/{considered})")
        return round(passed / considered, 4)
