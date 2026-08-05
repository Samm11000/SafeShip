"""
GitHub Actions adapter.

Everything comes from the runner's own environment and GITHUB_TOKEN. SafeShip
never holds a credential.

TWO TRAPS THIS HANDLES
  1. `actions: read` is NOT granted by default. Repositories created after
     February 2023 default GITHUB_TOKEN to contents+metadata read only, so the
     workflow-runs API returns 403 unless the workflow declares:

         permissions:
           contents: read
           actions: read          # <- required for build history

     A 403 is reported as a precise, actionable message rather than a silent
     fallback, because silently defaulting is what made the old integration
     useless.

  2. Run duration is not in the list response. The documented way to get it is
     GET /actions/runs/{id}/timing, which would cost one extra API call per
     historical run. `updated_at - created_at` from the single list call is used
     as a proxy instead — an approximation, and labelled as one.
"""
from __future__ import annotations

import calendar
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .base import Adapter, BuildRecord

_TIMEOUT = 6


class GitHubActions(Adapter):
    name = "github-actions"

    @classmethod
    def detect(cls, env: Optional[Dict[str, str]] = None) -> bool:
        e = os.environ if env is None else env
        return e.get("GITHUB_ACTIONS") == "true" or bool(e.get("GITHUB_RUN_ID"))

    # ── identity, all free from the environment ──────────────────────────────

    def actor(self) -> Optional[str]:
        # triggering_actor is the person who re-ran a workflow; actor is the
        # original author. For "who is deploying right now", triggering wins.
        return (self.env.get("GITHUB_TRIGGERING_ACTOR")
                or self.env.get("GITHUB_ACTOR") or None)

    def job_name(self) -> Optional[str]:
        wf = self.env.get("GITHUB_WORKFLOW")
        repo = self.env.get("GITHUB_REPOSITORY")
        if wf and repo:
            return f"{repo}/{wf}"
        return wf or repo

    def branch(self) -> Optional[str]:
        # On a pull_request event GITHUB_REF_NAME is "123/merge"; HEAD_REF is the
        # actual source branch, which is what a hotfix check should look at.
        return (self.env.get("GITHUB_HEAD_REF")
                or self.env.get("GITHUB_REF_NAME") or None)

    def commit(self) -> Optional[str]:
        return self.env.get("GITHUB_SHA") or None

    def build_number(self) -> Optional[str]:
        return self.env.get("GITHUB_RUN_NUMBER") or self.env.get("GITHUB_RUN_ID")

    def base_commit(self) -> Optional[str]:
        """
        The right diff base for this event, read from the event payload.

        push          -> event.before (all zeros on a new branch; caller ignores)
        pull_request  -> event.pull_request.base.sha
        """
        path = self.env.get("GITHUB_EVENT_PATH")
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as fh:
                event = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return None

        name = self.env.get("GITHUB_EVENT_NAME", "")
        if name == "pull_request":
            return (event.get("pull_request") or {}).get("base", {}).get("sha")
        return event.get("before")

    # ── the API ──────────────────────────────────────────────────────────────

    def _api(self, path: str, params: str = "") -> Optional[Any]:
        token = self.env.get("GITHUB_TOKEN") or self.env.get("INPUT_GITHUB_TOKEN")
        if not token:
            self.note("no GITHUB_TOKEN — history unavailable "
                      "(pass `github-token: ${{ github.token }}`)")
            return None

        base = self.env.get("GITHUB_API_URL", "https://api.github.com")
        url = f"{base.rstrip('/')}{path}"
        if params:
            url += "?" + params

        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "safeship-ci/1.0",
        })
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            if exc.code == 403:
                # The single most common misconfiguration. Say exactly what to fix.
                self.note("GitHub API 403: the workflow needs `permissions: "
                          "{actions: read}` to read run history")
            elif exc.code == 404:
                self.note(f"GitHub API 404 for {path} (workflow not found?)")
            else:
                self.note(f"GitHub API {exc.code} for {path}")
            return None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.note(f"GitHub API unreachable: {exc}")
            return None
        except json.JSONDecodeError:
            # A corporate proxy or a GitHub incident page answers 200 with HTML.
            self.note("GitHub API returned a non-JSON response (proxy?)")
            return None

    def history(self, limit: int = 10) -> Optional[List[BuildRecord]]:
        repo = self.env.get("GITHUB_REPOSITORY")
        workflow = self.env.get("GITHUB_WORKFLOW_REF") or self.env.get("GITHUB_WORKFLOW")
        if not repo:
            self.note("GITHUB_REPOSITORY not set")
            return None

        # Scope to this workflow *file* when we can — otherwise history mixes in
        # unrelated workflows (lint, docs) and the failure rate becomes noise.
        # GITHUB_WORKFLOW_REF looks like: owner/repo/.github/workflows/ci.yml@refs/...
        wf_file = None
        if workflow and ".github/workflows/" in workflow:
            wf_file = workflow.split(".github/workflows/", 1)[1].split("@", 1)[0]

        params = f"per_page={max(1, min(limit + 1, 100))}&status=completed&exclude_pull_requests=true"
        if wf_file:
            data = self._api(f"/repos/{repo}/actions/workflows/{wf_file}/runs", params)
        else:
            data = self._api(f"/repos/{repo}/actions/runs", params)
            if data:
                self.note("history not scoped to this workflow "
                          "(GITHUB_WORKFLOW_REF unavailable)")

        if not data or "workflow_runs" not in data:
            return None

        current = self.env.get("GITHUB_RUN_ID")
        records: List[BuildRecord] = []
        for run in data["workflow_runs"]:
            if current and str(run.get("id")) == str(current):
                continue  # never let the in-flight build judge itself

            conclusion = run.get("conclusion")
            if conclusion in ("success",):
                ok: Optional[bool] = True
            elif conclusion in ("failure", "timed_out"):
                ok = False
            else:
                # cancelled / skipped / neutral / action_required carry no signal.
                ok = None

            created = _epoch(run.get("created_at"))
            updated = _epoch(run.get("updated_at"))
            # Proxy for duration — see the module docstring.
            duration = (updated - created) if (created and updated and updated > created) else None

            records.append(BuildRecord(succeeded=ok, duration_secs=duration,
                                       finished_at=updated))
            if len(records) >= limit:
                break

        if records:
            self.note(f"{len(records)} previous run(s) from the Actions API"
                      + (f" (workflow: {wf_file})" if wf_file else ""))
        return records or None

    def current_duration_secs(self) -> Optional[float]:
        """
        Elapsed time of this run so far.

        Derived from the event payload's run start rather than an extra API call.
        It measures "time until the gate ran", which is the comparable quantity
        against historical totals only loosely — hence build_time_delta is the
        weakest of the history features.
        """
        started = self.env.get("SAFESHIP_RUN_STARTED_AT")
        if started:
            epoch = _epoch(started)
            if epoch:
                return max(0.0, time.time() - epoch)

        path = self.env.get("GITHUB_EVENT_PATH")
        if path and os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    event = json.load(fh)
                epoch = _epoch((event.get("workflow_run") or {}).get("created_at"))
                if epoch:
                    return max(0.0, time.time() - epoch)
            except (OSError, json.JSONDecodeError):
                pass
        return None


def _epoch(value: Optional[str]) -> Optional[float]:
    """
    Parse a GitHub ISO-8601 timestamp ('2026-08-06T09:12:33Z') to epoch seconds.

    calendar.timegm, not time.mktime: mktime interprets the fields as *local* time
    and ignores the parsed UTC offset, so a runner in a DST-observing timezone
    would misread every timestamp by an hour for half the year — enough to make
    days_since_deploy wrong.
    """
    if not value or not isinstance(value, str):
        return None
    txt = value.strip().replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            parsed = time.strptime(txt, fmt)
        except (ValueError, OverflowError):
            continue
        return calendar.timegm(parsed) - (parsed.tm_gmtoff or 0)
    return None
