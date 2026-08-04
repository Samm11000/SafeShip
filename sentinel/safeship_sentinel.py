"""
SafeShip Sentinel — the post-deploy safety net.

After you deploy, Sentinel probes your service for a short window, compares
its health (error rate + latency) against a quick baseline, and if things
regress it ALERTS you and can TRIGGER A ROLLBACK — turning "is prod on fire?"
from a 2am page into a 30-second automatic action.

It needs nothing but a URL: no Datadog, no Prometheus, no Kubernetes. Run it
as the last step of any deploy pipeline (or by hand).

    python sentinel/safeship_sentinel.py \
        --url https://myservice/health \
        --window 120 --interval 5 \
        --error-rate 0.20 --latency-mult 2.0 \
        --baseline-samples 5 \
        --rollback-cmd "kubectl rollout undo deploy/myservice" \
        --slack-webhook https://hooks.slack.com/services/...

Exit code: 0 = HEALTHY, 1 = DEGRADED — so CI can gate or roll back on it.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from statistics import median
from typing import Callable, List, Optional

try:
    import requests  # nicer + faster if available
except Exception:  # pragma: no cover - fallback path
    requests = None


# ─────────────────────────────────────────────────────────────────────────────
# Probing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Sample:
    ok: bool
    status: Optional[int]
    latency_ms: float


def http_probe(url: str, timeout: float = 5.0) -> Sample:
    """One health probe. Any non-2xx/3xx or exception counts as an error."""
    start = time.perf_counter()
    try:
        if requests is not None:
            resp = requests.get(url, timeout=timeout)
            code = resp.status_code
        else:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                code = r.getcode()
        latency = (time.perf_counter() - start) * 1000.0
        return Sample(ok=200 <= code < 400, status=code, latency_ms=latency)
    except Exception:
        latency = (time.perf_counter() - start) * 1000.0
        return Sample(ok=False, status=None, latency_ms=latency)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation (pure — easy to test)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Report:
    verdict: str            # "HEALTHY" | "DEGRADED"
    total: int
    errors: int
    error_rate: float
    latency_p50: float
    latency_p95: float
    baseline_p50: Optional[float]
    reasons: List[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return self.verdict == "HEALTHY"


def percentile(values: List[float], p: float) -> float:
    """Linear-interpolated percentile (p in [0,1]). 0.0 for empty input."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    if f + 1 < len(s):
        return s[f] + (k - f) * (s[f + 1] - s[f])
    return s[f]


def evaluate(
    samples: List[Sample],
    baseline_p50: Optional[float],
    error_rate_threshold: float,
    latency_mult: float,
) -> Report:
    total = len(samples)
    errors = sum(1 for s in samples if not s.ok)
    error_rate = (errors / total) if total else 1.0

    ok_lat = [s.latency_ms for s in samples if s.ok]
    p50 = percentile(ok_lat, 0.50)
    p95 = percentile(ok_lat, 0.95)

    reasons: List[str] = []
    degraded = False

    if total == 0:
        degraded = True
        reasons.append("no samples collected — service unreachable")

    if total and error_rate > error_rate_threshold:
        degraded = True
        reasons.append(
            f"error rate {error_rate:.0%} exceeds {error_rate_threshold:.0%} threshold "
            f"({errors}/{total} probes failed)"
        )

    if baseline_p50 and ok_lat and p50 > baseline_p50 * latency_mult:
        degraded = True
        reasons.append(
            f"latency p50 {p50:.0f}ms is over {latency_mult:g}x the baseline "
            f"({baseline_p50:.0f}ms)"
        )

    if not degraded:
        reasons.append("error rate and latency stayed within thresholds")

    return Report(
        verdict="DEGRADED" if degraded else "HEALTHY",
        total=total, errors=errors, error_rate=error_rate,
        latency_p50=p50, latency_p95=p95, baseline_p50=baseline_p50,
        reasons=reasons,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Watch loop (orchestration — clock/sleeper/probe injectable for tests)
# ─────────────────────────────────────────────────────────────────────────────

def run_watch(
    urls: List[str],
    window_s: float,
    interval_s: float,
    error_rate_threshold: float,
    latency_mult: float,
    baseline_samples: int = 0,
    probe_fn: Optional[Callable[[str], Sample]] = None,
    sleeper: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    log: Callable[[str], None] = lambda _msg: None,
) -> Report:
    """Optionally capture a baseline, then probe every interval for the window."""
    # Resolved at call time so a monkeypatched http_probe is honored.
    if probe_fn is None:
        probe_fn = http_probe
    baseline_p50: Optional[float] = None
    if baseline_samples > 0:
        log(f"[sentinel] capturing baseline ({baseline_samples} rounds)...")
        base: List[Sample] = []
        for _ in range(baseline_samples):
            for u in urls:
                base.append(probe_fn(u))
            sleeper(interval_s)
        ok_lat = [s.latency_ms for s in base if s.ok]
        baseline_p50 = median(ok_lat) if ok_lat else None
        if baseline_p50 is not None:
            log(f"[sentinel] baseline p50 latency: {baseline_p50:.0f}ms")

    log(f"[sentinel] watching {', '.join(urls)} for {window_s:g}s...")
    samples: List[Sample] = []
    start = clock()
    while True:
        for u in urls:
            samples.append(probe_fn(u))
        if clock() - start >= window_s:
            break
        sleeper(interval_s)

    return evaluate(samples, baseline_p50, error_rate_threshold, latency_mult)


# ─────────────────────────────────────────────────────────────────────────────
# Actions on a DEGRADED verdict
# ─────────────────────────────────────────────────────────────────────────────

def notify_slack(webhook: str, text: str) -> None:  # pragma: no cover - network
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        webhook, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        print(f"[sentinel] slack notify failed: {e}")


def _run_cmd(cmd: str) -> int:  # pragma: no cover - shells out
    print(f"[sentinel] running rollback: {cmd}")
    return subprocess.run(cmd, shell=True, check=False).returncode


def act_on(
    report: Report,
    rollback_cmd: Optional[str] = None,
    slack_webhook: Optional[str] = None,
    runner: Callable[[str], int] = _run_cmd,
    notifier: Callable[[str, str], None] = notify_slack,
) -> List[str]:
    """On DEGRADED: notify + roll back. Returns the actions taken (for tests/logs)."""
    if report.healthy:
        return []
    taken: List[str] = []
    summary = "; ".join(report.reasons)
    if slack_webhook:
        notifier(
            slack_webhook,
            f":rotating_light: *SafeShip Sentinel — deploy DEGRADED*\n{summary}",
        )
        taken.append("slack")
    if rollback_cmd:
        runner(rollback_cmd)
        taken.append("rollback")
    return taken


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

_GREEN, _RED, _DIM, _BOLD, _RST = "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


def render(report: Report) -> str:
    color = _GREEN if report.healthy else _RED
    icon = "PASS" if report.healthy else "FAIL"
    lines = [
        "",
        f"{_BOLD}  SafeShip Sentinel — post-deploy health{_RST}",
        f"  {'-' * 46}",
        f"  verdict       {color}{_BOLD}{report.verdict}{_RST}  [{icon}]",
        f"  probes        {report.total}  ({report.errors} failed)",
        f"  error rate    {report.error_rate:.0%}",
        f"  latency p50   {report.latency_p50:.0f}ms"
        + (f"   p95 {report.latency_p95:.0f}ms" if report.total else ""),
    ]
    if report.baseline_p50 is not None:
        lines.append(f"  baseline p50  {report.baseline_p50:.0f}ms")
    lines.append(f"  {'-' * 46}")
    for r in report.reasons:
        lines.append(f"  {_DIM}• {r}{_RST}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="safeship-sentinel",
        description="Watch a service after deploy; alert and roll back if it regresses.",
    )
    ap.add_argument("--url", action="append", required=True, dest="urls",
                    help="service URL to probe (repeatable)")
    ap.add_argument("--window", type=float, default=120.0, help="watch window seconds (default 120)")
    ap.add_argument("--interval", type=float, default=5.0, help="seconds between probes (default 5)")
    ap.add_argument("--error-rate", type=float, default=0.20,
                    help="max tolerated error rate, 0-1 (default 0.20)")
    ap.add_argument("--latency-mult", type=float, default=2.0,
                    help="degraded if p50 exceeds this x baseline (default 2.0)")
    ap.add_argument("--baseline-samples", type=int, default=5,
                    help="probe rounds to establish baseline (0 to skip)")
    ap.add_argument("--rollback-cmd", default=None, help="shell command to run on DEGRADED")
    ap.add_argument("--slack-webhook", default=None, help="Slack webhook to notify on DEGRADED")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a report")
    args = ap.parse_args(argv)

    report = run_watch(
        urls=args.urls,
        window_s=args.window,
        interval_s=args.interval,
        error_rate_threshold=args.error_rate,
        latency_mult=args.latency_mult,
        baseline_samples=args.baseline_samples,
        log=(lambda m: None) if args.json else (lambda m: print(m)),
    )

    actions = act_on(report, rollback_cmd=args.rollback_cmd, slack_webhook=args.slack_webhook)

    if args.json:
        print(json.dumps({
            "verdict": report.verdict,
            "total": report.total, "errors": report.errors,
            "error_rate": round(report.error_rate, 4),
            "latency_p50_ms": round(report.latency_p50, 1),
            "latency_p95_ms": round(report.latency_p95, 1),
            "baseline_p50_ms": round(report.baseline_p50, 1) if report.baseline_p50 else None,
            "reasons": report.reasons,
            "actions_taken": actions,
        }))
    else:
        print(render(report))
        if actions:
            print(f"  actions taken: {', '.join(actions)}\n")

    return 0 if report.healthy else 1


if __name__ == "__main__":
    sys.exit(main())
