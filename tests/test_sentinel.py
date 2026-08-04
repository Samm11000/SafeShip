"""
Deterministic tests for SafeShip Sentinel — no real network or sleeping.
We inject probe_fn / sleeper / clock / runner so verdicts are reproducible.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO, "sentinel"))

import safeship_sentinel as ss
from safeship_sentinel import Sample, evaluate, run_watch, act_on, percentile, Report


# ── helpers ───────────────────────────────────────────────────────────────────
def ok(latency=50.0):
    return Sample(ok=True, status=200, latency_ms=latency)


def err():
    return Sample(ok=False, status=500, latency_ms=10.0)


def fake_clock(step=1.0):
    t = {"v": 0.0}
    def c():
        v = t["v"]; t["v"] += step; return v
    return c


# ── percentile ──────────────────────────────────────────────────────────────
def test_percentile_basic():
    assert percentile([], 0.5) == 0.0
    assert percentile([100], 0.95) == 100
    assert percentile([10, 20, 30, 40], 0.5) == 25  # interpolated median


# ── evaluate (pure decision) ──────────────────────────────────────────────────
def test_healthy_when_clean():
    r = evaluate([ok(40), ok(45), ok(50)], baseline_p50=45, error_rate_threshold=0.2, latency_mult=2.0)
    assert r.verdict == "HEALTHY" and r.healthy
    assert r.error_rate == 0.0


def test_degraded_on_error_rate():
    samples = [ok(), ok(), err(), err(), err()]  # 60% errors
    r = evaluate(samples, baseline_p50=None, error_rate_threshold=0.2, latency_mult=2.0)
    assert r.verdict == "DEGRADED"
    assert any("error rate" in x for x in r.reasons)


def test_degraded_on_latency_regression():
    # all succeed but ~3x slower than the 100ms baseline
    samples = [ok(300), ok(310), ok(290)]
    r = evaluate(samples, baseline_p50=100, error_rate_threshold=0.5, latency_mult=2.0)
    assert r.verdict == "DEGRADED"
    assert any("latency" in x for x in r.reasons)


def test_degraded_when_no_samples():
    r = evaluate([], baseline_p50=None, error_rate_threshold=0.2, latency_mult=2.0)
    assert r.verdict == "DEGRADED"
    assert any("unreachable" in x for x in r.reasons)


def test_error_rate_at_threshold_is_healthy():
    # exactly at threshold should NOT trip (strictly greater trips)
    samples = [ok(), ok(), ok(), ok(), err()]  # 20%
    r = evaluate(samples, baseline_p50=None, error_rate_threshold=0.2, latency_mult=2.0)
    assert r.verdict == "HEALTHY"


# ── run_watch (orchestration, injected fakes) ─────────────────────────────────
def test_run_watch_healthy_service():
    calls = {"n": 0}
    def probe(_url):
        calls["n"] += 1
        return ok(30)
    r = run_watch(
        ["http://svc/health"], window_s=3, interval_s=1,
        error_rate_threshold=0.2, latency_mult=2.0, baseline_samples=2,
        probe_fn=probe, sleeper=lambda _s: None, clock=fake_clock(1.0),
    )
    assert r.verdict == "HEALTHY"
    assert calls["n"] >= 3  # baseline rounds + watch rounds happened


def test_run_watch_detects_outage():
    r = run_watch(
        ["http://svc/health"], window_s=3, interval_s=1,
        error_rate_threshold=0.2, latency_mult=2.0, baseline_samples=0,
        probe_fn=lambda _u: err(), sleeper=lambda _s: None, clock=fake_clock(1.0),
    )
    assert r.verdict == "DEGRADED"


# ── act_on (alert + rollback wiring) ──────────────────────────────────────────
def _degraded():
    return Report("DEGRADED", 5, 5, 1.0, 0, 0, None, ["service unreachable"])


def _healthy():
    return Report("HEALTHY", 5, 0, 0.0, 30, 40, 30, ["within thresholds"])


def test_act_on_triggers_rollback_when_degraded():
    ran = {}
    notified = {}
    taken = act_on(
        _degraded(), rollback_cmd="echo rollback", slack_webhook="http://hook",
        runner=lambda cmd: ran.setdefault("cmd", cmd) or 0,
        notifier=lambda hook, text: notified.setdefault("text", text),
    )
    assert ran["cmd"] == "echo rollback"
    assert "rollback" in taken and "slack" in taken
    assert "DEGRADED" in notified["text"]


def test_act_on_noop_when_healthy():
    ran = {}
    taken = act_on(
        _healthy(), rollback_cmd="echo rollback",
        runner=lambda cmd: ran.setdefault("cmd", cmd),
    )
    assert taken == []
    assert "cmd" not in ran  # never rolls back a healthy deploy


# ── exit codes via main() ─────────────────────────────────────────────────────
def test_main_exit_code_healthy(monkeypatch):
    monkeypatch.setattr(ss, "http_probe", lambda url, timeout=5.0: ok(20))
    code = ss.main(["--url", "http://svc/health", "--window", "0", "--interval", "0",
                    "--baseline-samples", "0", "--json"])
    assert code == 0


def test_main_exit_code_degraded(monkeypatch):
    monkeypatch.setattr(ss, "http_probe", lambda url, timeout=5.0: err())
    code = ss.main(["--url", "http://svc/health", "--window", "0", "--interval", "0",
                    "--baseline-samples", "0", "--json"])
    assert code == 1
