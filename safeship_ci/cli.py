"""
safeship — the pipeline-side CLI.

    safeship score               collect features, score, gate the deploy
    safeship log <0|1>           report the real outcome (the learning signal)
    safeship collect             print what this environment can measure, no network
    safeship watch --url ...     post-deploy health watch (delegates to Sentinel)

EXIT CODES — the important part
    0   proceed. Either the verdict allows it, or SafeShip itself failed and we
        fail OPEN by design.
    1   BLOCKED by the model, and --fail-open was not requested.
    2   bad usage (missing credentials, unknown platform).

    A risk gate that halts the pipeline when *it* is broken gets deleted in a
    week. So an unreachable API, a timeout, a 500 — all of those print a warning
    and exit 0. Only a real BLOCKED verdict stops a deploy.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

from . import http
from .adapters import detect
from .collect import apply_overrides, collect, env_overrides
from .contract import FEATURES

EXIT_OK, EXIT_BLOCKED, EXIT_USAGE = 0, 1, 2


# ── output helpers ───────────────────────────────────────────────────────────

def _is_gha() -> bool:
    return os.getenv("GITHUB_ACTIONS") == "true"


def info(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    # ::warning:: renders in the Actions UI; harmless text elsewhere.
    print(f"::warning::{msg}" if _is_gha() else f"WARNING: {msg}", flush=True)


def error(msg: str) -> None:
    print(f"::error::{msg}" if _is_gha() else f"ERROR: {msg}", file=sys.stderr, flush=True)


def _gha_output(key: str, value: str) -> None:
    """Expose a value to later workflow steps."""
    path = os.getenv("GITHUB_OUTPUT")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(f"{key}={value}\n")
    except OSError:
        pass


def _summary(md: str) -> None:
    """Write to the Actions job summary, so the score is visible in the UI."""
    path = os.getenv("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(md + "\n")
    except OSError:
        pass


def _creds(args) -> Optional[Dict[str, str]]:
    url = args.url or os.getenv("SAFESHIP_URL", "")
    tenant = args.tenant_id or os.getenv("SAFESHIP_TENANT_ID", "")
    key = args.api_key or os.getenv("SAFESHIP_API_KEY", "")
    missing = [n for n, v in (("--url/SAFESHIP_URL", url),
                              ("--tenant-id/SAFESHIP_TENANT_ID", tenant),
                              ("--api-key/SAFESHIP_API_KEY", key)) if not v]
    if missing:
        error("missing credentials: " + ", ".join(missing))
        return None
    if url.startswith("http://") and not args.allow_insecure_url:
        warn(f"{url} is plain HTTP — your API key will cross the network in "
             "cleartext. Use HTTPS, or pass --allow-insecure-url to silence this.")
    return {"url": url, "tenant_id": tenant, "api_key": key}


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_collect(args) -> int:
    out = apply_overrides(collect(detect(force=args.platform), cwd=args.cwd),
                          {**env_overrides(), **_flag_overrides(args)})
    if args.json:
        print(json.dumps({"platform": out.platform, "features": out.features,
                          "sources": out.sources, "meta": out.meta,
                          "notes": out.notes}, indent=2))
    else:
        info(out.summary())
    return EXIT_OK


def cmd_score(args) -> int:
    creds = _creds(args)
    if not creds:
        return EXIT_USAGE

    try:
        out = collect(detect(force=args.platform), cwd=args.cwd,
                      want_history=not args.no_history,
                      want_tests=not args.no_tests)
        out = apply_overrides(out, {**env_overrides(), **_flag_overrides(args)})
    except Exception as exc:
        # Collection must never break a build. Score with nothing and let the
        # server impute everything.
        warn(f"feature collection failed ({type(exc).__name__}) — scoring with "
             "imputed values")
        from .collect import Collection
        out = Collection()

    info(f"SafeShip: {out.platform}, measured {len(out.measured)}/{len(FEATURES)} features")
    for n in out.notes:
        info(f"  · {n}")

    try:
        result = http.score(
            creds["url"], creds["tenant_id"], creds["api_key"],
            out.features, meta=out.meta,
            timeout=args.timeout, retries=args.retries,
            request_id=out.meta.get("build_number") or None,
            insecure=args.insecure_tls,
        )
    except http.SafeShipError as exc:
        # FAIL OPEN. This is the whole contract.
        warn(f"SafeShip unavailable ({exc}) — proceeding without a gate")
        _gha_output("verdict", "UNAVAILABLE")
        _gha_output("score", "")
        return EXIT_OK

    score_val = result.get("score")
    verdict = str(result.get("verdict", "UNKNOWN"))
    imputed = result.get("imputed") or []
    build_id = result.get("build_id", "")

    info(f"SafeShip → score={score_val}/100  verdict={verdict}  "
         f"({result.get('model_phase', '?')} model)")
    for reason in (result.get("top_reasons") or [])[:3]:
        info(f"   - {reason.get('label')}: {reason.get('value_str')}")
    if imputed:
        warn(f"{len(imputed)} feature(s) were not measured and had to be "
             f"estimated: {', '.join(imputed)}")

    if build_id and args.build_id_file:
        try:
            with open(args.build_id_file, "w", encoding="utf-8") as fh:
                fh.write(build_id)
        except OSError as exc:
            warn(f"could not write {args.build_id_file}: {exc}")

    _gha_output("score", str(score_val))
    _gha_output("verdict", verdict)
    _gha_output("build-id", build_id)
    _summary(_markdown(result, out))

    if verdict == "BLOCKED":
        if args.fail_open:
            warn(f"BLOCKED (score {score_val}) but --fail-open is set — proceeding")
            return EXIT_OK
        error(f"SafeShip BLOCKED this deploy (score {score_val}/100). "
              "Override with fail-open, or address the reasons above.")
        return EXIT_BLOCKED

    return EXIT_OK


def cmd_log(args) -> int:
    creds = _creds(args)
    if not creds:
        return EXIT_USAGE

    build_id = args.build_id
    if not build_id and args.build_id_file and os.path.isfile(args.build_id_file):
        try:
            with open(args.build_id_file, "r", encoding="utf-8") as fh:
                build_id = fh.read().strip()
        except OSError:
            build_id = ""
    if not build_id:
        warn("no build_id — nothing to label. Did `safeship score` run first?")
        return EXIT_OK           # never fail a pipeline over a missing label

    try:
        http.log_outcome(creds["url"], creds["tenant_id"], creds["api_key"],
                         build_id, args.label, timeout=args.timeout,
                         retries=args.retries, insecure=args.insecure_tls)
        info(f"SafeShip: outcome {args.label} recorded for {build_id}")
    except http.SafeShipError as exc:
        warn(f"could not record outcome ({exc}) — the model just misses this "
             "one data point")
    return EXIT_OK


def cmd_watch(args) -> int:
    """
    Delegates to the Sentinel script, which already does this well.

    Credentials are passed through so Sentinel can record the outcome itself.
    That closes the learning loop without a separate `safeship log` step — and
    the label then describes what Sentinel *observed* in production rather than
    what the pipeline's exit status implied, which is a materially better signal.
    """
    import subprocess

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sentinel = os.path.join(here, "sentinel", "safeship_sentinel.py")
    if not os.path.isfile(sentinel):
        error(f"sentinel not found at {sentinel}")
        return EXIT_USAGE

    cmd = [sys.executable, sentinel, "--url", args.url_to_watch]
    for flag, value in (("--window", args.window), ("--interval", args.interval)):
        if value is not None:
            cmd += [flag, str(value)]
    if args.rollback_cmd:
        cmd += ["--rollback-cmd", args.rollback_cmd]

    if args.no_label:
        cmd += ["--no-label"]
    else:
        url = args.url or os.getenv("SAFESHIP_URL", "")
        tenant = args.tenant_id or os.getenv("SAFESHIP_TENANT_ID", "")
        key = args.api_key or os.getenv("SAFESHIP_API_KEY", "")
        if url and tenant and key:
            cmd += ["--safeship-url", url, "--tenant-id", tenant, "--api-key", key,
                    "--build-id-file", args.build_id_file]
        else:
            # Watching still works without them; it just teaches nothing.
            warn("no SafeShip credentials — watching but not recording the "
                 "outcome, so this deploy will not train the model")
            cmd += ["--no-label"]

    return subprocess.call(cmd)


def _markdown(result: Dict[str, Any], out) -> str:
    rows = "\n".join(
        f"| {r.get('label')} | {r.get('value_str')} |"
        for r in (result.get("top_reasons") or [])[:3]
    )
    imputed = result.get("imputed") or []
    tail = ("\n\n> ⚠️ Estimated (not measured): " + ", ".join(imputed)) if imputed else ""
    return (f"### SafeShip — {result.get('verdict')} ({result.get('score')}/100)\n\n"
            f"Platform `{out.platform}` · measured {len(out.measured)}/{len(FEATURES)} "
            f"features\n\n| Factor | Value |\n|---|---|\n{rows}{tail}")


def _flag_overrides(args) -> Dict[str, Any]:
    got = {}
    for feature in FEATURES:
        val = getattr(args, feature, None)
        if val is not None:
            got[feature] = val
    return got


# ── argument parsing ─────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="safeship",
        description="Measure a build and ask SafeShip whether it is safe to deploy.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def shared(sp, creds=True):
        if creds:
            sp.add_argument("--url", help="SafeShip base URL [SAFESHIP_URL]")
            sp.add_argument("--tenant-id", help="[SAFESHIP_TENANT_ID]")
            sp.add_argument("--api-key", help="[SAFESHIP_API_KEY]")
            sp.add_argument("--timeout", type=float, default=http.DEFAULT_TIMEOUT)
            sp.add_argument("--retries", type=int, default=http.DEFAULT_RETRIES)
            sp.add_argument("--insecure-tls", action="store_true",
                            help="skip TLS verification (self-signed staging only)")
            sp.add_argument("--allow-insecure-url", action="store_true",
                            help="suppress the plain-HTTP warning")
        sp.add_argument("--platform", choices=["github-actions", "jenkins",
                                              "bitbucket", "generic"],
                        help="override CI detection [SAFESHIP_PLATFORM]")
        sp.add_argument("--cwd", help="repository root (default: current directory)")

    def feature_flags(sp):
        g = sp.add_argument_group(
            "feature overrides",
            "supply or correct any feature; also settable as SAFESHIP_<FEATURE>")
        for f in FEATURES:
            kind = int if f in ("diff_size", "files_changed", "hour_of_day",
                                "day_of_week", "is_hotfix", "deployer_exp") else float
            g.add_argument("--" + f.replace("_", "-"), dest=f, type=kind, default=None)

    sc = sub.add_parser("score", help="collect, score, and gate the deploy")
    shared(sc)
    feature_flags(sc)
    sc.add_argument("--fail-open", action="store_true",
                    help="report a BLOCKED verdict but exit 0 anyway")
    sc.add_argument("--no-history", action="store_true",
                    help="skip CI API calls (offline / no token)")
    sc.add_argument("--no-tests", action="store_true", help="skip JUnit discovery")
    sc.add_argument("--build-id-file", default="safeship_build_id.txt",
                    help="where to write build_id for a later `safeship log`")
    sc.set_defaults(func=cmd_score)

    lg = sub.add_parser("log", help="record the real deploy outcome")
    shared(lg)
    lg.add_argument("label", type=int, choices=[0, 1],
                    help="0 = deploy was fine, 1 = it broke")
    lg.add_argument("--build-id")
    lg.add_argument("--build-id-file", default="safeship_build_id.txt")
    lg.set_defaults(func=cmd_log)

    co = sub.add_parser("collect", help="show what can be measured; no network")
    shared(co, creds=False)
    feature_flags(co)
    co.add_argument("--json", action="store_true")
    co.set_defaults(func=cmd_collect)

    wa = sub.add_parser("watch", help="post-deploy health watch (Sentinel)")
    wa.add_argument("--url", dest="url_to_watch", required=True,
                    help="service URL to probe")
    wa.add_argument("--window", type=int)
    wa.add_argument("--interval", type=int)
    wa.add_argument("--rollback-cmd")
    # Credentials so the watch can record its own verdict. Named --safeship-url
    # here because --url already means "the service to probe".
    wa.add_argument("--safeship-url", dest="url",
                    help="SafeShip base URL [SAFESHIP_URL]")
    wa.add_argument("--tenant-id", help="[SAFESHIP_TENANT_ID]")
    wa.add_argument("--api-key", help="[SAFESHIP_API_KEY]")
    wa.add_argument("--build-id-file", default="safeship_build_id.txt")
    wa.add_argument("--no-label", action="store_true",
                    help="watch only; do not record the outcome")
    wa.set_defaults(func=cmd_watch)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return EXIT_OK
    except Exception as exc:
        # Absolute last resort: an unexpected bug in us must not fail their build.
        warn(f"safeship crashed ({type(exc).__name__}: {exc}) — proceeding")
        return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
