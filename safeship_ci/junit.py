"""
junit.py — test_pass_rate from JUnit XML.

WHY THIS MATTERS
    test_pass_rate carries 25% of the model's decision weight, and only Jenkins
    exposes it through an API. GitHub Actions and Bitbucket Pipelines have no
    native test-results endpoint, so the only portable source is the JUnit XML
    that essentially every test runner can already emit:

        pytest   --junitxml=reports/junit.xml
        jest     --reporters=jest-junit
        go test  | go-junit-report > report.xml
        mvn                              (surefire-reports/*.xml by default)
        gradle                           (build/test-results/**/*.xml)
        dotnet test --logger "junit"

    Discovery is by glob over the usual locations, so most projects need to
    configure nothing.

DELIBERATELY NOT COUNTING SKIPS
    pass_rate = passed / (total - skipped). A suite of 100 tests where 90 are
    skipped and 10 pass is 100% passing, not 10% — skipped tests carry no signal
    about this build, and counting them as failures would punish teams who skip
    platform-specific tests.
"""
from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple

# Ordered by how specific they are. Checked relative to the working directory.
DEFAULT_GLOBS = (
    "**/junit*.xml",
    "**/TEST-*.xml",                     # maven surefire / ant
    "**/test-results/**/*.xml",          # gradle
    "**/surefire-reports/*.xml",
    "**/reports/**/*.xml",
    "**/*junit*.xml",
    "**/test-report*.xml",
    "**/pytest.xml",
)

# Directories that would produce false positives or take forever to walk.
_SKIP_DIRS = ("node_modules", ".venv", "venv", ".git", "site-packages",
              ".tox", "dist", "build/tmp", ".mypy_cache", ".pytest_cache")

_MAX_FILES = 200


def discover(root: str = ".", patterns: Optional[Tuple[str, ...]] = None) -> List[str]:
    """Find candidate JUnit XML files, newest first."""
    found: List[str] = []
    for pattern in (patterns or DEFAULT_GLOBS):
        try:
            for path in glob.iglob(os.path.join(root, pattern), recursive=True):
                if any(s in path.replace("\\", "/") for s in _SKIP_DIRS):
                    continue
                if os.path.isfile(path) and path not in found:
                    found.append(path)
                    if len(found) >= _MAX_FILES:
                        return found
        except OSError:
            continue
    # Newest first: a re-run's fresh report should win over a stale one.
    found.sort(key=lambda p: os.path.getmtime(p) if os.path.exists(p) else 0, reverse=True)
    return found


def _counts_from(path: str) -> Optional[Tuple[int, int, int, int]]:
    """
    (tests, failures, errors, skipped) from one file, or None if unparseable.

    Handles both a bare <testsuite> root and a <testsuites> wrapper. Prefers the
    root's own attributes when present, because nested suites would double-count.
    """
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError):
        return None

    def attrs(el) -> Tuple[int, int, int, int]:
        def n(name: str, *alts: str) -> int:
            for key in (name,) + alts:
                if key in el.attrib:
                    try:
                        return int(float(el.attrib[key]))
                    except (TypeError, ValueError):
                        continue
            return 0
        return (n("tests"), n("failures"), n("errors"),
                n("skipped", "skip", "disabled"))

    tag = root.tag.split("}")[-1].lower()

    if tag == "testsuite":
        return attrs(root)

    if tag == "testsuites":
        # A <testsuites> wrapper usually carries totals; trust them if present.
        totals = attrs(root)
        if totals[0] > 0:
            return totals
        summed = [0, 0, 0, 0]
        for suite in root.iter():
            if suite is root or suite.tag.split("}")[-1].lower() != "testsuite":
                continue
            for i, v in enumerate(attrs(suite)):
                summed[i] += v
        return tuple(summed) if summed[0] > 0 else None  # type: ignore[return-value]

    return None


def pass_rate(root: str = ".",
              patterns: Optional[Tuple[str, ...]] = None
              ) -> Tuple[Optional[float], str]:
    """
    Aggregate test_pass_rate across every discovered report.

    Returns (rate in 0..1, note). (None, reason) when nothing usable was found —
    which is a real answer: the server imputes rather than assuming 1.0.
    """
    files = discover(root, patterns)
    if not files:
        return None, "no JUnit XML found"

    tests = failures = errors = skipped = 0
    used = 0
    for path in files:
        counts = _counts_from(path)
        if not counts:
            continue
        t, f, e, s = counts
        if t <= 0:
            continue
        tests += t
        failures += f
        errors += e
        skipped += s
        used += 1

    if tests <= 0:
        return None, f"found {len(files)} XML file(s) but no test counts"

    considered = tests - skipped
    if considered <= 0:
        # Everything was skipped — no signal about this build either way.
        return None, f"all {tests} tests skipped"

    passed = max(0, considered - failures - errors)
    rate = round(min(1.0, passed / considered), 4)
    note = (f"{passed}/{considered} passed across {used} report(s)"
            + (f", {skipped} skipped (excluded)" if skipped else ""))
    return rate, note
