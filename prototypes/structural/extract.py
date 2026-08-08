"""
Structural features from a diff — prototype.

WHY
    SafeShip's ten features are entirely volume, process and timing: how many
    lines, how many files, what time of day, how flaky the pipeline has been.
    Not one of them says anything about the *shape* of the code being changed.

    The Prime Video deployment-risk study ablated exactly this split:

        change-volume features alone      F1 0.565
        code-level structural metrics     F1 0.809
        structural, volume removed        F1 0.848

    Their conclusion was that "structural code complexity, not raw change
    volume, is the primary driver". SafeShip currently has the 0.565 half.

    A benchmark on ApacheJIT put the ceiling of the current feature set in
    perspective: swapping RandomForest for gradient boosting moved AUC-PR by
    +0.006, while swapping synthetic data for real moved it by +0.130. The model
    is not the constraint. This is an attempt at the thing that is.

WHAT THIS EXTRACTS
    Ten structural signals about the changed code, all computed from the diff
    and the post-change file contents:

        added_complexity      cyclomatic complexity added by this change
        max_nesting_depth     deepest block nesting among changed lines
        decls_added           functions/classes/methods introduced
        decls_removed         functions/classes/methods deleted
        hunk_count            separate change regions (diffusion within files)
        max_hunk_size         largest single contiguous change
        comment_ratio         share of changed lines that are comments
        test_file_ratio       share of changed files that are tests
        languages_touched     distinct languages in one change
        config_file_ratio     share of changed files that are config/infra

TWO PATHS, BY DESIGN
    Python goes through the `ast` module — exact, stdlib, no dependency. Every
    other language goes through a lexical estimator that counts control-flow
    keywords and tracks brace/indent depth. The lexical path is an approximation
    and is labelled as one; `method` in the output says which was used, so a
    consumer can weight them differently rather than silently mixing exact and
    estimated values.

    Prime Video solved multi-language by putting an LLM in the extractor. That
    is a reasonable next step, but it puts a network call and a bill on the
    critical path of every build, so this prototype establishes what is
    reachable without one first.

STATUS: PROTOTYPE
    Nothing imports this. It does not touch the feature contract, the model, or
    safeship_ci. Adding a feature to the model is a contract change and belongs
    in its own deliberate step, after this is shown to carry signal.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

# Languages we can name. Extension is a rough guide, but it is what a diff gives
# us without reading every file.
LANGUAGES = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".jsx": "javascript", ".mjs": "javascript",
    ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".kt": "kotlin", ".scala": "scala",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".c": "c", ".h": "c", ".cc": "cpp", ".cpp": "cpp", ".hpp": "cpp",
    ".cs": "csharp", ".swift": "swift", ".m": "objc",
    ".sh": "shell", ".bash": "shell",
    ".sql": "sql",
}

CONFIG_EXTENSIONS = {".yml", ".yaml", ".json", ".toml", ".ini", ".cfg", ".env",
                     ".tf", ".tfvars", ".properties", ".conf", ".xml"}
CONFIG_NAMES = {"dockerfile", "makefile", "jenkinsfile", "procfile",
                "docker-compose.yml", "requirements.txt", "package.json",
                "go.mod", "cargo.toml", "pom.xml", "build.gradle"}

#: Substrings that mark a path as tests. A change that brings its own tests is
#: a different kind of risk from one that does not, and this is the cheapest
#: possible proxy for that.
TEST_MARKERS = ("test_", "_test.", "/tests/", "/test/", ".test.", ".spec.",
                "spec_", "_spec.", "/spec/", "__tests__")

#: Keywords that branch. Cyclomatic complexity is essentially "how many
#: independent paths", and each of these opens one. Deliberately a union across
#: languages rather than per-language sets — a prototype should not need a
#: grammar per ecosystem to be useful.
BRANCH_KEYWORDS = re.compile(
    r"\b(if|elif|else\s+if|for|foreach|while|case|when|catch|except|"
    r"rescue|and|or|&&|\|\||\?)\b|\?\.|\?\?"
)

COMMENT_PREFIXES = ("#", "//", "/*", "*", "--", "<!--", '"""', "'''")

_TIMEOUT = 15


def _git(*args: str, cwd: Optional[str] = None) -> Optional[str]:
    """Run git, returning stdout or None. Never raises."""
    try:
        out = subprocess.run(("git",) + args, cwd=cwd, capture_output=True,
                             text=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


# ── diff parsing ─────────────────────────────────────────────────────────────

class FileChange:
    """One file in a diff: which lines moved, and what was added or removed."""

    def __init__(self, path: str):
        self.path = path
        self.added: List[str] = []
        self.removed: List[str] = []
        self.hunks: List[Tuple[int, int]] = []   # (start_line, length) post-image

    @property
    def ext(self) -> str:
        return os.path.splitext(self.path)[1].lower()

    @property
    def language(self) -> str:
        return LANGUAGES.get(self.ext, "other")

    @property
    def is_test(self) -> bool:
        p = self.path.lower()
        return any(m in p for m in TEST_MARKERS)

    @property
    def is_config(self) -> bool:
        name = os.path.basename(self.path).lower()
        return self.ext in CONFIG_EXTENSIONS or name in CONFIG_NAMES

    def changed_lines(self) -> set:
        """Post-image line numbers touched, for matching against parsed code."""
        out = set()
        for start, length in self.hunks:
            out.update(range(start, start + max(length, 1)))
        return out


_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def parse_diff(diff_text: str) -> List[FileChange]:
    """
    Parse unified diff output into per-file changes.

    Hand-rolled rather than pulled from a library because this has to ship into
    customer pipelines eventually, where a dependency is a cost.
    """
    files: List[FileChange] = []
    current: Optional[FileChange] = None

    for line in (diff_text or "").splitlines():
        if line.startswith("diff --git"):
            parts = line.split(" b/", 1)
            current = FileChange(parts[1] if len(parts) == 2 else line.split()[-1])
            files.append(current)
            continue
        if current is None:
            continue
        if line.startswith("@@"):
            m = _HUNK.match(line)
            if m:
                start = int(m.group(1))
                length = int(m.group(2) or 1)
                current.hunks.append((start, length))
            continue
        # +++/--- are file headers, not content.
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+"):
            current.added.append(line[1:])
        elif line.startswith("-"):
            current.removed.append(line[1:])

    return files


# ── python: exact, via the ast module ────────────────────────────────────────

_BRANCH_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
                 ast.With, ast.AsyncWith, ast.Assert, ast.IfExp,
                 ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)
_DECL_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def python_metrics(source: str, changed: Optional[set] = None) -> Optional[Dict]:
    """
    Complexity and nesting for the declarations a change actually touched.

    Whole-file complexity would mostly measure how big the file already was —
    editing one line of a 4000-line module would look enormous. Scoping to the
    touched declarations measures the change.
    """
    # Parsing arbitrary customer code emits the customer's warnings — invalid
    # escape sequences, deprecations — into our output, where they look like
    # SafeShip is broken. Their code, their warnings; we only want the metrics.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        return None

    def span(node) -> range:
        return range(getattr(node, "lineno", 0),
                     getattr(node, "end_lineno", getattr(node, "lineno", 0)) + 1)

    def complexity(node) -> int:
        # One path to begin with, plus one per branch point.
        n = 1
        for child in ast.walk(node):
            if isinstance(child, _BRANCH_NODES):
                n += 1
            elif isinstance(child, ast.BoolOp):
                n += len(child.values) - 1
        return n

    def depth(node, level: int = 0) -> int:
        deepest = level
        for child in ast.iter_child_nodes(node):
            step = 1 if isinstance(child, _BRANCH_NODES + _DECL_NODES) else 0
            deepest = max(deepest, depth(child, level + step))
        return deepest

    decls = [n for n in ast.walk(tree) if isinstance(n, _DECL_NODES)]
    touched = [d for d in decls
               if changed is None or changed & set(span(d))]

    return {
        "complexity": sum(complexity(d) for d in touched),
        "max_nesting": max((depth(d) for d in touched), default=0),
        "decls_touched": len(touched),
        "decls_total": len(decls),
        "method": "ast",
    }


# ── everything else: lexical estimate ────────────────────────────────────────

def lexical_metrics(lines: List[str]) -> Dict:
    """
    Approximate complexity from added lines alone, for languages we cannot parse.

    Counts branch keywords and tracks the deepest brace or indent nesting. It
    will disagree with a real parser — a keyword inside a string literal counts,
    for one — so it is reported as `method: "lexical"` and should be trusted
    less than the ast path rather than averaged with it as though equivalent.
    """
    complexity, max_depth, depth = 0, 0, 0
    decls = 0
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith(COMMENT_PREFIXES):
            continue
        complexity += len(BRANCH_KEYWORDS.findall(line))
        if re.search(r"\b(def|function|func|fn|class|interface|struct|impl)\b", line):
            decls += 1
        depth += line.count("{") - line.count("}")
        max_depth = max(max_depth, depth)
        # Indentation as a fallback signal for brace-free languages.
        indent = (len(raw) - len(raw.lstrip())) // 4
        max_depth = max(max_depth, indent)
    return {
        "complexity": complexity,
        "max_nesting": max(0, max_depth),
        "decls_touched": decls,
        "decls_total": decls,
        "method": "lexical",
    }


def is_comment(line: str) -> bool:
    return line.strip().startswith(COMMENT_PREFIXES)


# ── the extractor ────────────────────────────────────────────────────────────

def formatting_ratio(base: str, head: str, cwd: Optional[str] = None) -> Optional[float]:
    """
    Share of changed lines that are pure formatting.

    Found by running this over 40 real commits: a line-ending normalisation
    scored higher for structural complexity than almost anything else in the
    repository. `git diff` reported 35,445 changed lines; `git diff -w` reported
    15. Every line had moved, and none of them meant anything.

    This is not only a prototype problem — SafeShip's existing `diff_size` has
    exactly the same blind spot, so a Prettier run or an import reorder reads as
    one of the largest and riskiest changes a team has ever shipped. Which is
    backwards: a mechanical reformat is among the *safest* things you can
    deploy.

    Returned as a feature in its own right, because "big diff, all formatting"
    and "big diff, all logic" deserve opposite verdicts.
    """
    def churn(*extra: str) -> Optional[int]:
        out = _git("diff", "--numstat", *extra, base, head, cwd=cwd)
        if out is None:
            return None
        total = 0
        for line in out.splitlines():
            parts = line.split("\t")
            if len(parts) >= 2:
                for token in parts[:2]:
                    if token.isdigit():
                        total += int(token)
        return total

    raw, substantive = churn(), churn("-w")
    if raw is None or substantive is None or raw == 0:
        return None
    return round(max(0.0, 1.0 - substantive / raw), 4)


def extract(base: str = "HEAD~1", head: str = "HEAD",
            cwd: Optional[str] = None) -> Dict:
    """
    Structural features for the change between two commits.

    Never raises — same contract as safeship_ci. A feature that cannot be
    computed comes back as None so the server can impute it, rather than as a
    zero that reads like a measurement.

    The diff is taken with -w so that reformatting does not masquerade as
    complexity; how much was reformatted is reported separately as
    `formatting_ratio`.
    """
    diff = _git("diff", "--unified=0", "-w", base, head, cwd=cwd)
    if diff is None:
        return {"error": f"could not diff {base}..{head}", "features": {}}

    files = parse_diff(diff)
    if not files:
        return {"error": None, "features": _empty(), "files": 0}

    total_complexity = 0
    max_nesting = 0
    decls_added = decls_removed = 0
    methods = defaultdict(int)
    languages = set()
    comment_lines = code_lines = 0
    hunk_count = 0
    max_hunk = 0

    for f in files:
        languages.add(f.language)
        hunk_count += len(f.hunks)
        max_hunk = max([max_hunk] + [n for _, n in f.hunks])

        for line in f.added:
            if is_comment(line):
                comment_lines += 1
            elif line.strip():
                code_lines += 1

        if f.language == "python":
            after = _git("show", f"{head}:{f.path}", cwd=cwd)
            m = python_metrics(after, f.changed_lines()) if after else None
            if m is None:                       # deleted, or unparseable
                m = lexical_metrics(f.added)
            before = _git("show", f"{base}:{f.path}", cwd=cwd)
            before_m = python_metrics(before) if before else None
        else:
            m = lexical_metrics(f.added)
            before_m = None

        methods[m["method"]] += 1
        total_complexity += m["complexity"]
        max_nesting = max(max_nesting, m["max_nesting"])

        added_decls = m["decls_touched"]
        if before_m is not None:
            delta = m["decls_total"] - before_m["decls_total"]
            decls_added += max(0, delta)
            decls_removed += max(0, -delta)
        else:
            decls_added += added_decls
            decls_removed += len(
                [l for l in f.removed
                 if re.search(r"\b(def|function|func|class)\b", l)])

    changed = len(files)
    return {
        "error": None,
        "files": changed,
        "features": {
            "added_complexity":  total_complexity,
            "max_nesting_depth": max_nesting,
            "decls_added":       decls_added,
            "decls_removed":     decls_removed,
            "hunk_count":        hunk_count,
            "max_hunk_size":     max_hunk,
            "comment_ratio":     round(comment_lines / max(1, comment_lines + code_lines), 4),
            "test_file_ratio":   round(sum(f.is_test for f in files) / changed, 4),
            "config_file_ratio": round(sum(f.is_config for f in files) / changed, 4),
            "languages_touched": len(languages),
            "formatting_ratio":  formatting_ratio(base, head, cwd),
        },
        # Exact for some files and estimated for others is worth knowing, so a
        # consumer can decide how far to trust the number.
        "method_counts": dict(methods),
        "languages": sorted(languages),
    }


def _empty() -> Dict:
    return {k: None for k in (
        "added_complexity", "max_nesting_depth", "decls_added", "decls_removed",
        "hunk_count", "max_hunk_size", "comment_ratio", "test_file_ratio",
        "config_file_ratio", "languages_touched", "formatting_ratio")}


FEATURE_NAMES = tuple(_empty().keys())
