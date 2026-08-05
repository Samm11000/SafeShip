"""
gitinfo.py — the features git gives away for free.

diff_size, files_changed and is_hotfix need no CI integration and no token: they
come from the checkout that is already on disk. That makes them the floor of what
any pipeline can report, on any platform.

THE SHALLOW-CLONE TRAP
    actions/checkout defaults to fetch-depth: 1, and Bitbucket clones shallow
    too. In a depth-1 clone `git diff HEAD~1` FAILS, because HEAD~1 does not
    exist locally. The old integration papered over that with a hardcoded
    diff_size of 120.

    Nothing here guesses. If the base commit cannot be resolved, the feature is
    reported as None (unknown), the server imputes it from history, and we print
    an actionable warning naming the exact fix.
"""
from __future__ import annotations

import os
import re
import subprocess
from typing import List, Optional, Tuple

from .contract import HOTFIX_PATTERNS

# The empty tree object — diffing against it yields "everything is new", which is
# the correct reading for a repository's first commit.
EMPTY_TREE = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

_TIMEOUT = 10


def _git(*args: str, cwd: Optional[str] = None) -> Optional[str]:
    """Run git, returning stripped stdout or None. Never raises."""
    try:
        out = subprocess.run(
            ("git",) + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=_TIMEOUT,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip()


def is_repo(cwd: Optional[str] = None) -> bool:
    return _git("rev-parse", "--git-dir", cwd=cwd) is not None


def is_shallow(cwd: Optional[str] = None) -> bool:
    return (_git("rev-parse", "--is-shallow-repository", cwd=cwd) or "false") == "true"


def head_sha(cwd: Optional[str] = None) -> Optional[str]:
    return _git("rev-parse", "HEAD", cwd=cwd)


def branch_name(cwd: Optional[str] = None) -> str:
    """Current branch, preferring CI env vars — a detached HEAD has no branch."""
    for var in ("GITHUB_HEAD_REF", "GITHUB_REF_NAME", "BRANCH_NAME",
                "BITBUCKET_BRANCH", "CI_COMMIT_REF_NAME"):
        val = os.getenv(var, "").strip()
        if val:
            return val
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd) or ""


def _rev_exists(rev: str, cwd: Optional[str] = None) -> bool:
    return _git("rev-parse", "--verify", "--quiet", rev + "^{commit}", cwd=cwd) is not None


def resolve_base(preferred: Optional[str] = None,
                 cwd: Optional[str] = None) -> Tuple[Optional[str], str]:
    """
    Find the commit to diff HEAD against.

    Order: an explicit/platform-supplied base, then HEAD~1, then the empty tree
    for a genuine first commit. Returns (base, how) where `how` explains the
    choice for logging — or (None, reason) when it cannot be resolved, which is a
    real answer rather than a fabricated one.
    """
    if preferred:
        p = preferred.strip()
        # A push event's "before" is all zeros for a new branch.
        if p and set(p) != {"0"} and _rev_exists(p, cwd):
            return p, "platform"

    if _rev_exists("HEAD~1", cwd):
        return "HEAD~1", "head~1"

    # No parent reachable. Either the first commit ever, or a shallow clone.
    count = _git("rev-list", "--count", "HEAD", cwd=cwd)
    if count == "1" and not is_shallow(cwd):
        return EMPTY_TREE, "empty-tree (first commit)"

    if is_shallow(cwd):
        return None, "shallow clone: HEAD~1 not present locally"
    return None, "no parent commit could be resolved"


def diff_stats(base: Optional[str] = None,
               cwd: Optional[str] = None) -> Tuple[Optional[int], Optional[int], str]:
    """
    Returns (diff_size, files_changed, note).

    diff_size is insertions + deletions — churn, which is what correlates with
    risk. A 500-line deletion is not a small change.

    (None, None, reason) when the base cannot be resolved. The server then imputes
    from this tenant's history instead of us inventing a number.
    """
    if not is_repo(cwd):
        return None, None, "not a git repository"

    resolved, how = resolve_base(base, cwd)
    if resolved is None:
        return None, None, how

    numstat = _git("diff", "--numstat", resolved, "HEAD", cwd=cwd)
    if numstat is None:
        return None, None, f"git diff against {how} failed"
    if numstat == "":
        return 0, 0, f"no changes vs {how}"

    added = deleted = files = 0
    for line in numstat.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        files += 1
        # "-" marks a binary file: real change, but no line count.
        for idx, target in ((0, "added"), (1, "deleted")):
            token = parts[idx]
            if token == "-":
                continue
            try:
                value = int(token)
            except ValueError:
                continue
            if target == "added":
                added += value
            else:
                deleted += value

    return added + deleted, files, f"vs {how}"


def is_hotfix(branch: Optional[str] = None, cwd: Optional[str] = None) -> int:
    """1 when the branch name signals urgency. Substring match, case-insensitive."""
    name = (branch if branch is not None else branch_name(cwd)).lower()
    if not name:
        return 0
    return 1 if any(p in name for p in HOTFIX_PATTERNS) else 0


def changed_files(base: Optional[str] = None, cwd: Optional[str] = None) -> List[str]:
    """Paths changed vs the base — used for messages, not for a feature."""
    resolved, _ = resolve_base(base, cwd)
    if resolved is None:
        return []
    out = _git("diff", "--name-only", resolved, "HEAD", cwd=cwd)
    return [l for l in (out or "").splitlines() if l.strip()]


def shallow_hint() -> str:
    """The actionable fix, tailored to whichever CI we appear to be running on."""
    if os.getenv("GITHUB_ACTIONS") == "true":
        return ("shallow checkout: add `with: {fetch-depth: 2}` to actions/checkout "
                "(use 0 for accurate pull-request bases)")
    if os.getenv("BITBUCKET_BUILD_NUMBER"):
        return ("shallow checkout: raise `clone: depth:` in bitbucket-pipelines.yml "
                "(e.g. `clone: {depth: 2}`)")
    if os.getenv("JENKINS_URL"):
        return ("shallow checkout: remove the shallow-clone option from the Git SCM "
                "step, or set depth >= 2")
    return "shallow checkout: fetch at least 2 commits so HEAD~1 exists"
