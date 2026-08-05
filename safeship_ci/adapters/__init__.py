"""
Adapter registry.

Order matters: the specific platforms are probed first and Generic matches
everything, so it must come last.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Type

from .base import Adapter, BuildRecord
from .bitbucket import BitbucketPipelines
from .generic import Generic
from .github import GitHubActions
from .jenkins import Jenkins

REGISTRY: List[Type[Adapter]] = [
    GitHubActions,
    Jenkins,
    BitbucketPipelines,
    Generic,       # must stay last — detect() always returns True
]

BY_NAME = {cls.name: cls for cls in REGISTRY}


def detect(env: Optional[Dict[str, str]] = None,
           force: Optional[str] = None) -> Adapter:
    """
    Pick an adapter for the current environment.

    `force` (SAFESHIP_PLATFORM) overrides detection, which is what makes the
    adapters testable offline and lets a user correct a bad guess.
    """
    e = dict(os.environ if env is None else env)
    name = (force or e.get("SAFESHIP_PLATFORM", "")).strip().lower()
    if name:
        cls = BY_NAME.get(name)
        if cls:
            return cls(e)
        raise ValueError(
            f"unknown platform {name!r}; expected one of {sorted(BY_NAME)}"
        )

    for cls in REGISTRY:
        try:
            if cls.detect(e):
                return cls(e)
        except Exception:
            continue
    return Generic(e)


__all__ = ["Adapter", "BuildRecord", "REGISTRY", "BY_NAME", "detect",
           "GitHubActions", "Jenkins", "BitbucketPipelines", "Generic"]
