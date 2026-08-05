"""
Fallback adapter: git only.

Used when no supported CI is detected — a laptop, an unsupported provider, or a
plain shell script. Reports the four features git can prove and None for the
rest, which the server imputes from tenant history.

This is the honest floor of the product: it still scores, it just says out loud
that half the signal is unavailable.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from .base import Adapter


class Generic(Adapter):
    name = "generic"

    @classmethod
    def detect(cls, env: Optional[Dict[str, str]] = None) -> bool:
        return True   # always matches; the registry tries it last

    def actor(self) -> Optional[str]:
        # Best effort, in order of trustworthiness.
        for var in ("SAFESHIP_ACTOR", "GIT_AUTHOR_NAME", "USER", "USERNAME"):
            val = self.env.get(var, "").strip()
            if val:
                return val
        return None

    def job_name(self) -> Optional[str]:
        return self.env.get("SAFESHIP_JOB_NAME") or None
