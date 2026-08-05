"""
Per-platform integration instructions.

WHY THIS PACKAGE EXISTS
    The dashboard used to build a Jenkins Groovy snippet inline, in the middle of
    a route function, unconditionally — right after a README that promises to be
    CI-agnostic. Worse, that snippet hardcoded seven of the ten features:

        "files_changed":5, "recent_failure_rate":0.0, "test_pass_rate":1.0,
        "deployer_exp":30, "days_since_deploy":2, "build_time_delta":0.0

    So anyone who followed the official instructions got a score that barely
    depended on their build. Every snippet here calls safeship_ci instead, which
    measures what it can and reports what it cannot.

THE SHAPE
    Every platform module exposes `describe(tenant_id, api_key, base_url)` and
    returns the same dict, so setup.html and dashboard.html stay
    platform-agnostic and adding GitLab later is one file plus one registry line.

        id             stable identifier, matches safeship_ci's adapter name
        name           human name
        tagline        one line for the platform picker
        docs_url       the provider's own documentation
        secrets        [{name, value, where}]  — where to paste each secret
        prerequisites  [{title, why, fix}]     — the steps that silently degrade
                                                 the score if skipped
        snippet        {filename, language, code}
        log_snippet    optional second snippet for the outcome/learning step
        history_note   what this platform needs before build history works

    `prerequisites` is not decoration. On every platform the default clone is
    shallow and the default token cannot read build history, and both failures
    are invisible — you still get a score, it is just built on medians. Telling
    people up front is the difference between the product working and the product
    appearing to work.
"""
from __future__ import annotations

import os

from . import bitbucket, github, jenkins


def public_base_url():
    """
    The SafeShip URL to put in a customer's pipeline.

    Configurable because the wizard's whole value is being copy-paste
    trustworthy, and a hardcoded IP is not. It still defaults to the current
    deployment so nothing breaks before DNS exists — but note that a plain-HTTP
    default means every build sends its API key in cleartext, which the wizard
    says out loud rather than hiding.
    """
    return os.getenv("SAFESHIP_PUBLIC_URL", "http://54.89.160.150").rstrip("/")

# Order is display order in the picker. GitHub first because it needs the least
# setup — the token is already there, and there is a composite action.
_MODULES = (github, jenkins, bitbucket)

PLATFORMS = {m.ID: m for m in _MODULES}

#: Ids in display order.
ORDER = tuple(m.ID for m in _MODULES)


class UnknownPlatform(ValueError):
    pass


def get(platform_id):
    """The module for a platform id, or raise UnknownPlatform."""
    try:
        return PLATFORMS[(platform_id or "").strip().lower()]
    except KeyError:
        raise UnknownPlatform(
            f"unknown platform {platform_id!r}; expected one of {list(ORDER)}"
        )


def is_valid(platform_id):
    return (platform_id or "").strip().lower() in PLATFORMS


def describe(platform_id, tenant_id, api_key, base_url):
    """Full instructions for one platform."""
    return get(platform_id).describe(tenant_id, api_key, base_url)


def summaries():
    """Just enough for the picker cards, with no credentials involved."""
    return [
        {"id": m.ID, "name": m.NAME, "tagline": m.TAGLINE, "docs_url": m.DOCS_URL,
         "setup_effort": m.SETUP_EFFORT}
        for m in _MODULES
    ]


__all__ = ["PLATFORMS", "ORDER", "UnknownPlatform", "get", "is_valid",
           "describe", "summaries", "public_base_url"]
