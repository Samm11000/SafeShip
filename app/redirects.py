"""
redirects.py — sending someone back where they were trying to go, safely.

WHY THIS IS NOT A ONE-LINER
    "Redirect to whatever ?next= says" is one of the oldest phishing setups
    there is. The URL is genuinely ours, the login page is genuinely ours, the
    user authenticates for real — and then we hand them to the attacker's site,
    which the browser now shows as arriving from us. So `next` is untrusted input
    and has to be validated as an *internal path*, not merely non-empty.

    Rejected, and why each one matters:

      https://evil.test/x     absolute URL, obviously
      //evil.test/x           protocol-relative: browsers read this as
                              https://evil.test/x, and it starts with "/" so a
                              naive `startswith("/")` check waves it through.
                              This is the one people miss.
      /\\evil.test/x           backslash variant; several browsers normalise
                              "/\\" to "//" before resolving.
      /x\\r\\nSet-Cookie: ...   CR/LF, in case anything downstream builds a header
                              from this value.

    Anything that survives is a same-site path, which is the only thing worth
    redirecting to.
"""
from __future__ import annotations

from urllib.parse import quote, urlsplit

DEFAULT_TARGET = "/dashboard"

#: Query parameters worth carrying through a login round-trip. Deliberately a
#: strict allowlist rather than "everything except credentials": /dashboard
#: accepts tenant_id and api_key in the query string, and echoing an api_key into
#: a ?next= — which then lands in browser history, logs and the Referer header —
#: would be a worse leak than the inconvenience it saves.
SAFE_QUERY_KEYS = ("platform",)


def safe_next(raw, default=DEFAULT_TARGET):
    """The path to return to after login, or `default` if it is not ours."""
    target = (raw or "").strip()
    if not target:
        return default

    # Control characters first: they can hide the rest of the checks from a
    # human reading a log line, and matter to anything building a header.
    if any(ch in target for ch in "\r\n\t\x00"):
        return default

    if not target.startswith("/"):
        return default

    # "//host" and "/\host" are absolute in a browser despite the leading slash.
    if target[1:2] in ("/", "\\"):
        return default

    # Belt and braces: a parsed scheme or netloc means it was never a path.
    split = urlsplit(target)
    if split.scheme or split.netloc:
        return default

    return target


def login_url(path, params=None):
    """
    A /login URL that will bounce back to `path` once the user is in.

    Only allowlisted query parameters are carried over — see SAFE_QUERY_KEYS.
    """
    target = safe_next(path, default="")
    if not target:
        return "/login"

    keep = []
    for key in SAFE_QUERY_KEYS:
        value = (params or {}).get(key)
        if value:
            keep.append(f"{key}={quote(str(value), safe='')}")
    if keep:
        target = target + "?" + "&".join(keep)

    return "/login?next=" + quote(target, safe="")


__all__ = ["safe_next", "login_url", "DEFAULT_TARGET", "SAFE_QUERY_KEYS"]
