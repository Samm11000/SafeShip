"""
The feature contract, as the client sees it.

DUPLICATE OF ml/features.py::FEATURES — INTENTIONAL.

safeship_ci ships to customers' pipelines as a standalone artefact (a composite
action, a zipapp, a pip package). It cannot import server code, so it carries its
own copy. `tests/test_features.py` pins this list against ml/features.py and fails
if either drifts, which is the same arrangement used for the isolated Lambda
handlers.

ORDER IS NOT SIGNIFICANT HERE — the client sends a JSON object keyed by name, so
only the *names* have to agree with the server. Order still matters on the server
side, where values become a positional model input.
"""

FEATURES = (
    "diff_size",
    "files_changed",
    "hour_of_day",
    "day_of_week",
    "recent_failure_rate",
    "test_pass_rate",
    "is_hotfix",
    "deployer_exp",
    "days_since_deploy",
    "build_time_delta",
)

# Derived server-side from build history; sending it is pointless and it is
# ignored once the tenant has history for this actor. Collected only as a hint
# for a never-before-seen actor.
SERVER_DERIVED = ("deployer_exp",)

# Branch names that mark an urgent, higher-risk change.
HOTFIX_PATTERNS = ("hotfix", "urgent", "emergency", "revert", "rollback", "patch")

__all__ = ["FEATURES", "SERVER_DERIVED", "HOTFIX_PATTERNS"]
