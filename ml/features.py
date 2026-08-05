"""
features.py — the single source of truth for the model's feature contract.

WHY THIS EXISTS
    The feature list was duplicated in seven places (app/scorer.py,
    app/retrain_cron.py, ml/train_base_model.py, ml/evaluate.py,
    ml/inject_test_data.py, lambda/retrain/handler.py, lambda/drift/handler.py).

    Training passed a named DataFrame; prediction passed a bare numpy array:

        X = np.array(features_list).reshape(1, -1)     # <- no names
        model.predict_proba(X)

    scikit-learn accepts that and warns, so **column order was the only thing
    keeping training and inference aligned**. Reorder one entry in one of the
    seven copies and every score silently becomes wrong — no exception, no
    failed test, just a model confidently reading test_pass_rate as diff_size.

    ORDER IS THE CONTRACT. Appending is safe. Reordering or renaming is a
    breaking change that invalidates every persisted model.

USAGE
    from features import FEATURES, to_frame, validate_model

    X = to_frame(values)                 # named, correctly ordered, 1 row
    model.predict_proba(X)               # sklearn now validates names for us

NOTE ON THE LAMBDA HANDLERS
    lambda/retrain and lambda/drift are packaged in isolation — their Dockerfile
    copies only handler.py — so they keep a literal copy of this list. That copy
    is pinned by tests/test_features.py, which fails if it ever drifts from this
    module. Do not "tidy" that duplication away without also changing the
    packaging.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

# --------------------------------------------------------------------------- #
# The contract
# --------------------------------------------------------------------------- #

# A tuple, not a list: this is a contract, not a working value. Order matters.
FEATURES: tuple[str, ...] = (
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

# Existing call sites (and app/routes/dashboard.py, which imports it from
# scorer) expect a mutable list under this name.
FEATURE_COLUMNS: list[str] = list(FEATURES)

N_FEATURES: int = len(FEATURES)

# Human-readable labels, kept beside the contract so a new feature cannot be
# added without one.
FEATURE_LABELS: dict[str, str] = {
    "diff_size": "Diff size (lines changed)",
    "files_changed": "Files changed",
    "hour_of_day": "Hour of day",
    "day_of_week": "Day of week",
    "recent_failure_rate": "Recent failure rate",
    "test_pass_rate": "Test pass rate",
    "is_hotfix": "Hotfix / emergency branch",
    "deployer_exp": "Deployer experience",
    "days_since_deploy": "Days since last deploy",
    "build_time_delta": "Build time delta",
}


class FeatureContractError(ValueError):
    """Raised when input or a model disagrees with the feature contract."""


# --------------------------------------------------------------------------- #
# Building model input
# --------------------------------------------------------------------------- #

def to_row(values: Sequence[Any] | Mapping[str, Any]) -> list[Any]:
    """
    Normalise input to a plain list in FEATURES order.

    Accepts a sequence (assumed already in order, length checked) or a mapping
    (reordered by name, which is the safe form for new callers).
    """
    if isinstance(values, Mapping):
        missing = [f for f in FEATURES if f not in values]
        if missing:
            raise FeatureContractError(f"missing features: {missing}")
        return [values[f] for f in FEATURES]

    row = list(values)
    if len(row) != N_FEATURES:
        raise FeatureContractError(
            f"expected {N_FEATURES} features in FEATURES order, got {len(row)}"
        )
    return row


def to_frame(values: Sequence[Any] | Mapping[str, Any] | Iterable[Sequence[Any]],
             many: bool = False):
    """
    Build a named, correctly ordered DataFrame for prediction.

    Passing a DataFrame rather than a bare array is the whole point: scikit-learn
    then compares column names against the model's `feature_names_in_` and raises
    on a mismatch, instead of silently trusting position.

    pandas is imported lazily so that importing this contract stays cheap for
    callers that only need FEATURES.
    """
    import pandas as pd

    if many:
        rows = [to_row(v) for v in values]  # type: ignore[arg-type]
    else:
        rows = [to_row(values)]  # type: ignore[arg-type]
    return pd.DataFrame(rows, columns=list(FEATURES))


# --------------------------------------------------------------------------- #
# Validating a loaded model
# --------------------------------------------------------------------------- #

def model_feature_names(model: Any) -> tuple[str, ...] | None:
    """The names a model was fitted with, or None if it was fitted positionally."""
    names = getattr(model, "feature_names_in_", None)
    return None if names is None else tuple(str(n) for n in names)


def validate_model(model: Any, *, strict: bool = True) -> tuple[bool, str]:
    """
    Check a loaded model against the contract.

    Called once per model load (not per request), so it costs nothing on the hot
    path while still catching a stale artefact before it can serve a single
    wrong score.

    Returns (ok, message). With strict=True a real mismatch raises instead.
    """
    n_in = getattr(model, "n_features_in_", None)
    if n_in is not None and int(n_in) != N_FEATURES:
        msg = f"model expects {n_in} features, contract defines {N_FEATURES}"
        if strict:
            raise FeatureContractError(msg)
        return False, msg

    names = model_feature_names(model)
    if names is None:
        # Fitted on a bare array. Nothing to compare, so order is unverifiable —
        # worth surfacing, but not fatal: older artefacts are like this.
        return True, "model was fitted without feature names; order unverified"

    if names != FEATURES:
        extra = [n for n in names if n not in FEATURES]
        missing = [f for f in FEATURES if f not in names]
        if extra or missing:
            msg = f"feature set mismatch (extra={extra}, missing={missing})"
        else:
            first = next(
                i for i, (a, b) in enumerate(zip(FEATURES, names)) if a != b
            )
            msg = (
                f"feature ORDER mismatch at index {first}: "
                f"contract={FEATURES[first]!r} model={names[first]!r}"
            )
        if strict:
            raise FeatureContractError(msg)
        return False, msg

    return True, "ok"
