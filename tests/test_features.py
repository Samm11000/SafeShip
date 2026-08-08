"""
Guards the feature contract.

The bug these tests exist to prevent: training passed a named DataFrame while
prediction passed a bare numpy array, so scikit-learn only checked the column
*count* and trusted position. The feature list was duplicated in seven files.
Reordering one entry in one copy would have produced confidently wrong scores
with no exception, no warning that mattered, and no failing test.

These tests make that failure loud.
"""
from __future__ import annotations

import os
import pathlib
import re
import sys

import numpy as np
import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
for sub in ("ml", "app"):
    p = str(REPO / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from features import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURE_LABELS,
    FEATURES,
    N_FEATURES,
    FeatureContractError,
    to_frame,
    to_row,
    validate_model,
)


# ── The contract itself ──────────────────────────────────────────────────────

def test_features_is_an_immutable_ordered_contract():
    assert isinstance(FEATURES, tuple), "FEATURES must be a tuple — order is a contract"
    assert len(FEATURES) == N_FEATURES == 10
    assert len(set(FEATURES)) == len(FEATURES), "duplicate feature names"


def test_every_feature_has_a_label():
    """A new feature cannot be added without a human-readable label."""
    assert set(FEATURE_LABELS) == set(FEATURES)


def test_feature_columns_mirrors_features():
    assert FEATURE_COLUMNS == list(FEATURES)


# ── Every duplicated copy must match, byte for byte, in order ────────────────

# Three places legitimately keep a literal copy, because each is packaged in
# isolation and cannot import ml/features.py:
#   - the two Lambda handlers (their Dockerfile copies only handler.py)
#   - safeship_ci/contract.py (ships to customers as a zipapp / composite action)
# Copying is allowed. Drifting is not — that is what this test is for.
_DUPLICATES = [
    ("lambda/retrain/handler.py", "FEATURE_COLUMNS"),
    ("lambda/drift/handler.py", "FEATURE_COLUMNS"),
    ("safeship_ci/contract.py", "FEATURES"),
]


@pytest.mark.parametrize("relpath,varname", _DUPLICATES)
def test_duplicated_feature_lists_have_not_drifted(relpath, varname):
    src = (REPO / relpath).read_text()
    # Accepts a list or a tuple literal — safeship_ci uses a tuple.
    m = re.search(rf"^{varname}\s*=\s*[\[(](.*?)[\])]", src, re.S | re.M)
    assert m, f"{relpath}: could not find {varname}"
    found = tuple(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert found == FEATURES, (
        f"{relpath} has drifted from ml/features.py.\n"
        f"  contract: {FEATURES}\n"
        f"  {relpath}: {found}\n"
        "Order is the contract for anything that builds a positional model input, "
        "and for the client copy it is what keeps the two files diffable. "
        "Fix the copy, or change both deliberately."
    )


def test_no_new_hardcoded_feature_lists_appear():
    """
    Anything that re-declares the full list is a new copy waiting to drift.
    Only ml/features.py and the two isolated Lambda handlers may do so.
    """
    # Every entry here must also be pinned by _DUPLICATES above, so an allowed
    # copy is still not a free copy.
    allowed = {"ml/features.py"} | {relpath for relpath, _ in _DUPLICATES}
    offenders = []
    for path in REPO.rglob("*.py"):
        rel = path.relative_to(REPO).as_posix()
        if rel in allowed or ".venv" in rel or "ansible/files" in rel:
            continue
        # Tests legitimately name features when asserting on them.
        if rel.startswith("tests/"):
            continue
        text = path.read_text(errors="ignore")
        # A list *or tuple* literal holding the first and last feature, in order.
        if re.search(r'"diff_size".*?"build_time_delta"', text, re.S):
            block = re.search(r"[\[(]\s*\"diff_size\".*?\"build_time_delta\",?\s*[\])]",
                              text, re.S)
            # A literal contains only names, commas, whitespace and comments.
            # Assigning columns one at a time — out["diff_size"] = ... through
            # out["build_time_delta"] = ... — also opens with "[" and closes
            # with "]", so the bracket match alone flags files that correctly
            # import the contract and merely have to name the columns to build
            # a frame. An "=" between the first and last name means it is code,
            # not a re-declared list.
            if block and "=" not in block.group(0):
                offenders.append(rel)
    assert not offenders, (
        "these files re-declare the feature list instead of importing it "
        f"from ml/features.py: {offenders}"
    )


# ── Input construction ───────────────────────────────────────────────────────

def test_to_frame_produces_named_columns_in_contract_order():
    df = to_frame(list(range(N_FEATURES)))
    assert list(df.columns) == list(FEATURES)
    assert df.shape == (1, N_FEATURES)


def test_to_row_accepts_a_mapping_and_reorders_by_name():
    """A dict is the safe form: order of the caller's keys cannot matter."""
    shuffled = {f: i for i, f in enumerate(reversed(FEATURES))}
    row = to_row(shuffled)
    assert row == [shuffled[f] for f in FEATURES]


def test_to_row_rejects_wrong_length():
    with pytest.raises(FeatureContractError):
        to_row([1, 2, 3])


def test_to_row_rejects_missing_keys():
    with pytest.raises(FeatureContractError):
        to_row({"diff_size": 1})


# ── Model validation ─────────────────────────────────────────────────────────

class _FakeModel:
    def __init__(self, names=None, n=None):
        if names is not None:
            self.feature_names_in_ = np.array(list(names))
        if n is not None:
            self.n_features_in_ = n


def test_validate_model_accepts_a_matching_model():
    ok, msg = validate_model(_FakeModel(names=FEATURES, n=N_FEATURES))
    assert ok and msg == "ok"


def test_validate_model_catches_reordered_features():
    """The exact bug: same features, swapped order."""
    swapped = list(FEATURES)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(FeatureContractError, match="ORDER mismatch"):
        validate_model(_FakeModel(names=swapped, n=N_FEATURES))


def test_validate_model_catches_a_renamed_feature():
    renamed = ("diff_sizes",) + FEATURES[1:]
    with pytest.raises(FeatureContractError, match="feature set mismatch"):
        validate_model(_FakeModel(names=renamed, n=N_FEATURES))


def test_validate_model_catches_wrong_feature_count():
    with pytest.raises(FeatureContractError, match="expects 9 features"):
        validate_model(_FakeModel(names=FEATURES[:9], n=9))


def test_validate_model_tolerates_a_positionally_fitted_model():
    """Older artefacts have no feature_names_in_; that is unverifiable, not fatal."""
    ok, msg = validate_model(_FakeModel(n=N_FEATURES))
    assert ok and "unverified" in msg


# ── End to end: the shipped base model honours the contract ──────────────────

def test_shipped_base_model_matches_the_contract():
    joblib = pytest.importorskip("joblib")
    path = REPO / "ml" / "data" / "base_model.pkl"
    if not path.exists():
        pytest.skip("base_model.pkl not present")
    ok, msg = validate_model(joblib.load(path), strict=False)
    assert ok, f"the committed base model violates the feature contract: {msg}"


def test_prediction_with_named_frame_matches_positional_order():
    """
    Proves names and position currently agree — and that prediction goes through
    the named path, so sklearn would raise if they ever stopped agreeing.
    """
    joblib = pytest.importorskip("joblib")
    path = REPO / "ml" / "data" / "base_model.pkl"
    if not path.exists():
        pytest.skip("base_model.pkl not present")
    model = joblib.load(path)
    values = [500, 9, 3, 5, 0.4, 0.7, 1, 2, 0.0, 0.2]

    named = model.predict_proba(to_frame(values))[0][1]
    positional = model.predict_proba(np.array(values).reshape(1, -1))[0][1]
    assert named == pytest.approx(positional), (
        "named and positional prediction disagree — the model's column order "
        "does not match ml/features.py"
    )
