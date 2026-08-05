"""
labels.py — where a label came from, and how much to trust it.

WHY THIS EXISTS
    A label is the only ground truth SafeShip ever gets, and not all labels are
    equally good evidence. "Sentinel watched production for two minutes and the
    error rate tripled" and "the pipeline went red somewhere" are both label=1,
    but only the first is actually about the deploy.

    Before this module, provenance was an ad-hoc string compared against a
    hardcoded list in one line of /log:

        row["sample_weight"] = 1.0 if label_src in ["failure", "safe"] else 0.7

    Two integrations disagreed about which string to send. The dashboard's Groovy
    sent "safe" for a good deploy; safeship_ci sends "success". Only one was on
    that list, so **every successful deploy logged by safeship_ci was silently
    weighted 0.7** — the most common label in the dataset, down-weighted by a
    typo, invisibly. That is exactly the class of bug that makes a model quietly
    worse and leaves no trace.

THE ONE SUBTLETY WORTH READING
    Labelling from CI status is tempting and slightly wrong. The model predicts
    "will this deploy cause a problem", so:

      - a pipeline that fails *before* the deploy step never deployed at all.
        That is not a deploy failure and should not be labelled 1 — it is not a
        data point about deployment risk, it is a data point about tests.
      - a pipeline that goes green while production breaks is the single most
        valuable row in the dataset, and CI status will never tell you about it.

    So Sentinel's verdict — did the service actually degrade after the deploy —
    is the trustworthy source, and CI status is a weaker proxy that is only
    meaningful at or after the deploy step. The weights below encode that.
"""
from __future__ import annotations

# ── the taxonomy ─────────────────────────────────────────────────────────────
#
# Weight is how much a row counts during retraining. It is not a probability;
# it is "how much would I bet this label describes what actually happened".

#: Directly observed outcomes. Something watched production and reported back.
OBSERVED = {
    # Sentinel probed the service after deploy and it regressed. The strongest
    # signal available: measured, attributable, and about the deploy itself.
    "sentinel_degraded": 1.0,
    # Sentinel probed and it stayed healthy through the window.
    "sentinel_healthy": 1.0,
    # Someone or something rolled the deploy back. Unambiguous.
    "rollback": 1.0,
}

#: A human said so. Trusted, but people label the memorable failures and forget
#: the uneventful successes, so this carries a mild survivorship bias.
REPORTED = {
    "manual": 0.9,
}

#: Derived from pipeline status. Weaker: see the note in the module docstring.
INFERRED = {
    # The deploy step itself failed. Reasonable evidence.
    "ci_failure": 0.7,
    # The pipeline finished green. Says nothing about production health, only
    # that nothing in the pipeline noticed.
    "ci_success": 0.6,
}

#: Nobody confirmed anything; we are assuming no news is good news after a
#: quiet period. Worth having — it is the only way to get enough negatives
#: without waiting a year — but it should never outvote an observation.
ASSUMED = {
    "assumed_ok": 0.4,
}

#: Strings earlier integrations sent, kept working so old rows and old pipelines
#: do not silently change weight. Mapped to their modern equivalent.
LEGACY_ALIASES = {
    "failure": "ci_failure",
    "success": "ci_success",
    "safe": "ci_success",
    "risky": "ci_failure",
    "seed": "synthetic",
    "synthetic": "synthetic",
}

WEIGHTS = {}
WEIGHTS.update(OBSERVED)
WEIGHTS.update(REPORTED)
WEIGHTS.update(INFERRED)
WEIGHTS.update(ASSUMED)
#: Bootstrap rows from ml/generate_synthetic.py. Kept at 1.0 so regenerating the
#: base model is unaffected, but real data should outweigh them over time.
WEIGHTS["synthetic"] = 1.0

#: An unrecognised source is trusted less than anything named, but not ignored:
#: a caller inventing a string is more likely to be a new integration than an
#: attack, and dropping the row loses real information.
DEFAULT_WEIGHT = 0.5
DEFAULT_SOURCE = "manual"


def normalise(source):
    """
    Canonical source name for whatever a caller sent.

    Unknown strings pass through unchanged rather than being coerced, so they
    show up in the data as themselves and can be added here deliberately.
    """
    name = (source or "").strip().lower()
    if not name:
        return DEFAULT_SOURCE
    return LEGACY_ALIASES.get(name, name)


def weight_for(source):
    """How much a row from this source counts during retraining."""
    return WEIGHTS.get(normalise(source), DEFAULT_WEIGHT)


def is_observed(source):
    """True when something actually watched production, rather than inferring."""
    return normalise(source) in OBSERVED


def describe(source):
    """(canonical_source, weight, observed?) — what /log echoes back."""
    canonical = normalise(source)
    return canonical, weight_for(canonical), canonical in OBSERVED


__all__ = ["WEIGHTS", "OBSERVED", "REPORTED", "INFERRED", "ASSUMED",
           "LEGACY_ALIASES", "DEFAULT_WEIGHT", "DEFAULT_SOURCE",
           "normalise", "weight_for", "is_observed", "describe"]
