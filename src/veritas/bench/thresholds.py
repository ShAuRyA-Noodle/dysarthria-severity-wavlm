"""Documented PASS / WARN / FAIL thresholds for the VERITAS validity battery.

Every threshold is a policy choice, stated here in ONE place so a report card is
interpretable and auditable. They encode the project's own hard-won findings
(leakage, silence, energy, cross-corpus) as reusable acceptance criteria for
*any* frozen-SSL clinical-speech metric.

Rationale per test lives next to each constant. `classify_*` turns a raw stat
into an :class:`Outcome`. Significance uses alpha = 0.05 throughout.
"""
from __future__ import annotations

from enum import Enum

import numpy as np

ALPHA = 0.05


class Outcome(str, Enum):
    """A single test verdict. Ordered worst-to-best for aggregation."""

    FAIL = "FAIL"
    WARN = "WARN"
    PASS = "PASS"
    NA = "N/A"  # test not applicable to this metric (e.g. leakage on a probe)

    @property
    def rank(self) -> int:
        return {"FAIL": 0, "WARN": 1, "PASS": 2, "N/A": 3}[self.value]


# --- Test 1: predictive validity (within-corpus LOSO Spearman) ----------------
# A metric that cannot recover severity *within* a single corpus (speaker-disjoint)
# is not measuring the disorder at all. PASS requires a moderate, significant
# rank correlation; WARN is weak-but-present; FAIL is no usable signal.
PREDICTIVE_PASS_RHO = 0.50
PREDICTIVE_WARN_RHO = 0.30


def classify_predictive(rho: float, p: float) -> Outcome:
    if not np.isfinite(rho):
        return Outcome.NA
    sig = np.isfinite(p) and p < ALPHA
    if rho >= PREDICTIVE_PASS_RHO and sig:
        return Outcome.PASS
    if rho >= PREDICTIVE_WARN_RHO:
        return Outcome.WARN
    return Outcome.FAIL


# --- Test 2: speaker-identity leakage (naive vs LOCO reference) ----------------
# delta = naive_rho - loco_rho. A reference that contains the test speaker's own
# data inflates the score. delta measures how much correlation is pure leakage.
LEAKAGE_PASS_DELTA = 0.05   # leakage-robust
LEAKAGE_WARN_DELTA = 0.15   # noticeable inflation


def classify_leakage(delta: float) -> Outcome:
    if not np.isfinite(delta):
        return Outcome.NA
    if delta <= LEAKAGE_PASS_DELTA:
        return Outcome.PASS
    if delta <= LEAKAGE_WARN_DELTA:
        return Outcome.WARN
    return Outcome.FAIL


# --- Test 3: silence confound -------------------------------------------------
# Two signals: (a) how much of the representation is silence (silence_fraction),
# (b) whether the severity correlation survives silence removal (retention =
# speech_only_rho / full_rho). A metric that lives in the silence and collapses
# when silence is stripped is measuring pausing, not articulation.
SILENCE_MASS_WARN = 0.50    # majority of frames are silence -> at least WARN
SILENCE_RETENTION_PASS = 0.70
SILENCE_RETENTION_WARN = 0.40


def classify_silence(retention_frac: float, silence_fraction: float) -> Outcome:
    if not np.isfinite(retention_frac):
        return Outcome.NA
    # correlation-survival verdict
    if retention_frac >= SILENCE_RETENTION_PASS:
        surv = Outcome.PASS
    elif retention_frac >= SILENCE_RETENTION_WARN:
        surv = Outcome.WARN
    else:
        surv = Outcome.FAIL
    # a majority-silence representation cannot fully PASS even if rho survives
    if surv == Outcome.PASS and np.isfinite(silence_fraction) and silence_fraction >= SILENCE_MASS_WARN:
        return Outcome.WARN
    return surv


# --- Test 4: energy confound (partial Spearman controlling mean speech energy) -
# retention = partial_rho / raw_rho. If controlling for loudness/energy erases
# the correlation, the metric is an energy artifact.
ENERGY_RETENTION_PASS = 0.70
ENERGY_RETENTION_WARN = 0.40


def classify_energy(retention_frac: float) -> Outcome:
    if not np.isfinite(retention_frac):
        return Outcome.NA
    if retention_frac >= ENERGY_RETENTION_PASS:
        return Outcome.PASS
    if retention_frac >= ENERGY_RETENTION_WARN:
        return Outcome.WARN
    return Outcome.FAIL


# --- Test 5: cross-corpus transfer --------------------------------------------
# Train on OTHER corpora, test held-out. Single-corpus success that does not
# transfer is overfitting to corpus idiosyncrasy. PASS requires a moderate,
# significant transfer correlation.
TRANSFER_PASS_RHO = 0.50
TRANSFER_WARN_RHO = 0.30


def classify_transfer(rho: float, p: float) -> Outcome:
    if not np.isfinite(rho):
        return Outcome.NA
    sig = np.isfinite(p) and p < ALPHA
    if rho >= TRANSFER_PASS_RHO and sig:
        return Outcome.PASS
    if rho >= TRANSFER_WARN_RHO and sig:
        return Outcome.WARN
    return Outcome.FAIL


# --- Test 6 (optional): leakage-by-layer speaker-identity AUC ------------------
# Speaker-classification AUC of the representation the metric uses. High AUC =
# the features strongly encode identity -> leakage risk for reference metrics.
# This is a RISK indicator (chance = 0.5).
IDENTITY_AUC_PASS = 0.70
IDENTITY_AUC_WARN = 0.85


def classify_identity_auc(auc: float) -> Outcome:
    if not np.isfinite(auc):
        return Outcome.NA
    if auc <= IDENTITY_AUC_PASS:
        return Outcome.PASS
    if auc <= IDENTITY_AUC_WARN:
        return Outcome.WARN
    return Outcome.FAIL


def worst(outcomes: list[Outcome]) -> Outcome:
    """Aggregate: the worst applicable outcome (N/A ignored unless all N/A)."""
    applicable = [o for o in outcomes if o != Outcome.NA]
    if not applicable:
        return Outcome.NA
    return min(applicable, key=lambda o: o.rank)


THRESHOLD_DOC = {
    "alpha": ALPHA,
    "predictive_validity": {
        "PASS": f"rho >= {PREDICTIVE_PASS_RHO} and p < {ALPHA}",
        "WARN": f"rho >= {PREDICTIVE_WARN_RHO}",
        "FAIL": f"rho < {PREDICTIVE_WARN_RHO}",
        "stat": "within-corpus leave-one-speaker-out Spearman rho (metric vs severity)",
    },
    "leakage_loco": {
        "PASS": f"naive-minus-LOCO delta <= {LEAKAGE_PASS_DELTA}",
        "WARN": f"delta <= {LEAKAGE_WARN_DELTA}",
        "FAIL": f"delta > {LEAKAGE_WARN_DELTA}",
        "NA": "metric is not reference-based",
        "stat": "rho(naive reference) - rho(leave-one-control-out reference)",
    },
    "silence_confound": {
        "PASS": f"speech-only retains >= {SILENCE_RETENTION_PASS} of full rho "
                f"AND silence_fraction < {SILENCE_MASS_WARN}",
        "WARN": f"retains >= {SILENCE_RETENTION_WARN}, or majority-silence representation",
        "FAIL": f"speech-only retains < {SILENCE_RETENTION_WARN} of full rho",
        "stat": "retention = rho(speech-only) / rho(full); plus mean silence frame fraction",
    },
    "energy_confound": {
        "PASS": f"partial rho retains >= {ENERGY_RETENTION_PASS} of raw rho",
        "WARN": f"retains >= {ENERGY_RETENTION_WARN}",
        "FAIL": f"retains < {ENERGY_RETENTION_WARN}",
        "stat": "partial Spearman(metric, severity | mean speech energy) vs raw Spearman",
    },
    "cross_corpus_transfer": {
        "PASS": f"transfer rho >= {TRANSFER_PASS_RHO} and p < {ALPHA}",
        "WARN": f"rho >= {TRANSFER_WARN_RHO} and p < {ALPHA}",
        "FAIL": f"rho < {TRANSFER_WARN_RHO} or not significant",
        "NA": "fewer than 2 corpora available, or metric exposes no representation",
        "stat": "train probe on other corpora -> test held-out corpus; speaker Spearman",
    },
    "leakage_by_layer": {
        "PASS": f"speaker-identity AUC <= {IDENTITY_AUC_PASS}",
        "WARN": f"AUC <= {IDENTITY_AUC_WARN}",
        "FAIL": f"AUC > {IDENTITY_AUC_WARN} (representation strongly encodes identity)",
        "stat": "macro one-vs-rest speaker-classification AUC of the metric's representation",
    },
}
