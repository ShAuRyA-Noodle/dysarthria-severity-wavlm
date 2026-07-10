"""Real, small, deterministic tests for VERITAS-Bench (seed 42).

Everything runs on a TINY REAL slice of TORGO (no synthetic data): the battery
executes end-to-end, the report-card schema validates, and the shipped metric
adapters behave per their kind. A toy user-supplied metric proves the plug-in
path works for code outside the repo.

Run:
  C:/Users/Dell/.conda/envs/dysarthria/python.exe -m pytest tests/test_veritas_bench.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import src.veritas as _veritas  # noqa: E402,F401  (offline HF-cache bootstrap)
from src.seed_utils import set_global_seed  # noqa: E402
from src.veritas.bench import (  # noqa: E402
    build_corpus_bundle,
    render_table,
    run_bench,
)
from src.veritas.bench.metric import VeritasMetric, check_disjoint_scores  # noqa: E402
from src.veritas.bench.metrics import ATDReferenceDistance, PooledWavLMProbe  # noqa: E402
from src.veritas.bench.report import (  # noqa: E402
    ReportCard,
    TestResult,
    build_report,
    validate_card,
    validate_report,
    write_report,
)
from src.veritas.bench import stats as st  # noqa: E402
from src.veritas.bench import thresholds as th  # noqa: E402
from src.veritas.corpora import torgo  # noqa: E402

SEED = 42
# Tiny speaker-disjoint slice: 2 controls + 2 dysarthric (graded 0/1/2).
_SLICE = ["FC01", "MC01", "F01", "M03"]
_CAP = 6
_VALID_OUTCOMES = {"PASS", "WARN", "FAIL", "N/A"}


# --- a toy user-supplied metric (proves the plug-in path) --------------------
class ToyNormMetric(VeritasMetric):
    """Intrinsic per-speaker statistic: norm of the mean pooled feature.

    Not expected to be valid — it exists to exercise the battery for an
    arbitrary user metric (no control reference, exposes a representation).
    """

    name = "toy_mean_norm"
    kind = "intrinsic"
    has_representation = True
    representation_desc = "toy: L9 mean-pool"

    def predictive_scores(self, views, *, feature_view: str = "full"):
        out = {}
        for v in views:
            feats = v.pooled_full if feature_view == "full" else v.pooled_speech
            out[v.speaker_id] = float(np.linalg.norm(feats.mean(axis=0)))
        return out


@pytest.fixture(scope="module")
def tiny_bundle():
    set_global_seed(SEED)
    recs = [r for r in torgo.load() if r.speaker_id in _SLICE]
    assert recs, "TORGO slice not available on disk"
    return build_corpus_bundle(
        recs, device="cuda:0", with_silence=True, need_speech_pool=True,
        max_utts_per_speaker=_CAP, seed=SEED)


# --- stats -------------------------------------------------------------------
def test_speaker_spearman_basic():
    rho, p, n = st.speaker_spearman(np.array([0, 1, 2, 3.0]), np.array([1, 2, 3, 4.0]))
    assert rho == pytest.approx(1.0)
    assert n == 4


def test_partial_spearman_removes_confound():
    # y and x are correlated ONLY through z; controlling for z -> ~0.
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=200)
    x = z + 0.01 * rng.normal(size=200)
    y = z + 0.01 * rng.normal(size=200)
    raw, _, _ = st.speaker_spearman(y, x)
    part, _, _ = st.partial_spearman(y, x, z)
    assert raw > 0.9
    assert abs(part) < 0.3  # confound removed


def test_partial_spearman_keeps_direct_signal():
    rng = np.random.default_rng(SEED)
    z = rng.normal(size=200)
    x = rng.normal(size=200)
    y = x + 0.05 * rng.normal(size=200)  # y depends on x, not z
    part, _, _ = st.partial_spearman(y, x, z)
    assert part > 0.8


# --- thresholds --------------------------------------------------------------
def test_classifiers_boundaries():
    assert th.classify_predictive(0.8, 0.001) == th.Outcome.PASS
    assert th.classify_predictive(0.4, 0.5) == th.Outcome.WARN
    assert th.classify_predictive(0.1, 0.9) == th.Outcome.FAIL
    assert th.classify_leakage(0.0) == th.Outcome.PASS
    assert th.classify_leakage(0.089) == th.Outcome.WARN
    assert th.classify_leakage(0.30) == th.Outcome.FAIL
    assert th.classify_transfer(0.43, 0.11) == th.Outcome.FAIL  # n.s. -> FAIL
    assert th.classify_identity_auc(0.99) == th.Outcome.FAIL


def test_worst_aggregation():
    assert th.worst([th.Outcome.PASS, th.Outcome.WARN, th.Outcome.NA]) == th.Outcome.WARN
    assert th.worst([th.Outcome.NA, th.Outcome.NA]) == th.Outcome.NA


# --- metric contract ---------------------------------------------------------
def test_disjoint_check_raises(tiny_bundle):
    with pytest.raises(ValueError):
        check_disjoint_scores({"NOT_A_SPEAKER": 1.0}, tiny_bundle.views)


def test_probe_leakage_is_na(tiny_bundle):
    probe = PooledWavLMProbe()
    assert probe.loco_scores(tiny_bundle.views) is None


def test_atd_is_reference_kind():
    atd = ATDReferenceDistance(layer=9)
    assert atd.kind == "reference"
    assert atd.name == "atd_wavlm_l9"


# --- report schema -----------------------------------------------------------
def test_report_card_schema_validates():
    card = ReportCard("m", "intrinsic", "torgo", 4, results=[
        TestResult("predictive_validity", "PASS", "rho=0.8", {"rho": 0.8}),
        TestResult("leakage_loco", "N/A", "n/a", {"applicable": False}),
    ])
    d = card.to_dict()
    assert validate_card(d) == []
    assert d["overall"] == "PASS"


def test_bad_outcome_is_rejected():
    card = ReportCard("m", "intrinsic", "torgo", 4, results=[
        TestResult("predictive_validity", "GREAT", "x", {})])
    errs = validate_card(card.to_dict())
    assert any("bad outcome" in e for e in errs)


# --- end-to-end battery on a tiny real slice ---------------------------------
def test_toy_metric_end_to_end(tiny_bundle):
    """The full battery runs for an arbitrary user metric; card validates."""
    card = run_bench(ToyNormMetric(), tiny_bundle, other_bundles=[], seed=SEED, log=lambda *a: None)
    names = {r.name for r in card.results}
    assert names == {
        "predictive_validity", "leakage_loco", "silence_confound",
        "energy_confound", "cross_corpus_transfer", "leakage_by_layer",
    }
    for r in card.results:
        assert r.outcome in _VALID_OUTCOMES, (r.name, r.outcome)
    # intrinsic metric has no control reference -> leakage N/A
    leak = next(r for r in card.results if r.name == "leakage_loco")
    assert leak.outcome == "N/A"
    # single corpus -> cross-corpus transfer N/A
    xc = next(r for r in card.results if r.name == "cross_corpus_transfer")
    assert xc.outcome == "N/A"
    assert validate_card(card.to_dict()) == []


def test_atd_metric_end_to_end(tiny_bundle):
    """The reference-based ATD metric runs and enables the leakage test."""
    card = run_bench(ATDReferenceDistance(layer=9), tiny_bundle, other_bundles=[],
                     tests=["predictive_validity", "leakage_loco", "leakage_by_layer"],
                     seed=SEED, log=lambda *a: None)
    leak = next(r for r in card.results if r.name == "leakage_loco")
    assert leak.outcome != "N/A"                 # reference metric -> leakage applies
    assert "delta_rho" in leak.stats
    pred = next(r for r in card.results if r.name == "predictive_validity")
    assert np.isfinite(pred.stats["rho"])
    assert validate_card(card.to_dict()) == []


def test_determinism(tiny_bundle):
    """Same seed + same inputs -> identical scores."""
    m = ToyNormMetric()
    a = m.predictive_scores(tiny_bundle.views)
    b = m.predictive_scores(tiny_bundle.views)
    assert a == b
    probe = PooledWavLMProbe()
    p1 = probe.predictive_scores(tiny_bundle.views)
    p2 = probe.predictive_scores(tiny_bundle.views)
    assert p1 == pytest.approx(p2)


def test_full_report_roundtrip(tiny_bundle, tmp_path):
    card = run_bench(ToyNormMetric(), tiny_bundle, other_bundles=[], seed=SEED,
                     log=lambda *a: None)
    report = build_report([card], {"seed": SEED, "corpus": "torgo"})
    assert validate_report(report) == []
    out = tmp_path / "report.json"
    write_report(report, out)
    import json
    reloaded = json.loads(out.read_text(encoding="utf-8"))
    assert validate_report(reloaded) == []
    # human-readable table renders without error
    assert "VERITAS-BENCH" in render_table([card])
