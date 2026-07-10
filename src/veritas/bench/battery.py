"""The VERITAS validity battery: six tests + the :func:`run_bench` orchestrator.

Each test takes a metric and a :class:`CorpusBundle` (plus, for transfer, the
other corpora) and returns a :class:`TestResult` with a PASS/WARN/FAIL/N-A
verdict, a one-line headline carrying the load-bearing number, and the raw
stats. :func:`run_bench` assembles them into a :class:`ReportCard`.
"""
from __future__ import annotations

import numpy as np

from src.veritas.bench.data import CorpusBundle, SpeakerView
from src.veritas.bench.metric import VeritasMetric, check_disjoint_scores
from src.veritas.bench.probe_common import loco_transfer_predict
from src.veritas.bench.report import ReportCard, TestResult
from src.veritas.bench import stats as st
from src.veritas.bench import thresholds as th


def _aligned(views: list[SpeakerView], scores: dict[str, float]):
    sev = np.array([v.severity for v in views], dtype=float)
    sc = np.array([scores.get(v.speaker_id, np.nan) for v in views], dtype=float)
    return sev, sc


# ---------------------------------------------------------------------------
# 1. predictive validity
# ---------------------------------------------------------------------------
def test_predictive_validity(metric: VeritasMetric, bundle: CorpusBundle,
                             scores: dict[str, float] | None = None) -> TestResult:
    if scores is None:
        scores = metric.predictive_scores(bundle.views, feature_view="full")
    check_disjoint_scores(scores, bundle.views)
    sev, sc = _aligned(bundle.views, scores)
    rho, p, n = st.speaker_spearman(sev, sc)
    outcome = th.classify_predictive(rho, p)
    return TestResult(
        name="predictive_validity",
        outcome=outcome.value,
        headline=f"within-corpus LOSO Spearman rho={rho:+.3f} (p={p:.3g}, n={n})",
        stats={"rho": rho, "p": p, "n_speakers": n, "eval": "leave-one-speaker-out"},
    )


# ---------------------------------------------------------------------------
# 2. leakage (naive vs leave-one-control-out reference)
# ---------------------------------------------------------------------------
def test_leakage_loco(metric: VeritasMetric, bundle: CorpusBundle) -> TestResult:
    loco = metric.loco_scores(bundle.views, feature_view="full")
    if loco is None:
        return TestResult(
            name="leakage_loco", outcome=th.Outcome.NA.value,
            headline="N/A - metric is not reference-based (no control reference to leak)",
            stats={"applicable": False, "metric_kind": metric.kind},
        )
    naive = metric.naive_scores(bundle.views, feature_view="full")
    check_disjoint_scores(loco, bundle.views)
    check_disjoint_scores(naive, bundle.views)
    sev, sc_naive = _aligned(bundle.views, naive)
    _, sc_loco = _aligned(bundle.views, loco)
    rho_naive, p_naive, _ = st.speaker_spearman(sev, sc_naive)
    rho_loco, p_loco, n = st.speaker_spearman(sev, sc_loco)
    delta = rho_naive - rho_loco
    outcome = th.classify_leakage(delta)
    return TestResult(
        name="leakage_loco", outcome=outcome.value,
        headline=f"naive rho={rho_naive:+.3f} -> LOCO rho={rho_loco:+.3f}; "
                 f"leakage inflation delta={delta:+.3f}",
        stats={"naive_rho": rho_naive, "naive_p": p_naive,
               "loco_rho": rho_loco, "loco_p": p_loco,
               "delta_rho": delta, "n_speakers": n},
    )


# ---------------------------------------------------------------------------
# 3. silence confound
# ---------------------------------------------------------------------------
def test_silence_confound(metric: VeritasMetric, bundle: CorpusBundle,
                          naive_full: dict[str, float] | None = None) -> TestResult:
    sil_fracs = [v.silence_fraction for v in bundle.views if np.isfinite(v.silence_fraction)]
    silence_fraction = float(np.mean(sil_fracs)) if sil_fracs else float("nan")
    if not bundle.silence_available or not sil_fracs:
        return TestResult(
            name="silence_confound", outcome=th.Outcome.NA.value,
            headline="N/A - no silence/VAD signal available for this corpus",
            stats={"applicable": False, "silence_fraction": silence_fraction},
        )
    # reference metrics trim frames themselves; pooled-feature metrics need the
    # bundle's speech-only pooled feature to have been computed.
    if metric.kind != "reference" and not bundle.speech_pool_available:
        return TestResult(
            name="silence_confound", outcome=th.Outcome.NA.value,
            headline="N/A - speech-only pooled feature not computed for this bundle "
                     f"(silence mass={silence_fraction:.0%})",
            stats={"applicable": False, "silence_fraction": silence_fraction},
        )
    if naive_full is None:
        naive_full = metric.naive_scores(bundle.views, feature_view="full")
    naive_speech = metric.naive_scores(bundle.views, feature_view="speech_only")
    sev, sc_full = _aligned(bundle.views, naive_full)
    _, sc_speech = _aligned(bundle.views, naive_speech)
    rho_full, p_full, n = st.speaker_spearman(sev, sc_full)
    rho_speech, p_speech, _ = st.speaker_spearman(sev, sc_speech)
    retention = st.retention(rho_speech, rho_full)
    outcome = th.classify_silence(retention, silence_fraction)
    ret_txt = f"{retention:.0%}" if np.isfinite(retention) else "n/a"
    return TestResult(
        name="silence_confound", outcome=outcome.value,
        headline=f"full rho={rho_full:+.3f} -> speech-only rho={rho_speech:+.3f} "
                 f"(retains {ret_txt}); silence mass={silence_fraction:.0%}",
        stats={"full_rho": rho_full, "full_p": p_full,
               "speech_only_rho": rho_speech, "speech_only_p": p_speech,
               "retention": retention, "silence_fraction": silence_fraction,
               "n_speakers": n, "protocol": "metric-as-published (naive) on full vs "
                                            "MFA speech-span frames"},
    )


# ---------------------------------------------------------------------------
# 4. energy confound
# ---------------------------------------------------------------------------
def test_energy_confound(metric: VeritasMetric, bundle: CorpusBundle,
                         scores: dict[str, float] | None = None) -> TestResult:
    energies = [v.energy for v in bundle.views if np.isfinite(v.energy)]
    if len(energies) < len(bundle.views) or not energies:
        return TestResult(
            name="energy_confound", outcome=th.Outcome.NA.value,
            headline="N/A - mean speech energy unavailable for this corpus",
            stats={"applicable": False},
        )
    if scores is None:
        scores = metric.predictive_scores(bundle.views, feature_view="full")
    sev = np.array([v.severity for v in bundle.views], dtype=float)
    sc = np.array([scores.get(v.speaker_id, np.nan) for v in bundle.views], dtype=float)
    energy = np.array([v.energy for v in bundle.views], dtype=float)
    raw_rho, raw_p, n = st.speaker_spearman(sev, sc)
    part_rho, part_p, _ = st.partial_spearman(sev, sc, energy)
    retention = st.retention(part_rho, raw_rho)
    outcome = th.classify_energy(retention)
    ret_txt = f"{retention:.0%}" if np.isfinite(retention) else "n/a"
    return TestResult(
        name="energy_confound", outcome=outcome.value,
        headline=f"raw rho={raw_rho:+.3f} -> energy-partial rho={part_rho:+.3f} "
                 f"(retains {ret_txt})",
        stats={"raw_rho": raw_rho, "raw_p": raw_p,
               "partial_rho": part_rho, "partial_p": part_p,
               "retention": retention, "control": "mean speech log-RMS energy",
               "n_speakers": n},
    )


# ---------------------------------------------------------------------------
# 5. cross-corpus transfer
# ---------------------------------------------------------------------------
def test_cross_corpus_transfer(metric: VeritasMetric, bundle: CorpusBundle,
                               other_bundles: list[CorpusBundle]) -> TestResult:
    if not metric.has_representation:
        return TestResult(
            name="cross_corpus_transfer", outcome=th.Outcome.NA.value,
            headline="N/A - metric exposes no frozen-SSL representation to transfer",
            stats={"applicable": False},
        )
    others = [b for b in other_bundles if b.name != bundle.name]
    if not others:
        return TestResult(
            name="cross_corpus_transfer", outcome=th.Outcome.NA.value,
            headline="N/A - needs >=1 other corpus on disk to train a transfer probe",
            stats={"applicable": False, "n_other_corpora": 0},
        )
    bundles = {bundle.name: bundle, **{b.name: b for b in others}}
    preds = loco_transfer_predict(bundles, held=bundle.name)
    sev, sc = _aligned(bundle.views, preds)
    rho, p, n = st.speaker_spearman(sev, sc)
    outcome = th.classify_transfer(rho, p)
    train_names = [b.name for b in others]
    return TestResult(
        name="cross_corpus_transfer", outcome=outcome.value,
        headline=f"train on {train_names} -> test {bundle.name}: "
                 f"transfer rho={rho:+.3f} (p={p:.3g}, n={n})",
        stats={"transfer_rho": rho, "p": p, "n_speakers": n,
               "train_corpora": train_names,
               "representation": metric.representation_desc},
    )


# ---------------------------------------------------------------------------
# 6. leakage-by-layer speaker-identity AUC (optional)
# ---------------------------------------------------------------------------
def test_leakage_by_layer(metric: VeritasMetric, bundle: CorpusBundle,
                          seed: int = 42) -> TestResult:
    if not metric.has_representation:
        return TestResult(
            name="leakage_by_layer", outcome=th.Outcome.NA.value,
            headline="N/A - metric exposes no representation to probe for identity",
            stats={"applicable": False},
        )
    X = bundle.X_utt
    spk = bundle.meta["speaker_id"].to_numpy()
    auc, n_spk = st.speaker_identity_auc(X, spk, seed=seed)
    outcome = th.classify_identity_auc(auc)
    return TestResult(
        name="leakage_by_layer", outcome=outcome.value,
        headline=f"speaker-identity AUC={auc:.3f} over {n_spk} speakers (chance 0.5)",
        stats={"speaker_identity_auc": auc, "n_speakers": n_spk,
               "representation": metric.representation_desc,
               "interpretation": "higher AUC = representation encodes identity "
                                 "-> leakage risk for reference metrics"},
    )


# ---------------------------------------------------------------------------
# orchestrator
# ---------------------------------------------------------------------------
_ALL_TESTS = [
    "predictive_validity", "leakage_loco", "silence_confound",
    "energy_confound", "cross_corpus_transfer", "leakage_by_layer",
]


def run_bench(
    metric: VeritasMetric,
    bundle: CorpusBundle,
    other_bundles: list[CorpusBundle] | None = None,
    tests: list[str] | None = None,
    repr_bundle: CorpusBundle | None = None,
    seed: int = 42,
    log=print,
) -> ReportCard:
    """Run the validity battery for one metric on one corpus -> a ReportCard.

    ``repr_bundle`` (default = ``bundle``) supplies the utterance features used
    by the representation-based tests (cross-corpus transfer, identity AUC). It
    lets the score-based tests run on a full-utterance bundle while the transfer
    probe uses a per-speaker-capped bundle matching the committed protocol.
    """
    other_bundles = other_bundles or []
    repr_bundle = repr_bundle or bundle
    run = tests or _ALL_TESTS
    card = ReportCard(
        metric_name=metric.name, metric_kind=metric.kind,
        corpus=bundle.name, n_speakers=bundle.n_speakers,
        notes=list(bundle.notes),
    )

    # shared computations (avoid recompute; ATD memoizes internally anyway)
    pred_full = None
    naive_full = None
    if "predictive_validity" in run or "energy_confound" in run:
        pred_full = metric.predictive_scores(bundle.views, feature_view="full")
    if "silence_confound" in run:
        naive_full = metric.naive_scores(bundle.views, feature_view="full")

    dispatch = {
        "predictive_validity": lambda: test_predictive_validity(metric, bundle, pred_full),
        "leakage_loco": lambda: test_leakage_loco(metric, bundle),
        "silence_confound": lambda: test_silence_confound(metric, bundle, naive_full),
        "energy_confound": lambda: test_energy_confound(metric, bundle, pred_full),
        "cross_corpus_transfer": lambda: test_cross_corpus_transfer(metric, repr_bundle, other_bundles),
        "leakage_by_layer": lambda: test_leakage_by_layer(metric, repr_bundle, seed),
    }
    for tname in _ALL_TESTS:
        if tname not in run:
            continue
        log(f"  [bench] {metric.name}: {tname} ...")
        card.results.append(dispatch[tname]())
        log(f"          -> {card.results[-1].outcome}: {card.results[-1].headline}")
    return card
