"""Coverage test: every input read by the paper-numbers consolidator must be
produced by some Step in reproduce_all.py (or be an explicitly-documented
orphan).

This guards the failure mode the audit found: ``reproduce_all.py`` used to
omit scripts (22..34, L2_*, V_*, rebuild_table1, 41, 42) whose outputs the
consolidator reads, so a fresh clone regenerated an incomplete
``paper_numbers.json``. If someone adds a new input to
``experiments/21_consolidate_paper_numbers.py`` without wiring its producer
into ``reproduce_all.py``, this test fails.

Run::

    pytest tests/test_reproduce_all_coverage.py -q
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import reproduce_all  # noqa: E402


# Authoritative input -> producing script map. Each consolidator input is
# produced by exactly one script; some are produced as a side artifact of a
# script whose Step.output_check names a *different* file (e.g. 06 writes both
# atd_scores.csv and speaker_atd.csv; 09 writes layer_analysis.csv and
# best_layer.json), which is why output_check alone is insufficient and this
# explicit map is required.
INPUT_PRODUCER: dict[str, str] = {
    "results/torgo_bootstrap_ci.json": "19_torgo_bootstrap_ci.py",
    "results/torgo_loo_diagnostics.json": "18_torgo_loo_diagnostics.py",
    "results/atd_validation_report.json": "07_validate_atd.py",  # via src/atd_metric
    "results/critical_fixes_report.json": "14_critical_fixes.py",
    "results/tables/layer_analysis.csv": "09_layer_analysis.py",
    "results/tables/table2_ablations.csv": "11_ablations.py",
    "results/perturbation_atd_vs_mfcc.json": "20_mfcc_perturbation.py",
    "results/cross_corpus/ipvs_vowel_bootstrap.json": "17_ipvs_vowel_bootstrap.py",
    "results/cross_corpus/zero_shot_torgo2ipvs.json": "CC_03_zero_shot_cross.py",
    "results/phoneme_summary.json": "16_phoneme_pilot.py",
    "results/uaspeech_validation_report.json": "UA_04_cross_corpus_atd.py",
    "results/combined_validation_report.json": "UA_05_combined_analysis.py",
    "results/best_layer.json": "09_layer_analysis.py",
    "results/streaming_parity.json": "22_streaming_latency_bench.py",
    "results/cross_corpus/ipvs_mfcc_baseline.json": "23_ipvs_mfcc_baseline.py",
    "results/differential_dx_loso.json": "25_diff_dx_classifier.py",
    "results/perturbation_layer_hierarchy.json": "27_perturbation_all_layers.py",
    "results/cross_corpus/ssl_backbone_compare.json": "28_cross_ssl_backbones.py",
    "results/torgo_mixed_effects.json": "32_mixed_effects_per_utterance.py",
    "results/torgo_bayesian_rho.json": "33_bayesian_rho_posterior.py",
    "results/uaspeech_intelligibility_anchor.json": "34_uaspeech_intelligibility_anchor.py",
    "results/power_analysis.json": "31_power_analysis.py",
    "results/l2arctic_specificity.json": "L2_05_specificity_test.py",
    "results/l2arctic_speaker_per.csv": "L2_04_atd_loco.py",  # via src/l2arctic_parser
    "results/vctk_specificity.json": "V_05_specificity_vctk.py",
    "results/cross_corpus_pooled_mixed_effects.json": "L2_06_pooled_mixed_effects.py",
    "results/differential_dx_zrank_loso.json": "30_per_corpus_zrank_classifier.py",
    "results/atd_confidence.csv": "26_atd_confidence_score.py",
}


def _consolidator_inputs() -> list[tuple[str, bool]]:
    """Load 21_consolidate, run main(), and return its registered inputs.

    main() may raise SystemExit if required inputs are missing — we don't
    care about that here; the input registry (_ALL_INPUTS) is populated as a
    side effect regardless of whether the files exist.
    """
    path = PROJECT_ROOT / "experiments" / "21_consolidate_paper_numbers.py"
    spec = importlib.util.spec_from_file_location("consolidate_21", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod.main()
    except SystemExit:
        pass
    except Exception:
        # main() may also fail later (e.g. groupby on an empty df) once a
        # required file is missing in CI; the registry is what we need.
        pass
    # De-dupe preserving order.
    seen: dict[str, bool] = {}
    for rel, required in mod._ALL_INPUTS:
        seen[rel] = seen.get(rel, False) or required
    return list(seen.items())


def _pipeline_scripts() -> set[str]:
    """Every script wired into reproduce_all.py across ALL pipelines."""
    scripts: set[str] = set()
    for pipe in (
        reproduce_all.TORGO_PIPELINE,
        reproduce_all.UASPEECH_PIPELINE,
        reproduce_all.CROSS_CORPUS_PIPELINE,
        reproduce_all.L2ARCTIC_PIPELINE,
        reproduce_all.VCTK_PIPELINE,
        reproduce_all.MULTI_CORPUS_PIPELINE,
    ):
        for step in pipe:
            scripts.add(step.script)
    scripts.add(reproduce_all.CONSOLIDATE_STEP.script)
    return scripts


def test_every_consolidator_input_has_a_pipeline_producer() -> None:
    inputs = _consolidator_inputs()
    assert inputs, "consolidator registered no inputs — registry wiring broke"

    # Import the orphan allowlist straight from the consolidator module.
    path = PROJECT_ROOT / "experiments" / "21_consolidate_paper_numbers.py"
    spec = importlib.util.spec_from_file_location("consolidate_21b", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orphans = set(mod.KNOWN_ORPHAN_INPUTS)

    pipeline_scripts = _pipeline_scripts()

    missing_producer: list[str] = []
    not_in_pipeline: list[tuple[str, str]] = []
    for rel, _required in inputs:
        if rel in orphans:
            continue
        producer = INPUT_PRODUCER.get(rel)
        if producer is None:
            missing_producer.append(rel)
            continue
        if producer not in pipeline_scripts:
            not_in_pipeline.append((rel, producer))

    assert not missing_producer, (
        "consolidator reads inputs with no known producer (update INPUT_PRODUCER "
        f"or add to KNOWN_ORPHAN_INPUTS): {missing_producer}"
    )
    assert not not_in_pipeline, (
        "consolidator inputs whose producer script is NOT a Step in "
        f"reproduce_all.py: {not_in_pipeline}"
    )


def test_required_inputs_are_not_orphans() -> None:
    """A required input must never also be on the orphan allowlist."""
    inputs = _consolidator_inputs()
    path = PROJECT_ROOT / "experiments" / "21_consolidate_paper_numbers.py"
    spec = importlib.util.spec_from_file_location("consolidate_21c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    orphans = set(mod.KNOWN_ORPHAN_INPUTS)
    bad = [rel for rel, required in inputs if required and rel in orphans]
    assert not bad, f"required inputs marked as orphan (contradiction): {bad}"


def test_orphan_inputs_are_documented_not_required() -> None:
    """Every orphan input is read as optional (required=False)."""
    inputs = dict(_consolidator_inputs())
    path = PROJECT_ROOT / "experiments" / "21_consolidate_paper_numbers.py"
    spec = importlib.util.spec_from_file_location("consolidate_21d", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for rel in mod.KNOWN_ORPHAN_INPUTS:
        if rel in inputs:
            assert inputs[rel] is False, f"orphan {rel} must be read as optional"


def test_pipeline_consolidator_runs_last() -> None:
    """In any --include-* combination, 21_consolidate is the final step."""
    import argparse

    for flags in ([], ["--include-all"], ["--include-uaspeech"],
                  ["--include-cross-corpus"], ["--include-l2arctic"],
                  ["--include-vctk"]):
        ns = argparse.Namespace(
            force=False, skip_extract=False, include_uaspeech="--include-uaspeech" in flags,
            include_cross_corpus="--include-cross-corpus" in flags,
            include_l2arctic="--include-l2arctic" in flags,
            include_vctk="--include-vctk" in flags,
            include_all="--include-all" in flags, smoke=True,
            from_step=None, only=None,
        )
        pipe = reproduce_all.build_pipeline(ns)
        assert pipe[-1].script == "21_consolidate_paper_numbers.py", (
            f"consolidator not last for flags={flags}: {pipe[-1].name}"
        )
