"""Single-source-of-truth paper_numbers.json refresher.

Reads every results/*.json + results/cross_corpus/*.json + relevant CSVs
and produces a flat results/paper_numbers.json with every numeric claim
the manuscript depends on. Re-running overwrites without preserving old
keys, so the file stays in sync with reality.

Strictness
----------
Every input path is registered with a ``required`` flag. A *required* input
missing on disk is a hard error: the script prints the offenders and exits
non-zero (instead of silently writing ``None`` into paper_numbers.json).
*Optional* inputs (corpus-specific artifacts that only exist when the
matching ``reproduce_all.py --include-*`` pipeline has been run, plus a few
pre-existing orphan files with no in-pipeline producer) are tolerated when
absent and simply skipped.

The set of input paths read here is asserted to be covered by
``reproduce_all.py`` in ``tests/test_reproduce_all_coverage.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Inputs that have no producing Step in reproduce_all.py. They are committed,
# pre-existing artifacts (or written by a src/ helper that no experiment
# currently invokes). Tolerated when absent; documented so the coverage test
# can exempt them explicitly rather than silently.
KNOWN_ORPHAN_INPUTS: set[str] = {
    "results/perturbation_layer1_summary.json",
    "results/cross_corpus/ipvs_atd_layer9.json",
    "results/cross_corpus/ipvs_atd_layer1.json",
    "results/frame_length_analysis.json",
    "results/atd_confound_analysis.json",
}

# Track everything we tried to read so the run can fail loudly on missing
# *required* inputs and the coverage test can introspect the full input set.
_MISSING_REQUIRED: list[str] = []
_ALL_INPUTS: list[tuple[str, bool]] = []  # (relpath, required)


def _register(rel: str, required: bool) -> Path:
    _ALL_INPUTS.append((rel, required))
    return PROJECT_ROOT / rel


def _load_json(rel: str, *, required: bool):
    import json
    path = _register(rel, required)
    if not path.exists():
        if required:
            _MISSING_REQUIRED.append(rel)
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_csv(rel: str, *, required: bool):
    path = _register(rel, required)
    if not path.exists():
        if required:
            _MISSING_REQUIRED.append(rel)
        return None
    return pd.read_csv(path)


def main() -> None:
    out: dict = {
        "generated_by": "experiments/21_consolidate_paper_numbers.py",
        "generated_at_utc": pd.Timestamp.utcnow().isoformat(),
        "branch": "audit-cleanup-2026-05-08",
        "datasets": {
            "torgo": {
                "n_speakers": 15,
                "n_dysarthric": 8,
                "n_controls": 7,
                "n_utterances_in_metadata": 8218,
                "n_utterances_with_atd": 7826,
                "n_mfa_aligned": 8007,
            },
            "uaspeech": {
                "n_dysarthric": 15,
            },
            "ipvs": {
                "n_speakers": 61,
                "n_pd": 24,
                "n_hc_old": 22,
                "n_hc_young": 15,
                "n_utterances": 831,
                "n_resampled_44100_to_16000": 160,
            },
        },
    }

    # ----- Headline TORGO (required: produced by the default pipeline) -----
    boot = _load_json("results/torgo_bootstrap_ci.json", required=True)
    if boot:
        out["torgo_headline"] = boot
    loo = _load_json("results/torgo_loo_diagnostics.json", required=True)
    if loo:
        out["torgo_leave_one_out"] = loo

    val = _load_json("results/atd_validation_report.json", required=True)
    if val:
        out["torgo_validation_report"] = val
    crit = _load_json("results/critical_fixes_report.json", required=True)
    if crit:
        out["torgo_critical_fixes_report"] = crit

    # Layer analysis CSV (required: 09_layer_analysis).
    layer_df = _read_csv("results/tables/layer_analysis.csv", required=True)
    if layer_df is not None:
        out["torgo_layer_analysis"] = layer_df.to_dict(orient="records")

    # Ablation table (required: 11_ablations).
    ab_df = _read_csv("results/tables/table2_ablations.csv", required=True)
    if ab_df is not None:
        out["torgo_ablations"] = ab_df.to_dict(orient="records")

    # ----- Track C — perturbation -----
    pert = _load_json("results/perturbation_atd_vs_mfcc.json", required=False)  # 20 (optional)
    if pert:
        out["track_c_perturbation_atd_vs_mfcc"] = pert
    pert_l1 = _load_json("results/perturbation_layer1_summary.json", required=False)  # orphan
    if pert_l1:
        out["track_c_perturbation_layer1"] = pert_l1

    # ----- IPVS (optional: --include-cross-corpus) -----
    ipvs = _load_json("results/cross_corpus/ipvs_atd_layer9.json", required=False)  # orphan
    if ipvs:
        out["ipvs_within_corpus_layer9"] = ipvs
    ipvs1 = _load_json("results/cross_corpus/ipvs_atd_layer1.json", required=False)  # orphan
    if ipvs1:
        out["ipvs_within_corpus_layer1"] = ipvs1
    boot_ipvs = _load_json("results/cross_corpus/ipvs_vowel_bootstrap.json", required=False)  # 17
    if boot_ipvs:
        out["ipvs_per_vowel_bootstrap"] = boot_ipvs
    zs = _load_json("results/cross_corpus/zero_shot_torgo2ipvs.json", required=False)  # CC_03
    if zs:
        out["ipvs_zero_shot_from_torgo"] = zs

    # Phoneme (optional: 16_phoneme_pilot).
    phon = _load_json("results/phoneme_summary.json", required=False)
    if phon:
        out["phoneme_decomposition_torgo"] = phon

    # UASpeech (optional: --include-uaspeech).
    ua = _load_json("results/uaspeech_validation_report.json", required=False)  # UA_04
    if ua:
        out["uaspeech_within_corpus"] = ua
    comb = _load_json("results/combined_validation_report.json", required=False)  # UA_05
    if comb:
        out["torgo_plus_uaspeech_pooled"] = comb

    # Confounds.
    conf = _load_json("results/atd_confound_analysis.json", required=False)  # orphan
    if conf:
        out["torgo_confound_analysis"] = conf
    fl = _load_json("results/frame_length_analysis.json", required=False)  # orphan
    if fl:
        out["frame_length_analysis"] = fl

    # Best layer (required: 09_layer_analysis).
    bl = _load_json("results/best_layer.json", required=True)
    if bl:
        out["torgo_best_layer"] = bl

    # Streaming parity + latency (Track A) (optional: 22).
    sp = _load_json("results/streaming_parity.json", required=False)
    if sp:
        out["track_a_streaming_parity"] = sp

    # IPVS MFCC baseline (Track D competitor) (optional: 23).
    ipvs_mfcc = _load_json("results/cross_corpus/ipvs_mfcc_baseline.json", required=False)
    if ipvs_mfcc:
        out["ipvs_mfcc_svr_baseline"] = ipvs_mfcc

    # Differential dx classifier (T3) (optional: 25, needs UA+IPVS).
    diff_dx = _load_json("results/differential_dx_loso.json", required=False)
    if diff_dx:
        out["differential_dx_classifier"] = diff_dx

    # Layer hierarchy (T1) (optional: 27).
    lh = _load_json("results/perturbation_layer_hierarchy.json", required=False)
    if lh:
        out["perturbation_layer_hierarchy"] = lh

    # SSL backbone comparison (T2) (optional: 28).
    ssl_bb = _load_json("results/cross_corpus/ssl_backbone_compare.json", required=False)
    if ssl_bb:
        out["ssl_backbone_comparison"] = ssl_bb

    # Mixed-effects per-utterance model (T32) (required: 32).
    me = _load_json("results/torgo_mixed_effects.json", required=True)
    if me:
        out["torgo_mixed_effects_per_utterance"] = me

    # Bayesian posterior on TORGO rho (T33) (required: 33).
    bayes = _load_json("results/torgo_bayesian_rho.json", required=True)
    if bayes:
        out["torgo_bayesian_rho_posterior"] = bayes

    # UASpeech intelligibility anchor (T34) (optional: 34, needs UA).
    ua_anchor = _load_json("results/uaspeech_intelligibility_anchor.json", required=False)
    if ua_anchor:
        out["uaspeech_intelligibility_anchor"] = ua_anchor

    # Power analysis (T31) (required: 31).
    pw = _load_json("results/power_analysis.json", required=True)
    if pw:
        out["torgo_power_analysis"] = pw

    # L2-ARCTIC accent-specificity (optional: --include-l2arctic).
    l2_spec = _load_json("results/l2arctic_specificity.json", required=False)  # L2_05
    if l2_spec:
        out["l2arctic_accent_specificity"] = l2_spec
    l2_per = _read_csv("results/l2arctic_speaker_per.csv", required=False)  # src/l2arctic_parser
    if l2_per is not None:
        out["l2arctic_per_per_speaker"] = l2_per.to_dict(orient="records")

    # VCTK native-accent specificity (optional: --include-vctk).
    vctk_spec = _load_json("results/vctk_specificity.json", required=False)  # V_05
    if vctk_spec:
        out["vctk_native_accent_specificity"] = vctk_spec

    # 4-corpus pooled mixed-effects / OLS (optional: L2_06, needs all corpora).
    pool = _load_json("results/cross_corpus_pooled_mixed_effects.json", required=False)
    if pool:
        out["four_corpus_pooled"] = pool

    # z-rank classifier (v2 of differential dx) (optional: 30, needs UA+IPVS).
    zr = _load_json("results/differential_dx_zrank_loso.json", required=False)
    if zr:
        out["differential_dx_zrank_classifier"] = zr

    # ATD confidence stratified means (T4) — built from CSV (required: 26).
    cdf = _read_csv("results/atd_confidence.csv", required=True)
    if cdf is not None and not cdf.empty:
        grp = cdf.groupby("severity_score")[
            ["atd_score", "path_entropy", "coverage_pct", "confidence_combined"]
        ].mean().round(4)
        out["atd_confidence_by_severity"] = grp.reset_index().to_dict(orient="records")

    # ----- Hard-fail on any missing REQUIRED input -----
    if _MISSING_REQUIRED:
        print("ERROR: missing required input(s) for paper_numbers.json:", file=sys.stderr)
        for rel in _MISSING_REQUIRED:
            print(f"  - {rel}", file=sys.stderr)
        print(
            "\nRun the producing steps first (see reproduce_all.py). Refusing to "
            "write a partial paper_numbers.json.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # ----- Write -----
    import json
    out_path = PROJECT_ROOT / "results/paper_numbers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print(f"top-level keys: {list(out.keys())}")


if __name__ == "__main__":
    main()
