"""Single-source-of-truth paper_numbers.json refresher.

Reads every results/*.json + results/cross_corpus/*.json + relevant CSVs
and produces a flat results/paper_numbers.json with every numeric claim
the manuscript depends on. Re-running overwrites without preserving old
keys, so the file stays in sync with reality.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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

    # Headline TORGO.
    boot = _load_json(PROJECT_ROOT / "results/torgo_bootstrap_ci.json")
    if boot:
        out["torgo_headline"] = boot
    loo = _load_json(PROJECT_ROOT / "results/torgo_loo_diagnostics.json")
    if loo:
        out["torgo_leave_one_out"] = loo

    # Validation report (old).
    val = _load_json(PROJECT_ROOT / "results/atd_validation_report.json")
    if val:
        out["torgo_validation_report"] = val
    crit = _load_json(PROJECT_ROOT / "results/critical_fixes_report.json")
    if crit:
        out["torgo_critical_fixes_report"] = crit

    # Layer analysis CSV.
    layer_csv = PROJECT_ROOT / "results/tables/layer_analysis.csv"
    if layer_csv.exists():
        df = pd.read_csv(layer_csv)
        out["torgo_layer_analysis"] = df.to_dict(orient="records")

    # Ablation table.
    ab_csv = PROJECT_ROOT / "results/tables/table2_ablations.csv"
    if ab_csv.exists():
        df = pd.read_csv(ab_csv)
        out["torgo_ablations"] = df.to_dict(orient="records")

    # Track C — perturbation.
    pert = _load_json(PROJECT_ROOT / "results/perturbation_atd_vs_mfcc.json")
    if pert:
        out["track_c_perturbation_atd_vs_mfcc"] = pert

    # IPVS.
    ipvs = _load_json(PROJECT_ROOT / "results/cross_corpus/ipvs_atd_layer9.json")
    if ipvs:
        out["ipvs_within_corpus_layer9"] = ipvs
    ipvs1 = _load_json(PROJECT_ROOT / "results/cross_corpus/ipvs_atd_layer1.json")
    if ipvs1:
        out["ipvs_within_corpus_layer1"] = ipvs1
    boot_ipvs = _load_json(PROJECT_ROOT / "results/cross_corpus/ipvs_vowel_bootstrap.json")
    if boot_ipvs:
        out["ipvs_per_vowel_bootstrap"] = boot_ipvs
    zs = _load_json(PROJECT_ROOT / "results/cross_corpus/zero_shot_torgo2ipvs.json")
    if zs:
        out["ipvs_zero_shot_from_torgo"] = zs

    # Phoneme.
    phon = _load_json(PROJECT_ROOT / "results/phoneme_summary.json")
    if phon:
        out["phoneme_decomposition_torgo"] = phon

    # UASpeech.
    ua = _load_json(PROJECT_ROOT / "results/uaspeech_validation_report.json")
    if ua:
        out["uaspeech_within_corpus"] = ua
    comb = _load_json(PROJECT_ROOT / "results/combined_validation_report.json")
    if comb:
        out["torgo_plus_uaspeech_pooled"] = comb

    # Confounds.
    conf = _load_json(PROJECT_ROOT / "results/atd_confound_analysis.json")
    if conf:
        out["torgo_confound_analysis"] = conf
    fl = _load_json(PROJECT_ROOT / "results/frame_length_analysis.json")
    if fl:
        out["frame_length_analysis"] = fl

    # Best layer.
    bl = _load_json(PROJECT_ROOT / "results/best_layer.json")
    if bl:
        out["torgo_best_layer"] = bl

    # Write.
    out_path = PROJECT_ROOT / "results/paper_numbers.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"wrote {out_path}")
    print(f"top-level keys: {list(out.keys())}")


if __name__ == "__main__":
    main()
