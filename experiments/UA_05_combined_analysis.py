"""
Experiment UA-05 — Combined analysis: TORGO + UA-Speech ATD results.

Loads both datasets' speaker-level ATD scores, normalizes intelligibility,
computes combined Spearman rho, and generates the cross-corpus validation report.

Usage:
    python experiments/UA_05_combined_analysis.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Ensure Unicode on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402


def main() -> None:
    set_global_seed(42)
    print(f"\n{'='*60}")
    print(f"  Combined TORGO + UA-Speech Analysis")
    print(f"{'='*60}\n")

    # Load TORGO speaker ATD
    torgo_df = pd.read_csv(PROJECT_ROOT / "results" / "speaker_atd.csv")
    print(f"TORGO speakers: {len(torgo_df)}")

    # Load UA-Speech speaker ATD
    ua_df = pd.read_csv(PROJECT_ROOT / "results" / "uaspeech_speaker_atd.csv")
    print(f"UA-Speech speakers: {len(ua_df)}")

    # ─── TORGO: normalize to intelligibility ──────────────────────
    # severity 0 = control (100% intelligibility)
    # severity 1 = mild-moderate (60% proxy)
    # severity 2 = severe (20% proxy)
    torgo_intel_map = {0: 100.0, 1: 60.0, 2: 20.0}
    torgo_df["intelligibility"] = torgo_df["severity_score"].map(torgo_intel_map)
    torgo_df["dataset"] = "TORGO"

    # ─── UA-Speech: use actual intelligibility ───────────────────
    ua_df["intelligibility"] = ua_df["intelligibility_score"]
    ua_df["dataset"] = "UA-Speech"

    # ─── Filter to dysarthric only ────────────────────────────────
    torgo_dys = torgo_df[torgo_df["is_control"] == False].copy()
    ua_dys = ua_df.copy()  # Already dysarthric only

    print(f"\nDysarthric speakers:")
    print(f"  TORGO: {len(torgo_dys)}")
    print(f"  UA-Speech: {len(ua_dys)}")

    # ─── Individual analyses ──────────────────────────────────────
    torgo_rho, torgo_p = spearmanr(torgo_dys["intelligibility"], torgo_dys["mean_atd"])
    ua_rho, ua_p = spearmanr(ua_dys["intelligibility"], ua_dys["mean_atd"])

    print(f"\nIndividual Spearman correlations:")
    print(f"  TORGO (n={len(torgo_dys)}): rho = {torgo_rho:.3f} (p={torgo_p:.4f})")
    print(f"  UA-Speech (n={len(ua_dys)}): rho = {ua_rho:.3f} (p={ua_p:.4f})")

    # ─── Combined analysis ────────────────────────────────────────
    combined = pd.concat([
        torgo_dys[["speaker_id", "mean_atd", "intelligibility", "dataset"]],
        ua_dys[["speaker_id", "mean_atd", "intelligibility", "dataset"]],
    ], ignore_index=True)

    combined_rho, combined_p = spearmanr(combined["intelligibility"], combined["mean_atd"])

    print(f"\n{'='*60}")
    print(f"  COMBINED (n={len(combined)}): Spearman rho = {combined_rho:.3f} (p={combined_p:.4f})")
    print(f"  TORGO only (n={len(torgo_dys)}): rho = {torgo_rho:.3f}")
    print(f"  UA-Speech only (n={len(ua_dys)}): rho = {ua_rho:.3f}")
    print(f"  Combined (n={len(combined)}): rho = {combined_rho:.3f}")
    print(f"{'='*60}")

    # ─── Save combined report ─────────────────────────────────────
    report = {
        "torgo_n_speakers": int(len(torgo_dys)),
        "torgo_spearman_rho": round(float(torgo_rho), 3),
        "torgo_p_value": round(float(torgo_p), 4),
        "uaspeech_n_speakers": int(len(ua_dys)),
        "uaspeech_spearman_rho": round(float(ua_rho), 3),
        "uaspeech_p_value": round(float(ua_p), 4),
        "combined_n_speakers": int(len(combined)),
        "combined_spearman_rho": round(float(combined_rho), 3),
        "combined_p_value": round(float(combined_p), 4),
        "method": "ATD (WavLM-Large, Layer 1, PCA=64, DTW)",
        "zero_shot_cross_corpus": True,
        "note": "TORGO severity mapped to intelligibility: 0->100%, 1->60%, 2->20%",
    }

    report_path = PROJECT_ROOT / "results" / "combined_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {report_path}")

    # Save combined data for figures
    combined.to_csv(PROJECT_ROOT / "results" / "combined_speaker_atd.csv", index=False)

    # ─── Generate figures ─────────────────────────────────────────
    print("\nGenerating cross-corpus figures...")
    from src.visualization import fig7_cross_corpus_scatter, fig8_cross_dataset  # noqa: E402
    fig7_cross_corpus_scatter(PROJECT_ROOT)
    fig8_cross_dataset(PROJECT_ROOT)

    print(f"\n{'='*60}")
    print(f"  COMBINED ANALYSIS COMPLETE")
    print(f"{'='*60}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
