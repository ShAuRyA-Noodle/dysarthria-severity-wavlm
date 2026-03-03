"""
Experiment 10 — LOSO validation for ATD metric.

For each dysarthric speaker:
  1. References are always built from control speakers only, so no rebuild needed
     (controls are never held out — they ARE the reference builders)
  2. Compute ATD for the held-out dysarthric speaker
  3. Check if ATD rank matches severity rank

This validates that ATD generalizes to unseen dysarthric speakers.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.atd_metric import TORGO_SEVERITY, compute_all_atd_scores, aggregate_speaker_atd, LAYER_POSITION_MAP

DYSARTHRIC = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
CONTROLS = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"]


def main():
    print("=" * 60)
    print("  LOSO Validation — ATD on Dysarthric Speakers")
    print("=" * 60)

    # Load data
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json") as f:
        ssl_index = json.load(f)
    with open(PROJECT_ROOT / "data/processed/references/reference_manifest.json") as f:
        ref_manifest = json.load(f)

    # References are built from control speakers ONLY. 
    # Since dysarthric speakers don't contribute to references,
    # each dysarthric speaker is effectively "held out" already.
    # We compute ATD for each dysarthric speaker individually.

    # Load existing ATD scores (already computed for all speakers)
    atd_path = PROJECT_ROOT / "results/atd_scores.csv"
    if atd_path.exists():
        atd_all = pd.read_csv(atd_path)
    else:
        print("Computing ATD scores...")
        atd_all = compute_all_atd_scores(
            ssl_index=ssl_index, metadata_df=metadata_df,
            reference_manifest=ref_manifest, model_name="wavlm_large",
            layer_idx=9, output_path="results/atd_scores.csv",
            project_root=PROJECT_ROOT,
        )

    # For LOSO: extract per-speaker mean ATD for dysarthric speakers only
    results = []
    for spk in DYSARTHRIC:
        spk_data = atd_all[(atd_all["speaker_id"] == spk) & atd_all["atd_score"].notna()]
        if len(spk_data) == 0:
            continue
        mean_atd = spk_data["atd_score"].mean()
        sev = TORGO_SEVERITY[spk]
        results.append({
            "held_out_speaker": spk,
            "severity": sev,
            "mean_atd": round(mean_atd, 4),
        })

    results_df = pd.DataFrame(results)

    # Compute ranks
    results_df["atd_rank"] = results_df["mean_atd"].rank(method="average").astype(int)
    results_df["severity_rank"] = results_df["severity"].rank(method="average")

    # "correct_rank" = does ATD correctly order this speaker relative to others
    # More precisely: does ATD rank higher for more severe speakers?
    # We check if speakers of the same severity level have ATD ranks
    # that are consistent (severe > mild)
    sev_groups = results_df.groupby("severity")["atd_rank"].mean()
    monotonic = sev_groups.is_monotonic_increasing

    # Per-speaker: check if this speaker's ATD rank is in the correct
    # severity band (mild speakers ranked lower than severe speakers)
    mild_max_atd = results_df[results_df.severity == 1]["mean_atd"].max()
    severe_min_atd = results_df[results_df.severity == 2]["mean_atd"].min()

    def check_rank(row):
        if row["severity"] == 1:
            # Mild should have lower ATD than at least some severe
            return row["mean_atd"] < results_df[results_df.severity == 2]["mean_atd"].median()
        elif row["severity"] == 2:
            # Severe should have higher ATD than at least some mild
            return row["mean_atd"] > results_df[results_df.severity == 1]["mean_atd"].median()
        return True

    results_df["correct_rank"] = results_df.apply(check_rank, axis=1)

    # Save
    out_path = PROJECT_ROOT / "results/tables/loso_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    # Print
    print(f"\n{'='*60}")
    print(results_df.to_string(index=False))
    correct = results_df["correct_rank"].sum()
    print(f"\nLOSO Rank Accuracy: {correct}/{len(results_df)} speakers correctly ranked")
    print(f"Severity group monotonic: {monotonic}")

    # Spearman on dysarthric speakers only
    rho, pval = spearmanr(results_df["severity"], results_df["mean_atd"])
    print(f"Dysarthric-only Spearman: rho={rho:.4f}, p={pval:.4f}")
    print(f"{'='*60}")
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
