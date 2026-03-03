"""
Experiment 09 — Analyze ATD across WavLM layers to validate Layer 9 choice.

Computes speaker-level Spearman correlation for each of 9 target layers.
Produces results/tables/layer_analysis.csv (Table 2 in paper).

Usage:
    python experiments/09_layer_analysis.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.atd_metric import compute_all_atd_scores, aggregate_speaker_atd, LAYER_POSITION_MAP

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]


def main() -> None:
    """Run layer analysis."""
    print(f"{'='*60}")
    print(f"  Layer Analysis — ATD across 9 WavLM layers")
    print(f"{'='*60}\n")

    # Load data
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json") as f:
        ssl_index = json.load(f)
    with open(PROJECT_ROOT / "data/processed/references/reference_manifest.json") as f:
        ref_manifest = json.load(f)

    results = []

    for layer in TARGET_LAYERS:
        print(f"\n--- Layer {layer} ---")

        atd_df = compute_all_atd_scores(
            ssl_index=ssl_index,
            metadata_df=metadata_df,
            reference_manifest=ref_manifest,
            model_name="wavlm_large",
            layer_idx=layer,
            output_path=f"results/atd_scores_layer{layer}.csv",
            project_root=PROJECT_ROOT,
        )

        speaker_df = aggregate_speaker_atd(
            atd_df, output_path=f"results/speaker_atd_layer{layer}.csv",
            project_root=PROJECT_ROOT,
        )

        rho, p_val = spearmanr(speaker_df["severity_score"], speaker_df["mean_atd"])

        results.append({
            "layer_idx": layer,
            "spearman_rho": round(rho, 4),
            "p_value": round(p_val, 6),
            "significant": p_val < 0.05,
            "n_speakers": len(speaker_df),
            "n_utterances": int(atd_df.atd_score.notna().sum()),
        })

        print(f"  Spearman rho = {rho:.4f}, p = {p_val:.6f}")

    # Save results
    results_df = pd.DataFrame(results)
    out_path = PROJECT_ROOT / "results/tables/layer_analysis.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)

    # Print formatted table
    print(f"\n{'='*60}")
    print(f"  Layer Analysis Results")
    print(f"{'='*60}")
    print(f"  {'Layer':>5} | {'Spearman ρ':>10} | {'p-value':>10} | {'Sig.':>5}")
    print(f"  {'-'*40}")
    for _, r in results_df.iterrows():
        sig = "***" if r.p_value < 0.001 else "**" if r.p_value < 0.01 else "*" if r.p_value < 0.05 else ""
        print(f"  {int(r.layer_idx):>5} | {r.spearman_rho:>10.4f} | {r.p_value:>10.6f} | {sig:>5}")

    best = results_df.loc[results_df.spearman_rho.idxmax()]
    print(f"\n  Best layer: {int(best.layer_idx)} (rho={best.spearman_rho:.4f})")
    print(f"{'='*60}")

    # Save best layer info
    with open(PROJECT_ROOT / "results/best_layer.json", "w") as f:
        json.dump({"best_layer": int(best.layer_idx), "spearman_rho": float(best.spearman_rho)}, f, indent=2)


if __name__ == "__main__":
    main()
