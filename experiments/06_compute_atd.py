"""
Experiment 06 — Compute ATD scores for all utterances and validate the metric.

Usage:
    python experiments/06_compute_atd.py
    python experiments/06_compute_atd.py --layer 12   # Use layer 12 instead of 9
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.atd_metric import (
    compute_all_atd_scores,
    aggregate_speaker_atd,
    validate_atd_metric,
    LAYER_POSITION_MAP,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "results" / "logs" / "atd_computation.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Run ATD computation and validation."""
    parser = argparse.ArgumentParser(description="Compute ATD scores")
    parser.add_argument("--layer", type=int, default=9, help="Transformer layer (default: 9)")
    parser.add_argument("--model", type=str, default="wavlm_large")
    args = parser.parse_args()

    layer_idx = args.layer
    model_name = args.model

    print(f"{'='*60}")
    print(f"  ATD Computation — {model_name}, Layer {layer_idx}")
    print(f"{'='*60}\n")

    # Load data
    meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    manifest_path = PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"

    metadata_df = pd.read_csv(meta_path)
    print(f"Metadata: {len(metadata_df)} utterances")

    with open(index_path, "r") as f:
        ssl_index = json.load(f)
    print(f"SSL index: {len(ssl_index)} entries")

    with open(manifest_path, "r") as f:
        reference_manifest = json.load(f)
    print(f"References: {reference_manifest['total_references']} total")

    # Step 1: Compute ATD
    print(f"\n--- Step 1: Computing ATD scores ---")
    atd_df = compute_all_atd_scores(
        ssl_index=ssl_index,
        metadata_df=metadata_df,
        reference_manifest=reference_manifest,
        model_name=model_name,
        layer_idx=layer_idx,
        output_path="results/atd_scores.csv",
        project_root=PROJECT_ROOT,
    )

    # Step 2: Aggregate to speaker level
    print(f"\n--- Step 2: Aggregating to speaker level ---")
    speaker_atd_df = aggregate_speaker_atd(
        atd_df=atd_df,
        output_path="results/speaker_atd.csv",
        project_root=PROJECT_ROOT,
    )

    # Step 3: Validate
    print(f"\n--- Step 3: Validating ATD metric ---")
    report = validate_atd_metric(
        atd_df=atd_df,
        speaker_atd_df=speaker_atd_df,
        output_dir="results/",
        project_root=PROJECT_ROOT,
    )

    # Final verdict
    if report["passed"]:
        print("\n  PAPER CONFIRMED — ATD works!")
    else:
        rho = report["speaker_spearman"]
        if rho > 0.40:
            print(f"\n  ATD is WEAK (rho={rho:.3f}) — try other layers")
        else:
            print(f"\n  ATD FAILED (rho={rho:.3f}) — see diagnosis above")


if __name__ == "__main__":
    main()
