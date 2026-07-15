"""
Experiment 05 — Build reference trajectories from control speakers (severity=0).

Builds articulatory reference trajectories for every word in the vocabulary
using control speakers only. These references represent "normal articulation"
and are used to measure ATD deviation.

Usage:
    python experiments/05_build_references.py
    python experiments/05_build_references.py --verify-only
"""

import argparse
import json
import logging
import os
import pickle
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Project root ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

from src.reference_builder import build_all_references

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "results" / "logs" / "reference_building.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42
TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]


def run_verification(manifest_path: Path, output_lines: list) -> bool:
    """
    Run all Phase 3 verification checks.

    Args:
        manifest_path: Path to reference_manifest.json.
        output_lines: List to append verification output lines to.

    Returns:
        True if all checks pass.
    """
    random.seed(RANDOM_SEED)
    all_pass = True

    def p(s: str) -> None:
        print(s)
        output_lines.append(s)

    p("=" * 60)
    p("  Phase 3 Verification Gate")
    p("=" * 60)

    # CHECK 1: References built for expected words
    p("\nCHECK 1 - References built for expected words")
    if not manifest_path.exists():
        p("  [FAIL] reference_manifest.json not found")
        return False

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    n_words = len(manifest["words_with_references"])
    n_skipped = len(manifest["words_skipped"])
    n_total = manifest["total_references"]
    ok1 = n_words >= 10
    p(f"  [{'PASS' if ok1 else 'FAIL'}] Words with references: {n_words}")
    p(f"  Words skipped: {n_skipped}")
    p(f"  Total references (word x layer): {n_total}")
    if not ok1:
        all_pass = False
        p("  Need at least 10 words with references!")

    # Average control speakers per word reference
    if manifest["word_layer_ref"]:
        avg_ctrl = np.mean([
            v["n_control_speakers"]
            for v in manifest["word_layer_ref"].values()
        ])
        p(f"  Avg control speakers per reference: {avg_ctrl:.1f}")

    # CHECK 2: Reference shapes are valid
    p("\nCHECK 2 - Reference shapes and PCA")
    ok2 = True
    sample_words = manifest["words_with_references"][:5]
    if len(sample_words) < 3:
        sample_words = manifest["words_with_references"]

    for word in sample_words:
        key = f"{word}_9"
        if key not in manifest["word_layer_ref"]:
            # Try another layer
            for layer in TARGET_LAYERS:
                key = f"{word}_{layer}"
                if key in manifest["word_layer_ref"]:
                    break

        if key not in manifest["word_layer_ref"]:
            p(f"  [FAIL] No reference found for word '{word}'")
            ok2 = False
            continue

        ref_info = manifest["word_layer_ref"][key]
        ref_path = PROJECT_ROOT / ref_info["path"]
        if not ref_path.exists():
            p(f"  [FAIL] File not found: {ref_path}")
            ok2 = False
            continue

        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)

        shape = ref_data["reference"].shape
        has_pca = ref_data["pca_model"] is not None
        speakers = ref_data["control_speakers_used"]
        shape_ok = len(shape) == 2 and shape[1] <= 64
        pca_ok = has_pca

        icon = "PASS" if (shape_ok and pca_ok) else "FAIL"
        p(f"  [{icon}] '{word}': shape={shape}, PCA={has_pca}, speakers={speakers}")

        if not (shape_ok and pca_ok):
            ok2 = False

    if not ok2:
        all_pass = False

    # CHECK 3: Visual sanity - plot reference trajectory
    p("\nCHECK 3 - Visual sanity (reference trajectory plot)")
    ok3 = False
    plot_word = None

    # Pick a word that has a layer-9 reference
    for word in manifest["words_with_references"]:
        key = f"{word}_9"
        if key in manifest["word_layer_ref"]:
            plot_word = word
            break

    if plot_word is None and manifest["words_with_references"]:
        plot_word = manifest["words_with_references"][0]
        # Find any available layer
        for layer in TARGET_LAYERS:
            key = f"{plot_word}_{layer}"
            if key in manifest["word_layer_ref"]:
                break

    if plot_word and key in manifest["word_layer_ref"]:
        ref_info = manifest["word_layer_ref"][key]
        ref_path = PROJECT_ROOT / ref_info["path"]

        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)

        ref_traj = ref_data["reference"]  # (T, 64)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: First 2 PCA dims as a trajectory path
        axes[0].plot(ref_traj[:, 0], ref_traj[:, 1], "b-o", markersize=3, alpha=0.7)
        axes[0].plot(ref_traj[0, 0], ref_traj[0, 1], "go", markersize=10, label="Start")
        axes[0].plot(ref_traj[-1, 0], ref_traj[-1, 1], "ro", markersize=10, label="End")
        axes[0].set_xlabel("PCA Dim 1")
        axes[0].set_ylabel("PCA Dim 2")
        axes[0].set_title(f"Reference trajectory: '{plot_word}' (PC1 vs PC2)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: First 4 PCA dims over time
        for dim in range(min(4, ref_traj.shape[1])):
            axes[1].plot(ref_traj[:, dim], label=f"PC{dim+1}", alpha=0.8)
        axes[1].set_xlabel("Frame")
        axes[1].set_ylabel("Value")
        axes[1].set_title(f"Reference dims over time: '{plot_word}'")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = PROJECT_ROOT / "results" / "figures" / "reference_sanity_check.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(fig_path), dpi=150, bbox_inches="tight")
        plt.close()

        ok3 = True
        p(f"  [PASS] Plot saved: {fig_path}")
        p(f"  Word: '{plot_word}', shape: {ref_traj.shape}")
    else:
        p("  [FAIL] No word found for plotting")

    if not ok3:
        all_pass = False

    # Storage summary
    p("\nStorage summary:")
    ref_dir = PROJECT_ROOT / "data" / "processed" / "references"
    total_bytes = sum(f.stat().st_size for f in ref_dir.rglob("*.pkl"))
    p(f"  Reference files: {total_bytes / (1024**2):.1f} MB")

    # SUMMARY
    p("\n" + "=" * 60)
    if all_pass:
        p("  PHASE 3 COMPLETE - All checks passed")
    else:
        p("  PHASE 3 INCOMPLETE - Some checks failed")
    p("=" * 60)

    return all_pass


def main() -> None:
    """Main entry point for reference building."""
    set_global_seed(42)
    parser = argparse.ArgumentParser(
        description="Build reference trajectories from control speakers"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run verification, skip building",
    )
    args = parser.parse_args()

    # Load data
    meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    manifest_path = PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"

    metadata_df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(metadata_df)} utterances")

    if args.verify_only:
        output_lines = []
        run_verification(manifest_path, output_lines)
        # Save report
        report_path = PROJECT_ROOT / "results" / "logs" / "phase3_verification.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
        return

    # Load SSL index
    with open(index_path, "r") as f:
        ssl_index = json.load(f)
    print(f"Loaded SSL index: {len(ssl_index)} entries")

    # Load shared vocabulary
    vocab_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "shared_vocabulary.txt"
    if vocab_path.exists():
        shared_vocab = vocab_path.read_text().strip().split("\n")
        shared_vocab = [w.strip() for w in shared_vocab if w.strip()]
        print(f"Shared vocabulary: {len(shared_vocab)} words: {shared_vocab}")
    else:
        shared_vocab = []
        print("No shared_vocabulary.txt found — will use all control words")

    # Build references
    print(f"\nBuilding references for WavLM-Large, {len(TARGET_LAYERS)} layers...")
    manifest = build_all_references(
        ssl_index=ssl_index,
        metadata_df=metadata_df,
        model_name="wavlm_large",
        target_layers=TARGET_LAYERS,
        output_dir="data/processed/references/",
        min_control_utterances=2,
        project_root=PROJECT_ROOT,
    )

    # Run verification
    output_lines = []
    run_verification(manifest_path, output_lines)

    # Save report
    report_path = PROJECT_ROOT / "results" / "logs" / "phase3_verification.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
