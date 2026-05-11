"""
Experiment UA-04 — Cross-corpus ATD validation on UA-Speech.

Step 1: Build UA-Speech references from UA-Speech control speakers.
Step 2: Compute ATD for UA-Speech dysarthric speakers.
Step 3: Statistical validation (Spearman ρ, severity groups).

ATD hyperparameters (Layer 1, PCA dim=64, DTW normalization) all from TORGO — zero-shot.

Usage:
    python experiments/UA_04_cross_corpus_atd.py
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, mannwhitneyu
from tqdm import tqdm

# Ensure Unicode on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reference_builder import build_all_references, LAYER_IDX_MAP  # noqa: E402
from src.atd_metric import (  # noqa: E402
    compute_single_atd,
    _load_reference_cache,
    LAYER_POSITION_MAP,
)
from src.uaspeech_parser import UA_INTELLIGIBILITY, UA_SEVERITY  # noqa: E402

LOG_DIR = PROJECT_ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "ua_atd.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Step 1: Build UA-Speech references
# ─────────────────────────────────────────────────────────────────────

def build_uaspeech_references(target_layers: list[int]) -> str:
    """Build references from UA-Speech control speakers."""
    print(f"\n{'='*60}")
    print(f"  Building UA-Speech References")
    print(f"{'='*60}\n")

    # Load UA-Speech metadata
    meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv"
    ua_meta = pd.read_csv(meta_path)
    print(f"UA-Speech metadata: {len(ua_meta)} utterances")

    # Load unified SSL index
    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    with open(index_path) as f:
        ssl_index = json.load(f)

    # Filter to UA-Speech entries only
    ua_index = {k: v for k, v in ssl_index.items() if v.get("dataset") == "uaspeech"}
    print(f"UA-Speech entries in SSL index: {len(ua_index)}")

    # Count controls
    ua_controls = ua_meta[ua_meta["is_control"] == True]
    print(f"UA-Speech control utterances: {len(ua_controls)}")
    print(f"UA-Speech control speakers: {ua_controls['speaker_id'].unique().tolist()}")

    # Build references using reusable function
    ref_dir = "data/processed/references_uaspeech/"
    manifest = build_all_references(
        ssl_index=ua_index,
        metadata_df=ua_meta,
        model_name="wavlm_large",
        target_layers=target_layers,
        output_dir=ref_dir,
        min_control_utterances=2,
        project_root=PROJECT_ROOT,
    )

    # Save manifest
    manifest_path = PROJECT_ROOT / ref_dir / "reference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nUA-Speech references built: {manifest.get('total_references', 0)}")
    return ref_dir


# ─────────────────────────────────────────────────────────────────────
# Step 2: Compute ATD for UA-Speech dysarthric speakers
# ─────────────────────────────────────────────────────────────────────

def compute_uaspeech_atd(ref_dir: str, layer_idx: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute ATD for all UA-Speech dysarthric utterances at given layer."""
    print(f"\n{'='*60}")
    print(f"  ATD Computation — Layer {layer_idx}")
    print(f"{'='*60}\n")

    layer_position = LAYER_POSITION_MAP.get(layer_idx, 0)

    # Load metadata
    meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv"
    ua_meta = pd.read_csv(meta_path)

    # Load SSL index (only UA-Speech)
    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    with open(index_path) as f:
        full_index = json.load(f)
    ua_index = {k: v for k, v in full_index.items() if v.get("dataset") == "uaspeech"}

    # Load reference manifest
    manifest_path = PROJECT_ROOT / ref_dir / "reference_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Load reference cache for this layer
    ref_cache = _load_reference_cache(manifest, layer_idx, PROJECT_ROOT)
    print(f"Loaded {len(ref_cache)} word references for layer {layer_idx}")

    # Filter to dysarthric speakers only
    dysarthric_meta = ua_meta[ua_meta["is_control"] == False].copy()
    print(f"Dysarthric utterances: {len(dysarthric_meta)}")
    print(f"Dysarthric speakers: {sorted(dysarthric_meta['speaker_id'].unique().tolist())}")

    # Compute ATD for each utterance
    results = []
    n_computed = 0
    n_no_ref = 0
    n_failed = 0

    for _, row in tqdm(dysarthric_meta.iterrows(), total=len(dysarthric_meta),
                       desc=f"ATD L{layer_idx}"):
        utt_id = row["utterance_id"]
        speaker_id = row["speaker_id"]
        word = row.get("word", "")

        atd = compute_single_atd(
            utterance_id=utt_id,
            ssl_index=ua_index,
            reference_cache=ref_cache,
            model_name="wavlm_large",
            layer_idx=layer_idx,
            layer_position=layer_position,
            project_root=PROJECT_ROOT,
        )

        if atd is not None:
            n_computed += 1
        else:
            # Check if it's a reference miss vs other failure
            ref_key = f"{word}_{layer_idx}"
            if ref_key not in ref_cache:
                n_no_ref += 1
            else:
                n_failed += 1

        results.append({
            "utterance_id": utt_id,
            "speaker_id": speaker_id,
            "word": word,
            "intelligibility_score": float(UA_INTELLIGIBILITY.get(speaker_id, -1)),
            "severity_proxy": int(UA_SEVERITY.get(speaker_id, -1)),
            "atd_score": atd,
            "layer_idx": layer_idx,
            "is_control": False,
        })

    atd_df = pd.DataFrame(results)

    # Save
    out_path = PROJECT_ROOT / f"results/uaspeech_atd_scores_layer{layer_idx}.csv"
    atd_df.to_csv(out_path, index=False)

    print(f"\nATD computation (Layer {layer_idx}):")
    print(f"  Computed: {n_computed}")
    print(f"  No reference: {n_no_ref}")
    print(f"  Other failures: {n_failed}")
    print(f"  Saved: {out_path}")

    # Aggregate to speaker level
    valid = atd_df.dropna(subset=["atd_score"])
    speaker_stats = valid.groupby("speaker_id").agg(
        mean_atd=("atd_score", "mean"),
        std_atd=("atd_score", "std"),
        median_atd=("atd_score", "median"),
        n_utterances=("atd_score", "count"),
        intelligibility_score=("intelligibility_score", "first"),
        severity_proxy=("severity_proxy", "first"),
    ).reset_index()

    # Add severity group
    def severity_group(intel):
        if intel < 25:
            return "very_low"
        elif intel < 50:
            return "low"
        elif intel < 75:
            return "mid"
        else:
            return "high"

    speaker_stats["severity_group"] = speaker_stats["intelligibility_score"].apply(severity_group)

    spk_path = PROJECT_ROOT / f"results/uaspeech_speaker_atd_layer{layer_idx}.csv"
    speaker_stats.to_csv(spk_path, index=False)

    print(f"\nSpeaker-level ATD (Layer {layer_idx}):")
    print(speaker_stats.to_string(index=False))

    return atd_df, speaker_stats


# ─────────────────────────────────────────────────────────────────────
# Step 3: Statistical validation
# ─────────────────────────────────────────────────────────────────────

def statistical_validation(speaker_atd: pd.DataFrame, layer_idx: int) -> dict:
    """Run statistical validation on UA-Speech ATD results."""
    print(f"\n{'='*60}")
    print(f"  Statistical Validation — Layer {layer_idx}")
    print(f"{'='*60}\n")

    # Primary: Spearman rho (ATD vs intelligibility)
    # Higher intelligibility = LOWER ATD expected (negative rho)
    rho, p = spearmanr(speaker_atd["intelligibility_score"], speaker_atd["mean_atd"])
    print(f"Spearman rho (ATD vs intelligibility): {rho:.3f} (p={p:.4f})")
    print(f"Abs Spearman rho: {abs(rho):.3f}")

    # Severity group separation
    groups = {
        "very_low": speaker_atd[speaker_atd["intelligibility_score"] < 25]["mean_atd"],
        "low": speaker_atd[(speaker_atd["intelligibility_score"] >= 25) &
                           (speaker_atd["intelligibility_score"] < 50)]["mean_atd"],
        "mid": speaker_atd[(speaker_atd["intelligibility_score"] >= 50) &
                           (speaker_atd["intelligibility_score"] < 75)]["mean_atd"],
        "high": speaker_atd[speaker_atd["intelligibility_score"] >= 75]["mean_atd"],
    }

    for k, v in groups.items():
        if len(v) > 0:
            print(f"  {k}: n={len(v)}, mean_atd={v.mean():.1f}")

    # Monotonic check
    means = {k: round(float(v.mean()), 1) for k, v in groups.items() if len(v) > 0}
    print(f"Severity group means: {means}")

    # Build report
    report = {
        "dataset": "UA-Speech",
        "layer": layer_idx,
        "n_dysarthric_speakers": int(len(speaker_atd)),
        "n_utterances": int(speaker_atd["n_utterances"].sum()),
        "spearman_rho": round(float(rho), 3),
        "p_value": round(float(p), 4),
        "abs_spearman_rho": round(abs(float(rho)), 3),
        "references_from": "UA-Speech_controls",
        "zero_shot_cross_corpus": True,
        "method_from": "TORGO (Layer 1, PCA=64, DTW norm)",
        "severity_group_means": means,
    }

    return report


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main() -> None:
    start_time = time.time()

    # Best layer from TORGO analysis
    best_layer_path = PROJECT_ROOT / "results" / "best_layer.json"
    with open(best_layer_path) as f:
        best_layer_info = json.load(f)
    best_layer = best_layer_info["best_layer"]  # Layer 1
    comparison_layer = 9  # For comparison

    print(f"Best layer from TORGO: Layer {best_layer} (rho={best_layer_info['spearman_rho']})")

    # Step 1: Build UA-Speech references for both layers
    target_layers = sorted(set([best_layer, comparison_layer]))
    ref_dir = build_uaspeech_references(target_layers)

    # Step 2: Compute ATD at best layer
    atd_df_best, speaker_atd_best = compute_uaspeech_atd(ref_dir, best_layer)

    # Also save as the "primary" files
    speaker_atd_best.to_csv(PROJECT_ROOT / "results" / "uaspeech_speaker_atd.csv", index=False)
    atd_df_best.to_csv(PROJECT_ROOT / "results" / "uaspeech_atd_scores.csv", index=False)

    # Step 2b: Compute ATD at comparison layer
    atd_df_cmp, speaker_atd_cmp = compute_uaspeech_atd(ref_dir, comparison_layer)

    # Step 3: Statistical validation
    report_best = statistical_validation(speaker_atd_best, best_layer)
    report_cmp = statistical_validation(speaker_atd_cmp, comparison_layer)

    # Save primary report (best layer)
    report_path = PROJECT_ROOT / "results" / "uaspeech_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_best, f, indent=2)

    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  UA-Speech ATD Validation Complete")
    print(f"{'='*60}")
    print(f"  Layer {best_layer}: Spearman rho = {report_best['spearman_rho']:.3f} "
          f"(|rho| = {report_best['abs_spearman_rho']:.3f}, p = {report_best['p_value']:.4f})")
    print(f"  Layer {comparison_layer}: Spearman rho = {report_cmp['spearman_rho']:.3f} "
          f"(|rho| = {report_cmp['abs_spearman_rho']:.3f})")
    print(f"  Time: {elapsed/60:.1f} min")

    # GO/NO-GO
    rho = report_best['abs_spearman_rho']
    if rho > 0.4:
        print(f"\n  STRONG CROSS-CORPUS RESULT (|rho| = {rho:.3f})")
    elif rho > 0.3:
        print(f"\n  ACCEPTABLE CROSS-CORPUS RESULT (|rho| = {rho:.3f})")
    else:
        print(f"\n  WEAK RESULT (|rho| = {rho:.3f}) — investigate")

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
