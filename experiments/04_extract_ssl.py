"""
Experiment 04 — Extract SSL embeddings from WavLM-Large (frozen) for all audio.

Usage:
    python experiments/04_extract_ssl.py                   # Full extraction
    python experiments/04_extract_ssl.py --status           # Check status from another terminal
    python experiments/04_extract_ssl.py --model hubert_large  # Extract with HuBERT instead

This script is fully resumable. If interrupted, re-run the same command.
"""

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Project root ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ssl_extractor import (
    SSLExtractor,
    TARGET_LAYERS,
    MODELS_CONFIG,
    extract_dataset_embeddings,
    build_ssl_index,
)

# ── Logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "results" / "logs" / "extraction.log",
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger(__name__)

RANDOM_SEED = 42


def print_extraction_status(log_path: str) -> None:
    """
    Can be called from a SECOND terminal while extraction runs.
    Reads extraction_log.json and prints live status.

    Args:
        log_path: Path to extraction_log.json.
    """
    p = Path(log_path)
    if not p.exists():
        print(f"Log file not found: {p}")
        print("Extraction may not have started yet.")
        return

    with open(p, "r", encoding="utf-8") as f:
        log = json.load(f)

    completed = len(log.get("completed", []))
    failed = len(log.get("failed", []))
    total = log.get("total", "?")
    model = log.get("model_name", "?")
    layers = log.get("target_layers", [])

    start_time = log.get("start_time", 0)
    last_update = log.get("last_update", 0)
    elapsed = last_update - start_time if start_time else 0

    remaining = total - completed - failed if isinstance(total, int) else "?"
    rate = completed / elapsed if elapsed > 0 else 0
    eta = remaining / rate if isinstance(remaining, int) and rate > 0 else "?"

    # Find current speaker
    last_completed = log["completed"][-1] if log.get("completed") else "N/A"
    # Extract speaker from utterance_id (TORGO_SPEAKER_...)
    current_speaker = "N/A"
    if last_completed != "N/A":
        parts = last_completed.split("_")
        if len(parts) >= 2:
            current_speaker = parts[1]

    print(f"\n{'='*60}")
    print(f"  SSL Extraction Status Monitor")
    print(f"{'='*60}")
    print(f"  Model:             {model}")
    print(f"  Target layers:     {layers}")
    print(f"  Completed:         {completed} / {total}")
    print(f"  Failed:            {failed}")
    print(f"  Remaining:         {remaining}")
    print(f"  Current speaker:   {current_speaker}")
    print(f"  Elapsed:           {elapsed / 60:.1f} min")
    print(f"  Rate:              {rate:.2f} utt/s")
    if isinstance(eta, float):
        print(f"  ETA:               {eta / 60:.1f} min")
    else:
        print(f"  ETA:               {eta}")
    print(f"  Last update:       {time.ctime(last_update)}")

    if log.get("failed"):
        print(f"\n  Failed utterances ({len(log['failed'])}):")
        for entry in log["failed"][:10]:
            if isinstance(entry, dict):
                print(f"    - {entry['utterance_id']}: {entry['reason']}")
            else:
                print(f"    - {entry}")
        if len(log["failed"]) > 10:
            print(f"    ... and {len(log['failed']) - 10} more")

    print(f"{'='*60}\n")


def run_sanity_checks(
    metadata_df: pd.DataFrame,
    index_path: Path,
    target_layers: list[int],
) -> bool:
    """
    Post-extraction sanity checks.

    Args:
        metadata_df: The metadata DataFrame.
        index_path: Path to ssl_index.json.
        target_layers: Expected layer indices.

    Returns:
        True if all checks pass.
    """
    print(f"\n{'='*60}")
    print(f"  Post-Extraction Sanity Checks")
    print(f"{'='*60}\n")

    all_pass = True

    # CHECK 1: Extraction completeness
    log_path = PROJECT_ROOT / "data" / "embeddings" / "wavlm_large" / "extraction_log.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            log = json.load(f)
        n_completed = len(log.get("completed", []))
        n_failed = len(log.get("failed", []))
        n_total = len(metadata_df)
        fail_rate = n_failed / n_total if n_total > 0 else 0
        ok = fail_rate < 0.05
        icon = "✅" if ok else "❌"
        print(f"{icon} CHECK 1 — Extraction completeness")
        print(f"   Completed: {n_completed}")
        print(f"   Failed: {n_failed}")
        print(f"   Remaining: {n_total - n_completed - n_failed}")
        print(f"   Fail rate: {fail_rate:.2%} {'(< 5%)' if ok else '(>= 5% — INVESTIGATE)'}")
        if not ok:
            all_pass = False
    else:
        print(f"❌ CHECK 1 — extraction_log.json not found")
        all_pass = False

    print()

    # CHECK 2: Embedding shapes correct
    if index_path.exists():
        with open(index_path, "r") as f:
            idx = json.load(f)
        items_with_emb = [(k, v) for k, v in idx.items() if v.get("wavlm_large")]
        if items_with_emb:
            random.seed(RANDOM_SEED)
            samples = random.sample(items_with_emb, min(10, len(items_with_emb)))
            shapes_ok = True
            print(f"✅ CHECK 2 — Embedding shapes (sampling {len(samples)})")
            for uid, info in samples:
                emb_path = PROJECT_ROOT / info["wavlm_large"]
                emb = np.load(str(emb_path))
                ok = emb.shape[0] == len(target_layers) and emb.shape[2] == 1024
                icon = "✅" if ok else "❌"
                print(f"   {icon} {uid}: shape={emb.shape}")
                if not ok:
                    shapes_ok = False
                    all_pass = False
            if not shapes_ok:
                all_pass = False
        else:
            print(f"❌ CHECK 2 — No embeddings found in index")
            all_pass = False
    else:
        print(f"❌ CHECK 2 — ssl_index.json not found")
        all_pass = False

    print()

    # CHECK 3: Embeddings not degenerate
    if index_path.exists():
        with open(index_path, "r") as f:
            idx = json.load(f)
        items_with_emb = [(k, v) for k, v in idx.items() if v.get("wavlm_large")]
        if len(items_with_emb) >= 50:
            random.seed(RANDOM_SEED)
            deg_samples = random.sample(items_with_emb, 50)
            embeddings_sample = []
            for uid, info in deg_samples:
                emb = np.load(str(PROJECT_ROOT / info["wavlm_large"]))
                # Use layer index 3 in the stacked array (which is layer 9)
                embeddings_sample.append(emb[3].mean(0))  # layer at index 3, mean pool
            embeddings_sample = np.stack(embeddings_sample)
            variance = embeddings_sample.var(0).mean()
            ok = variance > 0.01
            icon = "✅" if ok else "❌"
            print(f"{icon} CHECK 3 — Embedding variance: {variance:.4f}")
            print(f"   {'Not degenerate' if ok else 'DEGENERATE — investigate!'}")
            if not ok:
                all_pass = False
        else:
            print(f"⚠️  CHECK 3 — Only {len(items_with_emb)} embeddings, need 50+ for variance check")

    print()

    # CHECK 4: Disk space
    import shutil
    total, used, free = shutil.disk_usage("C:\\")
    gb_free = free / (2**30)
    ok = gb_free > 20
    icon = "✅" if ok else "⚠️"
    print(f"{icon} CHECK 4 — Disk free: {gb_free:.1f} GB")
    if not ok:
        print(f"   Low disk space warning!")

    print()

    if all_pass:
        print(f"{'='*60}")
        print(f"  ✅ PHASE 2 COMPLETE — All checks passed")
        print(f"{'='*60}")
    else:
        print(f"{'='*60}")
        print(f"  ❌ PHASE 2 INCOMPLETE — Some checks failed")
        print(f"{'='*60}")

    return all_pass


def main() -> None:
    """Main entry point for SSL extraction."""
    parser = argparse.ArgumentParser(
        description="Extract SSL embeddings from frozen WavLM-Large"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="wavlm_large",
        choices=list(MODELS_CONFIG.keys()),
        help="SSL model to use (default: wavlm_large)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show extraction status (can run from a second terminal)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only run sanity checks, skip extraction",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh (ignore previous extraction log)",
    )
    args = parser.parse_args()

    model_name = args.model
    output_dir = f"data/embeddings/{model_name}/"

    # Status mode — can run while extraction is happening
    if args.status:
        log_path = PROJECT_ROOT / output_dir / "extraction_log.json"
        print_extraction_status(str(log_path))
        return

    # Load metadata
    meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
    if not meta_path.exists():
        print(f"❌ Metadata not found: {meta_path}")
        print("Run Phase 1 first (experiments/02_explore_torgo.py + 03_preprocess_audio.py)")
        sys.exit(1)

    metadata_df = pd.read_csv(meta_path)
    print(f"Loaded metadata: {len(metadata_df)} utterances, "
          f"{metadata_df['speaker_id'].nunique()} speakers")

    # Verify-only mode
    if args.verify_only:
        index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
        run_sanity_checks(metadata_df, index_path, TARGET_LAYERS)
        return

    # ── Pre-flight checks ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Pre-flight Checks")
    print(f"{'='*60}")

    # Check GPU
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Cannot proceed.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"  GPU: {gpu_name}")
    print(f"  VRAM: {total_vram:.1f} GB")

    if total_vram < 8:
        print("  ⚠️  Low VRAM — extraction may fail on longer utterances")

    # Estimate extraction time
    avg_dur = metadata_df["duration_sec"].mean()
    n_utt = len(metadata_df)
    # Rough estimate: ~15 utterances/sec for short clips on RTX 4080
    est_time_min = n_utt / 15 / 60
    print(f"  Utterances: {n_utt}")
    print(f"  Avg duration: {avg_dur:.2f}s")
    print(f"  Estimated time: ~{est_time_min:.0f} min")
    print(f"  Output: {output_dir}")

    # Print model info
    print(f"\n  Loading model...")
    temp_extractor = SSLExtractor(model_name=model_name, device="cuda", use_fp16=True)
    info = temp_extractor.get_model_info()
    vram = temp_extractor.get_vram_usage()
    print(f"  Model: {info['hf_name']}")
    print(f"  Parameters: {info['n_parameters_M']}M")
    print(f"  VRAM (model): {vram['allocated_gb']:.2f} GB")
    print(f"  VRAM (free): {vram['free_gb']:.2f} GB")
    del temp_extractor
    torch.cuda.empty_cache()

    print(f"{'='*60}\n")

    # ── Run extraction ─────────────────────────────────────────────────
    log_path = extract_dataset_embeddings(
        metadata_df=metadata_df,
        model_name=model_name,
        output_dir=output_dir,
        target_layers=TARGET_LAYERS,
        batch_size=1,
        resume=not args.no_resume,
        project_root=PROJECT_ROOT,
    )

    # ── Build SSL index ────────────────────────────────────────────────
    print("Building SSL index...")
    build_ssl_index(
        metadata_df=metadata_df,
        embedding_dirs={model_name: output_dir},
        output_path="data/embeddings/ssl_index.json",
        project_root=PROJECT_ROOT,
    )

    # ── Run sanity checks ──────────────────────────────────────────────
    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    run_sanity_checks(metadata_df, index_path, TARGET_LAYERS)


if __name__ == "__main__":
    main()
