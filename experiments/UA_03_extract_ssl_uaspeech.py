"""
Experiment UA-03 — Extract SSL embeddings from WavLM-Large for ALL UA-Speech speakers.

Reuses src/ssl_extractor.py SSLExtractor class (frozen, no modifications).
Saves .npy to data/embeddings/wavlm_large/{speaker_id}/{utterance_id}.npy
Uses SAME extraction_log.json as TORGO, with "uaspeech_{utterance_id}" keys.

Usage:
    python experiments/UA_03_extract_ssl_uaspeech.py             # Full extraction
    python experiments/UA_03_extract_ssl_uaspeech.py --status    # Check status
    python experiments/UA_03_extract_ssl_uaspeech.py --verify    # Verify only

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
import torchaudio

# Ensure Unicode on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# ── Project root ───────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

from src.ssl_extractor import (  # noqa: E402
    SSLExtractor,
    TARGET_LAYERS,
    _load_extraction_log,
    _save_extraction_log,
)
from src.uaspeech_parser import UA_INTELLIGIBILITY, UA_SEVERITY  # noqa: E402

# ── Logging ────────────────────────────────────────────────────────────
LOG_DIR = PROJECT_ROOT / "results" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "ua_extraction.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
MODEL_NAME = "wavlm_large"
OUTPUT_DIR = PROJECT_ROOT / "data" / "embeddings" / "wavlm_large"
LOG_KEY_PREFIX = "uaspeech_"


# ---------------------------------------------------------------------------
# Status printer
# ---------------------------------------------------------------------------

def print_extraction_status() -> None:
    """Print current extraction status from extraction_log.json."""
    log_path = OUTPUT_DIR / "extraction_log.json"
    if not log_path.exists():
        print("extraction_log.json not found. Extraction may not have started.")
        return

    log = _load_extraction_log(log_path)
    completed = log.get("completed", [])
    failed = log.get("failed", [])

    ua_completed = [k for k in completed if k.startswith(LOG_KEY_PREFIX)]
    torgo_completed = [k for k in completed if k.startswith("TORGO_")]

    print(f"\n{'='*60}")
    print(f"  SSL Extraction Status")
    print(f"{'='*60}")
    print(f"  TORGO completed:      {len(torgo_completed)}")
    print(f"  UA-Speech completed:  {len(ua_completed)}")
    print(f"  Total completed:      {len(completed)}")
    print(f"  Failed:               {len(failed)}")
    print(f"  Last update:          {time.ctime(log.get('last_update', 0))}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def run_extraction(resume: bool = True) -> None:
    """Extract WavLM-Large embeddings for all UA-Speech utterances."""

    # ── Load metadata ─────────────────────────────────────────────────
    meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv"
    if not meta_path.exists():
        print(f"Metadata not found: {meta_path}")
        print("Run Phase UA-1 first (UA_02_preprocess_uaspeech.py)")
        sys.exit(1)

    meta_df = pd.read_csv(meta_path)
    # Filter out rows without processed audio
    meta_df = meta_df[meta_df["filepath_processed"].notna() & (meta_df["filepath_processed"] != "")]
    print(f"Loaded metadata: {len(meta_df)} utterances, "
          f"{meta_df['speaker_id'].nunique()} speakers")

    # ── Load/create extraction log (shared with TORGO) ────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "extraction_log.json"
    log = _load_extraction_log(log_path)

    # Determine what needs extraction
    completed_set = set(log.get("completed", []))

    all_items = []
    for _, row in meta_df.iterrows():
        utt_id = row["utterance_id"]
        log_key = f"{LOG_KEY_PREFIX}{utt_id}"
        all_items.append({
            "utterance_id": utt_id,
            "log_key": log_key,
            "speaker_id": row["speaker_id"],
            "filepath_processed": row["filepath_processed"],
        })

    if resume:
        remaining = [item for item in all_items if item["log_key"] not in completed_set]
        n_already = len(all_items) - len(remaining)
        print(f"Resume mode: {n_already} already done, {len(remaining)} remaining")
    else:
        remaining = all_items

    if not remaining:
        print("All UA-Speech utterances already extracted. Nothing to do.")
        _save_extraction_log(log, log_path)
        return

    # ── Pre-flight checks ─────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("CUDA not available. Cannot proceed.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"GPU: {gpu_name} ({total_vram:.1f} GB VRAM)")

    # Enable cudnn benchmark
    torch.backends.cudnn.benchmark = True

    # ── Initialize extractor (FP16) ───────────────────────────────────
    extractor = SSLExtractor(model_name=MODEL_NAME, device="cuda", use_fp16=True)

    # ── Extract ───────────────────────────────────────────────────────
    from tqdm import tqdm

    start_time = time.time()
    n_done = 0
    current_speaker = None
    new_failures = []

    print(f"\n{'='*60}")
    print(f"  UA-Speech SSL Extraction: {MODEL_NAME}")
    print(f"  Remaining: {len(remaining)} / {len(all_items)} utterances")
    print(f"  Target layers: {TARGET_LAYERS}")
    print(f"{'='*60}\n")

    for i, item in enumerate(tqdm(remaining, desc="Extracting UA-Speech")):
        utt_id = item["utterance_id"]
        log_key = item["log_key"]
        speaker_id = item["speaker_id"]

        try:
            # Clear VRAM cache between speakers
            if speaker_id != current_speaker:
                if current_speaker is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                current_speaker = speaker_id

            # Resolve audio path
            audio_path = Path(item["filepath_processed"])
            if not audio_path.is_absolute():
                audio_path = PROJECT_ROOT / audio_path
            if not audio_path.exists():
                logger.warning(f"Audio not found: {audio_path}")
                log["failed"].append({"utterance_id": log_key, "reason": "file_not_found"})
                _save_extraction_log(log, log_path)
                continue

            # Load audio
            waveform, sr = torchaudio.load(str(audio_path))
            waveform = waveform.squeeze(0).numpy()  # (T_samples,)

            # Extract embeddings from all target layers
            layer_embeddings = extractor.extract_all_layers(
                waveform, sample_rate=sr, target_layers=TARGET_LAYERS
            )

            # Stack into (n_layers, T_frames, 1024)
            stacked = np.stack(
                [layer_embeddings[l] for l in TARGET_LAYERS], axis=0
            )

            # Save .npy
            speaker_dir = OUTPUT_DIR / speaker_id
            speaker_dir.mkdir(parents=True, exist_ok=True)
            npy_path = speaker_dir / f"{utt_id}.npy"
            np.save(str(npy_path), stacked)

            # Update log with uaspeech_ prefixed key
            log["completed"].append(log_key)
            n_done += 1

            # Progress every 50 utterances
            if n_done % 50 == 0:
                elapsed = time.time() - start_time
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(remaining) - (i + 1)) / rate if rate > 0 else 0
                vram = extractor.get_vram_usage()
                n_total_completed = sum(1 for k in log["completed"] if k.startswith(LOG_KEY_PREFIX))
                print(
                    f"\n  [{n_total_completed}/{len(all_items)}] "
                    f"Rate: {rate:.1f} utt/s | "
                    f"ETA: {eta / 60:.1f} min | "
                    f"VRAM: {vram['allocated_gb']:.2f} GB | "
                    f"Failed: {len(new_failures)}"
                )
                _save_extraction_log(log, log_path)

        except torch.cuda.OutOfMemoryError:
            logger.error(f"CUDA OOM on {utt_id} -- clearing cache and skipping")
            torch.cuda.empty_cache()
            log["failed"].append({"utterance_id": log_key, "reason": "cuda_oom"})
            new_failures.append(log_key)
            _save_extraction_log(log, log_path)
            continue

        except Exception as e:
            logger.error(f"Failed on {utt_id}: {e}")
            log["failed"].append({"utterance_id": log_key, "reason": str(e)[:200]})
            new_failures.append(log_key)
            _save_extraction_log(log, log_path)
            continue

    # Final save
    log["last_update"] = time.time()
    _save_extraction_log(log, log_path)

    # Free GPU
    del extractor
    torch.cuda.empty_cache()

    # Summary
    total_time = time.time() - start_time
    n_ua_completed = sum(1 for k in log["completed"] if k.startswith(LOG_KEY_PREFIX))
    print(f"\n{'='*60}")
    print(f"  UA-Speech Extraction Complete")
    print(f"  Total time: {total_time / 60:.1f} min")
    print(f"  UA-Speech completed: {n_ua_completed} / {len(all_items)}")
    print(f"  New failures: {len(new_failures)}")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Build unified ssl_index.json
# ---------------------------------------------------------------------------

def build_unified_ssl_index() -> None:
    """Build ssl_index.json with both TORGO and UA-Speech entries.

    Adds "dataset" field: "torgo" or "uaspeech" to each entry.
    """
    from src.ssl_extractor import TORGO_SEVERITY

    print("Building unified SSL index...")

    index = {}

    # ── TORGO entries ─────────────────────────────────────────────────
    torgo_meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
    if torgo_meta_path.exists():
        torgo_df = pd.read_csv(torgo_meta_path)
        print(f"  TORGO metadata: {len(torgo_df)} utterances")

        for _, row in torgo_df.iterrows():
            utt_id = row["utterance_id"]
            speaker_id = row["speaker_id"]
            npy_path = OUTPUT_DIR / speaker_id / f"{utt_id}.npy"

            entry = {
                "utterance_id": utt_id,
                "dataset": "torgo",
                "speaker": speaker_id,
                "word": str(row.get("word", "")),
                "severity": int(TORGO_SEVERITY.get(speaker_id, -1)),
                "is_control": bool(TORGO_SEVERITY.get(speaker_id, -1) == 0),
                "duration_sec": float(row.get("duration_sec", 0)),
            }

            if npy_path.exists():
                rel_path = str(npy_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                entry["wavlm_large"] = rel_path
                entry["filepath"] = rel_path

                # Read shape once
                try:
                    emb = np.load(str(npy_path))
                    entry["shape"] = list(emb.shape)
                    entry["layer_indices"] = TARGET_LAYERS
                except Exception:
                    pass
            else:
                entry["wavlm_large"] = None
                entry["filepath"] = None

            index[utt_id] = entry
    else:
        print("  [warn] TORGO metadata not found, skipping TORGO entries")

    # ── UA-Speech entries ─────────────────────────────────────────────
    ua_meta_path = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv"
    if ua_meta_path.exists():
        ua_df = pd.read_csv(ua_meta_path)
        ua_df = ua_df[ua_df["filepath_processed"].notna() & (ua_df["filepath_processed"] != "")]
        print(f"  UA-Speech metadata: {len(ua_df)} utterances")

        for _, row in ua_df.iterrows():
            utt_id = row["utterance_id"]
            speaker_id = row["speaker_id"]
            npy_path = OUTPUT_DIR / speaker_id / f"{utt_id}.npy"

            entry = {
                "utterance_id": utt_id,
                "dataset": "uaspeech",
                "speaker": speaker_id,
                "word": str(row.get("word", "")),
                "severity": int(UA_SEVERITY.get(speaker_id, -1)),
                "intelligibility_score": float(UA_INTELLIGIBILITY.get(speaker_id, -1)),
                "is_control": bool(row.get("is_control", False)),
                "duration_sec": float(row.get("duration_sec", 0)),
            }

            if npy_path.exists():
                rel_path = str(npy_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                entry["wavlm_large"] = rel_path
                entry["filepath"] = rel_path

                try:
                    emb = np.load(str(npy_path))
                    entry["shape"] = list(emb.shape)
                    entry["layer_indices"] = TARGET_LAYERS
                except Exception:
                    pass
            else:
                entry["wavlm_large"] = None
                entry["filepath"] = None

            index[utt_id] = entry
    else:
        print("  [warn] UA-Speech metadata not found, skipping UA-Speech entries")

    # ── Save ──────────────────────────────────────────────────────────
    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    n_torgo = sum(1 for v in index.values() if v.get("dataset") == "torgo")
    n_ua = sum(1 for v in index.values() if v.get("dataset") == "uaspeech")
    n_with_emb = sum(1 for v in index.values() if v.get("wavlm_large"))

    print(f"\n  SSL Index saved: {index_path}")
    print(f"  Total entries:     {len(index)}")
    print(f"  TORGO entries:     {n_torgo}")
    print(f"  UA-Speech entries: {n_ua}")
    print(f"  With embeddings:   {n_with_emb}")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_verification() -> bool:
    """Run verification gate checks."""
    print(f"\n{'='*60}")
    print(f"  UA-Speech SSL Extraction Verification")
    print(f"{'='*60}\n")

    index_path = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
    if not index_path.exists():
        print("ssl_index.json not found!")
        return False

    with open(index_path) as f:
        index = json.load(f)

    uaspeech_entries = [v for v in index.values() if v.get("dataset") == "uaspeech"]
    torgo_entries = [v for v in index.values() if v.get("dataset") == "torgo"]

    print(f"TORGO utterances:      {len(torgo_entries)}")
    print(f"UA-Speech utterances:  {len(uaspeech_entries)}")

    # Check embeddings exist
    ua_with_emb = [v for v in uaspeech_entries if v.get("wavlm_large")]
    print(f"UA-Speech with emb:    {len(ua_with_emb)}")

    if not ua_with_emb:
        print("No UA-Speech embeddings found!")
        return False

    # Check 10 random samples
    random.seed(42)
    sample = random.sample(ua_with_emb, min(10, len(ua_with_emb)))
    all_ok = True
    for entry in sample:
        emb_path = PROJECT_ROOT / entry["filepath"]
        if not emb_path.exists():
            print(f"  MISSING: {entry['utterance_id']}")
            all_ok = False
            continue
        emb = np.load(str(emb_path))
        ok = emb.shape[0] == 9 and emb.shape[2] == 1024
        icon = "OK" if ok else "FAIL"
        print(f"  {icon} {entry['utterance_id']}: shape={emb.shape}")
        if not ok:
            all_ok = False

    # Fail rate
    total_expected = len(uaspeech_entries)
    fail_rate = 1 - len(ua_with_emb) / total_expected if total_expected > 0 else 1.0
    print(f"\nFail rate: {fail_rate*100:.1f}%")

    if fail_rate < 0.05 and all_ok:
        print("PHASE UA-2 COMPLETE")
        return True
    else:
        print("Too many failures or shape errors")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_global_seed(42)
    parser = argparse.ArgumentParser(
        description="Extract WavLM-Large SSL embeddings for UA-Speech"
    )
    parser.add_argument("--status", action="store_true", help="Show extraction status")
    parser.add_argument("--verify", action="store_true", help="Run verification only")
    parser.add_argument("--index-only", action="store_true", help="Only rebuild ssl_index.json")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh")
    args = parser.parse_args()

    if args.status:
        print_extraction_status()
        return

    if args.verify:
        run_verification()
        return

    if args.index_only:
        build_unified_ssl_index()
        return

    # ── Run extraction ─────────────────────────────────────────────────
    run_extraction(resume=not args.no_resume)

    # ── Build unified SSL index ────────────────────────────────────────
    build_unified_ssl_index()

    # ── Verify ────────────────────────────────────────────────────────
    run_verification()


if __name__ == "__main__":
    main()
