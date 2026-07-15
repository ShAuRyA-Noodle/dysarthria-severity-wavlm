"""Experiment 03 — Preprocess audio: resample to 16 kHz, normalize, and validate."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure emoji/Unicode print correctly on Windows terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from tqdm import tqdm

# Allow importing from src/ when run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

from src.data_pipeline import (  # noqa: E402
    build_torgo_metadata,
    create_data_splits,
    resample_and_normalize,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TORGO_RAW_DIR  = PROJECT_ROOT / "data" / "raw" / "torgo"
META_DIR       = PROJECT_ROOT / "data" / "processed" / "metadata"
AUDIO_OUT_ROOT = PROJECT_ROOT / "data" / "processed" / "audio_16khz"
SPLITS_PATH    = PROJECT_ROOT / "data" / "splits" / "splits.json"
LOG_DIR        = PROJECT_ROOT / "results" / "logs"

META_RAW_CSV    = META_DIR / "torgo_metadata_raw.csv"
META_FINAL_CSV  = META_DIR / "metadata_unified.csv"
FAIL_LOG        = LOG_DIR  / "preprocessing_failures.txt"

LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_OUT_ROOT.mkdir(parents=True, exist_ok=True)
META_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SR = 16_000
TARGET_DB = -20.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    set_global_seed(42)
    # ── Step 1: Build (or reload) raw metadata ────────────────────────────
    if META_RAW_CSV.exists():
        print(f"[load] Found existing {META_RAW_CSV.name} — loading.")
        df = pd.read_csv(str(META_RAW_CSV))
    else:
        print("[build] Raw metadata not found — crawling TORGO directory …")
        df = build_torgo_metadata(str(TORGO_RAW_DIR), mic_type="headMic")

    print(f"[info] Metadata rows: {len(df)}  |  Speakers: {df['speaker_id'].nunique()}")

    # ── Step 2: Resample + normalise every utterance ──────────────────────
    failures: list[str] = []
    processed_count = 0
    skipped_count = 0

    with tqdm(total=len(df), desc="Preprocessing audio", unit="file") as pbar:
        for idx, row in df.iterrows():
            raw_path  = row["filepath_raw"]
            proc_path = row["filepath_processed"]

            # Resumable: skip if output already exists
            if Path(proc_path).exists() and Path(proc_path).stat().st_size > 0:
                skipped_count += 1
                pbar.update(1)
                continue

            ok = resample_and_normalize(
                input_path=raw_path,
                output_path=proc_path,
                target_sr=TARGET_SR,
                target_db=TARGET_DB,
            )

            if ok:
                processed_count += 1
            else:
                failures.append(f"{row['utterance_id']} | raw={raw_path}")

            pbar.update(1)

    # ── Step 3: Log failures ───────────────────────────────────────────────
    if failures:
        with open(str(FAIL_LOG), "w", encoding="utf-8") as fh:
            fh.write(f"Preprocessing failures ({len(failures)} total)\n")
            fh.write("=" * 60 + "\n")
            for line in failures:
                fh.write(line + "\n")
        print(f"\n[warn] Failures logged → {FAIL_LOG}")

    # ── Step 4: Update filepath_processed for files that now exist ─────────
    def _check_exists(p) -> str:
        if not isinstance(p, str) or not p:
            return ""
        return p if (Path(p).exists() and Path(p).stat().st_size > 0) else ""

    df["filepath_processed"] = df["filepath_processed"].apply(_check_exists)

    # ── Step 5: Compute final duration from processed files ─────────────────
    # (stays the same after resampling, but update to reflect actual file state)
    # Re-use original duration_sec — resampling does not change duration.
    # Mark rows where processing failed (empty filepath) with duration 0
    failed_mask = df["filepath_processed"] == ""
    df.loc[failed_mask, "duration_sec"] = 0.0

    # ── Step 6: Save unified metadata ─────────────────────────────────────
    df["filepath_processed"] = df["filepath_processed"].fillna("")
    df.to_csv(str(META_FINAL_CSV), index=False)
    print(f"[save] metadata_unified.csv → {META_FINAL_CSV}")

    # ── Step 7: Create data splits (idempotent — skip if already exists) ───
    successful_df = df[df["filepath_processed"] != ""].copy()
    if SPLITS_PATH.exists():
        print(f"[skip] splits.json already exists — not regenerating.")
    else:
        create_data_splits(
            metadata_df=successful_df,
            output_path=str(SPLITS_PATH),
            random_seed=42,
        )

    # ── Step 8: Summary ────────────────────────────────────────────────────
    total_processed = processed_count + skipped_count
    total_failed    = len(failures)
    total_hours     = df.loc[df["filepath_processed"] != "", "duration_sec"].sum() / 3600

    # Storage used
    storage_bytes = sum(
        Path(p).stat().st_size
        for p in df["filepath_processed"]
        if p and Path(p).exists()
    )
    storage_gb = storage_bytes / (1024 ** 3)

    print()
    print(f"✅ Processed: {total_processed} utterances")
    print(f"⚠️  Failed:    {total_failed} utterances"
          + (f" (see {FAIL_LOG.name})" if total_failed else ""))
    print(f"📊 Total audio: {total_hours:.2f} hours")
    print(f"💾 Storage used: {storage_gb:.2f} GB")


if __name__ == "__main__":
    main()
