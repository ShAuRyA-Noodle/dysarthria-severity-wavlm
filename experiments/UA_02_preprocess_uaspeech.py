"""Experiment UA-02 — Preprocess UA-Speech audio.

Resumable pipeline:
  1. Build raw metadata via uaspeech_parser
  2. Resample each WAV to 16 kHz
  3. Normalize amplitude to -20 dBFS
  4. Save to data/processed/audio_16khz/uaspeech/{speaker_id}/
  5. Build metadata CSV: data/processed/metadata/metadata_uaspeech.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure Unicode prints correctly on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uaspeech_parser import build_uaspeech_metadata  # noqa: E402
from src.data_pipeline import resample_and_normalize       # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Raw audio source directories
C_AUDIO_DIR  = PROJECT_ROOT / "UASpeech_original_C" / "UASpeech" / "audio" / "original"
FM_AUDIO_DIR = PROJECT_ROOT / "UASpeech_original_FM" / "UASpeech" / "audio" / "original"

# Alternate FM locations
FM_ALT_DIRS = [
    PROJECT_ROOT / "UASpeech_original_FM" / "audio" / "original",
    PROJECT_ROOT / "UASpeech_original_FM" / "UASpeech",
]

# Output paths
META_DIR        = PROJECT_ROOT / "data" / "processed" / "metadata"
AUDIO_OUT_ROOT  = PROJECT_ROOT / "data" / "processed" / "audio_16khz" / "uaspeech"
LOG_DIR         = PROJECT_ROOT / "results" / "logs"

META_RAW_CSV    = META_DIR / "uaspeech_metadata_raw.csv"
META_FINAL_CSV  = META_DIR / "metadata_uaspeech.csv"
FAIL_LOG        = LOG_DIR  / "uaspeech_preprocessing_failures.txt"

# Processing parameters (same as TORGO)
TARGET_SR = 16_000
TARGET_DB = -20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_fm_audio_dir() -> Path | None:
    """Find the actual FM audio directory after extraction."""
    if FM_AUDIO_DIR.exists():
        return FM_AUDIO_DIR
    for alt in FM_ALT_DIRS:
        if alt.exists() and list(alt.rglob("*.wav")):
            return alt
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Build (or reload) raw metadata ────────────────────────────
    if META_RAW_CSV.exists():
        print(f"[load] Found existing {META_RAW_CSV.name} -- loading.")
        df = pd.read_csv(str(META_RAW_CSV))
    else:
        print("[build] Raw metadata not found -- crawling UA-Speech directories...")

        root_dirs = []
        if C_AUDIO_DIR.exists():
            root_dirs.append(str(C_AUDIO_DIR))
            print(f"  C-channel: {C_AUDIO_DIR}")
        else:
            print(f"  [warn] C-channel not found: {C_AUDIO_DIR}")

        fm_dir = find_fm_audio_dir()
        if fm_dir:
            root_dirs.append(str(fm_dir))
            print(f"  FM-channel: {fm_dir}")
        else:
            print("  [warn] FM-channel audio directory not found!")

        if not root_dirs:
            print("[error] No audio directories found. Aborting.")
            return

        df = build_uaspeech_metadata(
            root_dirs=root_dirs,
            output_csv=str(META_RAW_CSV),
            mic_filter="M3",
            processed_audio_root=str(AUDIO_OUT_ROOT),
        )

    print(f"[info] Metadata rows: {len(df)}  |  Speakers: {df['speaker_id'].nunique()}")

    # ── Step 2: Resample + normalise every utterance ──────────────────────
    failures: list[str] = []
    processed_count = 0
    skipped_count = 0

    with tqdm(total=len(df), desc="Preprocessing UA-Speech audio", unit="file") as pbar:
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
            fh.write(f"UA-Speech preprocessing failures ({len(failures)} total)\n")
            fh.write("=" * 60 + "\n")
            for line in failures:
                fh.write(line + "\n")
        print(f"\n[warn] {len(failures)} failures logged -> {FAIL_LOG}")

    # ── Step 4: Validate processed files ──────────────────────────────────
    def _check_exists(p) -> str:
        if not isinstance(p, str) or not p:
            return ""
        return p if (Path(p).exists() and Path(p).stat().st_size > 0) else ""

    df["filepath_processed"] = df["filepath_processed"].apply(_check_exists)

    # Mark rows where processing failed with duration 0
    failed_mask = df["filepath_processed"] == ""
    df.loc[failed_mask, "duration_sec"] = 0.0

    # Add sample_rate column (all processed files are 16kHz)
    df["sample_rate"] = 0
    df.loc[~failed_mask, "sample_rate"] = TARGET_SR

    # ── Step 5: Save final metadata ───────────────────────────────────────
    # Select columns matching the specification
    final_cols = [
        "utterance_id", "speaker_id", "word", "intelligibility_score",
        "severity_proxy", "filepath_processed", "duration_sec",
        "is_control", "sample_rate",
    ]
    # Keep extra columns that exist for debugging
    extra_cols = [c for c in df.columns if c not in final_cols]
    out_df = df[final_cols + extra_cols].copy()
    out_df.to_csv(str(META_FINAL_CSV), index=False)
    print(f"[save] metadata_uaspeech.csv -> {META_FINAL_CSV}")

    # ── Step 6: Summary ────────────────────────────────────────────────────
    total_processed = processed_count + skipped_count
    total_failed    = len(failures)
    successful_df   = out_df[out_df["filepath_processed"] != ""]
    total_hours     = successful_df["duration_sec"].sum() / 3600

    # Storage used
    storage_bytes = sum(
        Path(p).stat().st_size
        for p in successful_df["filepath_processed"]
        if p and Path(p).exists()
    )
    storage_gb = storage_bytes / (1024 ** 3)

    print()
    print(f"Processed:          {total_processed} utterances")
    print(f"Failed:             {total_failed} utterances"
          + (f" (see {FAIL_LOG.name})" if total_failed else ""))
    print(f"Unique speakers:    {successful_df['speaker_id'].nunique()}")
    print(f"  Dysarthric:       {successful_df[~successful_df.is_control]['speaker_id'].nunique()}")
    print(f"  Control:          {successful_df[successful_df.is_control]['speaker_id'].nunique()}")
    print(f"Total audio:        {total_hours:.2f} hours")
    print(f"Storage used:       {storage_gb:.2f} GB")


if __name__ == "__main__":
    main()
