"""VCTK Corpus 0.92 parser.

Produces speaker metadata + utterance manifest. We use mic1 (DPA 4035) only
for acoustic consistency. Restrict to the top-K most-shared sentences for
the ATD LOCO experiment so per-sentence references have many contributors.

Outputs:
  data/processed/metadata/metadata_vctk.csv  (per-utterance)
  data/processed/metadata/vctk_speakers.csv  (per-speaker)
"""
from __future__ import annotations

import logging
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

VCTK_ROOT = PROJECT_ROOT / "DS_10283_3443" / "VCTK-Corpus-0.92"
AUDIO_ROOT = VCTK_ROOT / "wav48_silence_trimmed"
TXT_ROOT = VCTK_ROOT / "txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _parse_speaker_info() -> pd.DataFrame:
    f = VCTK_ROOT / "speaker-info.txt"
    rows = []
    with open(f, encoding="utf-8") as fh:
        lines = fh.readlines()
    for line in lines[1:]:  # skip header
        line = line.rstrip("\n")
        if not line.strip():
            continue
        # split on 2+ whitespace to keep multi-word REGION + COMMENTS together
        parts = re.split(r"\s{2,}", line.strip())
        if len(parts) < 4:
            parts = line.split()
        sid = parts[0]
        try:
            age = int(parts[1])
        except (ValueError, IndexError):
            age = None
        gender = parts[2] if len(parts) > 2 else None
        accent = parts[3] if len(parts) > 3 else None
        region = parts[4] if len(parts) > 4 else None
        comments = parts[5] if len(parts) > 5 else None
        rows.append({"speaker_id": sid, "age": age, "gender": gender,
                     "accent": accent, "region": region, "comments": comments})
    return pd.DataFrame(rows)


def main(top_k_shared: int = 50) -> None:
    if not VCTK_ROOT.exists():
        raise SystemExit(f"VCTK root missing: {VCTK_ROOT}")

    spk_df = _parse_speaker_info()
    logger.info("speaker-info: %d rows", len(spk_df))

    # Find shared sentences first.
    sid_counts: Counter = Counter()
    for spk_dir in TXT_ROOT.iterdir():
        if not spk_dir.is_dir():
            continue
        for txt in spk_dir.glob("*.txt"):
            sid = txt.stem.split("_")[-1]
            sid_counts[sid] += 1
    top_shared = {sid for sid, _ in sid_counts.most_common(top_k_shared)}
    logger.info("selected %d most-shared sentence IDs", len(top_shared))

    # Build per-utterance manifest restricted to mic1 + shared sentences.
    rows = []
    for spk in spk_df["speaker_id"]:
        spk_audio_dir = AUDIO_ROOT / spk
        if not spk_audio_dir.exists():
            continue
        for flac in sorted(spk_audio_dir.glob(f"{spk}_*_mic1.flac")):
            stem = flac.stem  # p225_001_mic1
            parts = stem.split("_")
            if len(parts) < 3:
                continue
            sid = parts[1]
            if sid not in top_shared:
                continue
            txt_path = TXT_ROOT / spk / f"{spk}_{sid}.txt"
            prompt = ""
            if txt_path.exists():
                prompt = txt_path.read_text(encoding="utf-8").strip()
            try:
                info = sf.info(str(flac))
                dur = info.duration
            except Exception:
                dur = None
            rows.append({
                "speaker_id": spk, "sentence_id": sid,
                "wav_path": str(flac.relative_to(PROJECT_ROOT)),
                "txt_path": str(txt_path.relative_to(PROJECT_ROOT)) if txt_path.exists() else "",
                "prompt": prompt,
                "duration_s": dur,
            })
    md = pd.DataFrame(rows).merge(spk_df, on="speaker_id", how="left")
    md_out = PROJECT_ROOT / "data/processed/metadata/metadata_vctk.csv"
    spk_out = PROJECT_ROOT / "data/processed/metadata/vctk_speakers.csv"
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md.to_csv(md_out, index=False)
    spk_df.to_csv(spk_out, index=False)
    logger.info("wrote %s (%d rows)", md_out, len(md))
    logger.info("wrote %s (%d speakers)", spk_out, len(spk_df))
    print("\naccent distribution:")
    print(spk_df["accent"].value_counts().to_string())


if __name__ == "__main__":
    main()
