"""Italian Parkinson's Voice and Speech (IPVS) parser (Track D scaffold).

Builds metadata_ipvs.csv from the HuggingFace mirror layout
(`birgermoell/Italian_Parkinsons_Voice_and_Speech`). Wires the corpus into
the unified `cross_corpus` pipeline.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


# IPVS HuggingFace layout (subject to upstream changes — verify after download):
#   data/raw/ipvs/<group>/<speaker>/<task>/<filename>.wav
# Groups observed in the literature: PD, HC_Young, HC_Old.
# Tasks: sustained vowels (a,e,i,o,u), syllables (pa, ta, ka), read text.
# Speaker IDs are anonymized integers within each group.

GROUP_ALIAS = {
    "PD": "PD",
    "Parkinson": "PD",
    "Patient": "PD",
    "HC_Young": "HC_Young",
    "Young_Healthy": "HC_Young",
    "HC_Old": "HC_Old",
    "Old_Healthy": "HC_Old",
    "Control": "HC_Old",
}


def parse_ipvs_filename(path: Path) -> dict | None:
    """Parse IPVS path like `data/raw/ipvs/PD/03/vowel_a/PD_03_vowel_a_001.wav`.

    Returns dict with speaker_id, group, task, item, severity_score.
    Returns None if the path does not match the expected pattern.
    """
    parts = path.parts
    try:
        idx = parts.index("ipvs")
    except ValueError:
        return None
    rel = parts[idx + 1 :]
    if len(rel) < 2:
        return None
    raw_group = rel[0]
    group = GROUP_ALIAS.get(raw_group, raw_group)
    speaker = rel[1] if len(rel) >= 2 else "unknown"
    task = rel[2] if len(rel) >= 3 else "unknown"
    stem = path.stem
    speaker_id = f"IPVS_{group}_{speaker}"
    severity = 0 if group.startswith("HC") else 1  # placeholder; refine via UPDRS xlsx
    return {
        "utterance_id": f"IPVS_{group}_{speaker}_{stem}",
        "speaker_id": speaker_id,
        "group": group,
        "task": task,
        "item": stem,
        "severity_score": severity,
        "is_control": group.startswith("HC"),
        "filepath_processed": str(path),
    }


def build_ipvs_metadata(
    raw_dir: Path,
    out_csv: Path,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
) -> pd.DataFrame:
    """Walk `data/raw/ipvs/`, build metadata CSV, return DataFrame."""
    raw_dir = Path(raw_dir)
    rows: list[dict] = []
    for ext in extensions:
        for path in raw_dir.rglob(f"*{ext}"):
            row = parse_ipvs_filename(path)
            if row is not None:
                rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"no audio files found under {raw_dir}")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def load_updrs_xlsx_if_available(xlsx_path: Path) -> pd.DataFrame | None:
    """Optional UPDRS-III speech subscale loader. IPVS HF mirror may or may
    not include the original Excel sheet — handle gracefully if missing.
    """
    if not xlsx_path.exists():
        return None
    try:
        df = pd.read_excel(xlsx_path)
    except Exception:  # noqa: BLE001
        return None
    return df
