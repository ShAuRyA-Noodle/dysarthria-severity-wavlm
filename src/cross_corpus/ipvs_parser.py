"""Italian Parkinson's Voice and Speech (IPVS) parser (Track D).

Builds ``metadata_ipvs.csv`` from the actual HuggingFace mirror layout
observed in ``birgermoell/Italian_Parkinsons_Voice_and_Speech``::

    data/raw/ipvs/italian_parkinson/<N> <Group label>/<full speaker name>/<task_code>.wav

Real example::

    data/raw/ipvs/italian_parkinson/15 Young Healthy Control/Alberto R/B1LBULCAAS94M100120171015.wav

Speaker dirs use full first-name + last-initial which is not safe to
publish. We hash to an 8-char SHA-1 prefix and prepend a group code.
A local mapping is preserved in ``metadata_ipvs_speaker_map.csv``
(gitignored — never commit) for our own debugging.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

# Group label patterns (top-level under italian_parkinson/).
# Numbers prefix (e.g. "15 ", "22 ", "28 ") refer to speaker counts.
GROUP_PATTERNS = {
    "Young Healthy": "HC_Young",
    "Elderly Healthy": "HC_Old",
    "Parkinson": "PD",
    "Patient": "PD",
    "Control": "HC_Old",  # ambiguous fallback if "Healthy" missing
}

# IPVS task codes inferred from the file-stem prefix conventions documented
# in Dimauro et al. 2017 / Suppa et al. 2022. Refine after manual audit.
TASK_PREFIX_TO_TASK = {
    "B1": "sustained_vowel_a",
    "B2": "sustained_vowel_e",
    "B3": "sustained_vowel_i",
    "B4": "sustained_vowel_o",
    "B5": "sustained_vowel_u",
    "VS": "vowel_sequence",
    "PR": "read_passage",
    "FR": "read_phrase",
    "DDK": "ddk_pataka",
    "WD": "isolated_word",
}


def _classify_group(group_dir_name: str) -> str:
    """Map a top-level dir name (e.g. '15 Young Healthy Control') to a code."""
    name = re.sub(r"^\d+\s+", "", group_dir_name).strip()
    for key, code in GROUP_PATTERNS.items():
        if key.lower() in name.lower():
            return code
    return "UNKNOWN"


def _speaker_hash(full_name: str, group: str) -> str:
    h = hashlib.sha1(full_name.strip().encode("utf-8")).hexdigest()[:8]
    return f"IPVS_{group}_{h}"


def _classify_task(stem: str) -> str:
    upper = stem.upper()
    for prefix, task in TASK_PREFIX_TO_TASK.items():
        if upper.startswith(prefix):
            return task
    return "unknown"


def parse_ipvs_filename(path: Path) -> dict | None:
    """Parse one IPVS audio path. Returns dict or None if layout doesn't match."""
    parts = path.parts
    try:
        idx = parts.index("italian_parkinson")
    except ValueError:
        return None
    rel = parts[idx + 1:]
    if len(rel) < 3:  # group / speaker / file
        return None
    group_dir = rel[0]
    speaker_name = rel[1]
    group = _classify_group(group_dir)
    if group == "UNKNOWN":
        return None
    speaker_id = _speaker_hash(speaker_name, group)
    stem = path.stem
    task = _classify_task(stem)
    severity = 0 if group.startswith("HC") else 1  # binary placeholder
    return {
        "utterance_id": f"{speaker_id}_{stem}",
        "speaker_id": speaker_id,
        "group": group,
        "speaker_name_raw": speaker_name,  # kept locally for the map; dropped on commit
        "task": task,
        "task_code": stem[:3].upper(),
        "item": stem,
        "severity_score": severity,
        "is_control": group.startswith("HC"),
        "filepath_processed": str(path),
    }


def build_ipvs_metadata(
    raw_dir: Path,
    out_csv: Path,
    extensions: tuple[str, ...] = (".wav", ".flac", ".mp3"),
    write_speaker_map: bool = True,
) -> pd.DataFrame:
    """Walk ``data/raw/ipvs/``, build metadata CSV, return DataFrame.

    Two CSVs are produced:
      * ``out_csv`` — the public metadata (anonymized speaker_id only).
      * ``out_csv.with_name(out_csv.stem + '_speaker_map.csv')`` — the
        local mapping ``speaker_id → real name``. Gitignore this.
    """
    raw_dir = Path(raw_dir)
    out_csv = Path(out_csv)
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
    public_cols = [c for c in df.columns if c != "speaker_name_raw"]
    df[public_cols].to_csv(out_csv, index=False)

    if write_speaker_map:
        map_path = out_csv.with_name(out_csv.stem + "_speaker_map.csv")
        df[["speaker_id", "speaker_name_raw", "group"]].drop_duplicates(
            subset=["speaker_id"]
        ).to_csv(map_path, index=False)

    return df[public_cols]


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
