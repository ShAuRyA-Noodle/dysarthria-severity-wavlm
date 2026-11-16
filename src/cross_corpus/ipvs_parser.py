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

# IPVS task codes derived from empirical inspection of the HuggingFace
# mirror (Dimauro 2019). Stem starts with 2-3 letters + intensity digit.
# Vowels: VA/VE/VI/VO/VU at intensity 1 or 2.
# PR: read passage. FB: fixed phrase. B*/D*: DDK / disyllable variants.
TASK_PREFIX_TO_TASK = {
    "VA1": "sustained_vowel_a", "VA2": "sustained_vowel_a",
    "VE1": "sustained_vowel_e", "VE2": "sustained_vowel_e",
    "VI1": "sustained_vowel_i", "VI2": "sustained_vowel_i",
    "VO1": "sustained_vowel_o", "VO2": "sustained_vowel_o",
    "VU1": "sustained_vowel_u", "VU2": "sustained_vowel_u",
    "PR1": "read_passage", "PR2": "read_passage",
    "FB1": "read_phrase",  "FB2": "read_phrase",
    # DDK syllable buckets (B = bilabial, D = alveolar; suffix = vowel context).
    "B1A": "ddk_baba", "B1E": "ddk_baba", "B1I": "ddk_baba",
    "B1O": "ddk_baba", "B1U": "ddk_baba", "B1L": "ddk_baba", "B1G": "ddk_baba",
    "B1V": "ddk_baba",
    "B2A": "ddk_baba", "B2E": "ddk_baba", "B2I": "ddk_baba",
    "B2O": "ddk_baba", "B2U": "ddk_baba", "B2L": "ddk_baba", "B2G": "ddk_baba",
    "B2V": "ddk_baba",
    "D1A": "ddk_dada", "D1E": "ddk_dada", "D1I": "ddk_dada",
    "D1O": "ddk_dada", "D1U": "ddk_dada", "D1L": "ddk_dada", "D1G": "ddk_dada",
    "D1V": "ddk_dada",
    "D2A": "ddk_dada", "D2E": "ddk_dada", "D2I": "ddk_dada",
    "D2O": "ddk_dada", "D2U": "ddk_dada", "D2L": "ddk_dada", "D2G": "ddk_dada",
    "D2V": "ddk_dada",
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
    # Try 3-char prefix first (most specific: VA1, PR1, B1A, D2V, ...),
    # then 2-char (less specific) as a fallback.
    for n in (3, 2):
        prefix = upper[:n]
        if prefix in TASK_PREFIX_TO_TASK:
            return TASK_PREFIX_TO_TASK[prefix]
    return "unknown"


_RANGE_PATTERN = re.compile(r"^\d+-\d+$")


def parse_ipvs_filename(path: Path) -> dict | None:
    """Parse one IPVS audio path. Returns dict or None if layout doesn't match.

    Two layouts observed in the HuggingFace mirror:

    * HC groups (3 levels): ``italian_parkinson/<group>/<speaker_name>/<file>.wav``
    * PD group  (4 levels): ``italian_parkinson/<group>/<range>/<speaker_name>/<file>.wav``

    Detection: if the component after the group matches ``\d+-\d+`` we
    treat it as a range bucket and read the speaker_name one level deeper.
    """
    parts = path.parts
    try:
        idx = parts.index("italian_parkinson")
    except ValueError:
        return None
    rel = parts[idx + 1:]
    if len(rel) < 3:
        return None
    group_dir = rel[0]
    group = _classify_group(group_dir)
    if group == "UNKNOWN":
        return None

    # Disambiguate 3-level (HC) vs 4-level (PD) layout.
    if len(rel) >= 4 and _RANGE_PATTERN.match(rel[1]):
        speaker_name = rel[2]
    else:
        speaker_name = rel[1]

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
