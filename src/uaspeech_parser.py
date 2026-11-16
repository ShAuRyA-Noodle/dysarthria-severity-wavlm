"""UA-Speech dataset parser — filename parsing, metadata building, speaker info.

Mirrors the TORGO pipeline in data_pipeline.py but for the UA-Speech corpus.
"""

from __future__ import annotations

import re
import wave
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Speaker intelligibility scores (ground truth from original paper)
# ---------------------------------------------------------------------------

UA_INTELLIGIBILITY: dict[str, float] = {
    # Dysarthric speakers — intelligibility % from Kim et al., Interspeech 2008
    "F02": 29.0,  "F03": 6.0,   "F04": 2.0,   "F05": 58.0,
    "M01": 24.0,  "M04": 43.0,  "M05": 58.0,  "M07": 2.6,
    "M08": 93.0,  "M09": 6.0,   "M10": 13.0,  "M11": 86.7,
    "M12": 93.0,  "M14": 28.9,  "M16": 19.8,  "M17": 7.0,
    # Control speakers
    "CF02": 100.0, "CF03": 100.0, "CF04": 100.0, "CF05": 100.0,
    "CM01": 100.0, "CM04": 100.0, "CM05": 100.0, "CM06": 100.0,
    "CM08": 100.0, "CM09": 100.0, "CM10": 100.0, "CM12": 100.0,
    "CM13": 100.0,
}


def _intelligibility_to_severity(score: float) -> int:
    """Map intelligibility percentage to severity proxy.

    Very Low (<25%)  → 3
    Low (25-50%)     → 2
    Mid (50-75%)     → 1
    High (>75%)      → 0  (near-normal)
    Control (100%)   → 0
    """
    if score >= 75.0:
        return 0
    elif score >= 50.0:
        return 1
    elif score >= 25.0:
        return 2
    else:
        return 3


# Pre-computed severity proxy for every speaker
UA_SEVERITY: dict[str, int] = {
    spk: _intelligibility_to_severity(score)
    for spk, score in UA_INTELLIGIBILITY.items()
}


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

# Pattern: {speaker}_{block}_{word}_{mic}.wav
# Examples: M16_B1_C1_M3.wav, CF02_B2_UW45_M5.wav
_UA_FILENAME_RE = re.compile(
    r"^(?P<speaker>[A-Z]+\d+)_(?P<block>B\d+)_(?P<word>[A-Za-z0-9]+)_(?P<mic>M\d+)$"
)


def parse_uaspeech_filename(filepath: str) -> dict:
    """Parse a UA-Speech WAV filepath into its structural components.

    Parameters
    ----------
    filepath:
        Path to a UA-Speech WAV file (absolute or relative).

    Returns
    -------
    dict
        Keys: ``speaker_id``, ``word``, ``block``, ``mic``, ``original_path``.
    """
    path = Path(filepath)
    stem = path.stem  # e.g. "M16_B1_C1_M3"
    m = _UA_FILENAME_RE.match(stem)

    if m is None:
        return {
            "speaker_id": "unknown",
            "word": "unknown",
            "block": "unknown",
            "mic": "unknown",
            "original_path": str(filepath),
        }

    return {
        "speaker_id": m.group("speaker"),
        "word": m.group("word"),
        "block": m.group("block"),
        "mic": m.group("mic"),
        "original_path": str(filepath),
    }


# ---------------------------------------------------------------------------
# Audio info (fast header-only read)
# ---------------------------------------------------------------------------

def _get_audio_info(filepath: str) -> dict:
    """Return audio metadata without loading the full waveform."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(filepath)
    with wave.open(str(path), "r") as wf:
        sr = wf.getframerate()
        n_samples = wf.getnframes()
        n_channels = wf.getnchannels()
    return {
        "duration_sec": n_samples / sr if sr > 0 else 0.0,
        "sample_rate": sr,
        "n_channels": n_channels,
        "n_samples": n_samples,
    }


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def build_uaspeech_metadata(
    root_dirs: list[str],
    output_csv: str,
    mic_filter: str = "M3",
    processed_audio_root: Optional[str] = None,
) -> pd.DataFrame:
    """Crawl UA-Speech audio directories and build a metadata DataFrame.

    Parameters
    ----------
    root_dirs:
        List of root directories to crawl for WAV files.  Typically
        the C-channel and FM-channel ``audio/original`` directories.
    output_csv:
        Where to save the resulting CSV.
    mic_filter:
        Only include files from this microphone (default ``"M3"``).
        Set to ``None`` to include all microphones.
    processed_audio_root:
        Root directory for processed audio files.  If None, defaults to
        ``data/processed/audio_16khz/uaspeech`` relative to project root.

    Returns
    -------
    pd.DataFrame
        One row per utterance.
    """
    if processed_audio_root is None:
        # Infer project root from this file: src/uaspeech_parser.py → project root
        project_root = Path(__file__).resolve().parent.parent
        proc_root = project_root / "data" / "processed" / "audio_16khz" / "uaspeech"
    else:
        proc_root = Path(processed_audio_root)

    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    seen: set[str] = set()  # de-duplicate across root_dirs

    for root_dir in root_dirs:
        root = Path(root_dir)
        if not root.exists():
            print(f"[warn] Directory not found, skipping: {root}")
            continue

        # Crawl for WAV files — they may be in speaker sub-dirs or flat
        wav_files = sorted(root.rglob("*.wav"))
        print(f"[crawl] Found {len(wav_files)} WAV files in {root}")

        for wav_path in wav_files:
            parsed = parse_uaspeech_filename(str(wav_path))
            spk_id = parsed["speaker_id"]
            mic = parsed["mic"]

            if spk_id == "unknown":
                continue

            # Microphone filter
            if mic_filter is not None and mic != mic_filter:
                continue

            # De-duplicate (same file may appear in overlapping root_dirs)
            dedup_key = f"{spk_id}_{parsed['block']}_{parsed['word']}_{mic}"
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            # Audio info
            try:
                info = _get_audio_info(str(wav_path))
                duration = info["duration_sec"]
                sr_orig = info["sample_rate"]
            except Exception:
                duration = 0.0
                sr_orig = -1

            # Intelligibility and severity
            intell = UA_INTELLIGIBILITY.get(spk_id, -1.0)
            severity = UA_SEVERITY.get(spk_id, -1)
            is_control = spk_id.startswith("C")

            # Utterance ID
            utterance_id = f"UA_{spk_id}_{parsed['block']}_{parsed['word']}_{mic}"

            # Pre-computed processed path
            processed_path = proc_root / spk_id / f"{utterance_id}.wav"

            rows.append({
                "utterance_id": utterance_id,
                "dataset": "uaspeech",
                "speaker_id": spk_id,
                "word": parsed["word"],
                "block": parsed["block"],
                "mic": mic,
                "intelligibility_score": intell,
                "severity_proxy": severity,
                "filepath_raw": str(wav_path),
                "filepath_processed": str(processed_path),
                "duration_sec": duration,
                "sample_rate_orig": sr_orig,
                "is_control": is_control,
            })

    df = pd.DataFrame(rows)

    if len(df) > 0:
        # Sort for reproducibility
        df = df.sort_values(["speaker_id", "block", "word"]).reset_index(drop=True)

    df.to_csv(str(out_csv), index=False)
    print(f"[build_uaspeech_metadata] Saved {len(df)} rows → {out_csv}")

    return df


# ---------------------------------------------------------------------------
# Speaker info helper
# ---------------------------------------------------------------------------

def get_uaspeech_speaker_info() -> pd.DataFrame:
    """Return a DataFrame with all UA-Speech speakers and their metadata.

    Returns
    -------
    pd.DataFrame
        Columns: ``speaker_id``, ``intelligibility_score``, ``severity_proxy``,
        ``is_control``, ``severity_label``.
    """
    severity_labels = {0: "High/Control", 1: "Mid", 2: "Low", 3: "Very Low"}

    rows = []
    for spk_id, score in UA_INTELLIGIBILITY.items():
        sev = UA_SEVERITY[spk_id]
        rows.append({
            "speaker_id": spk_id,
            "intelligibility_score": score,
            "severity_proxy": sev,
            "is_control": spk_id.startswith("C"),
            "severity_label": severity_labels.get(sev, "Unknown"),
        })

    df = pd.DataFrame(rows).sort_values("speaker_id").reset_index(drop=True)
    return df
