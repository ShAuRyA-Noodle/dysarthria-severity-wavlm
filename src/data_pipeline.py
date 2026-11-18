"""Data pipeline for loading, validating, and preprocessing TORGO audio data."""

from __future__ import annotations

import itertools
import json
import wave
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

# ---------------------------------------------------------------------------
# Speaker metadata (authoritative, from the TORGO documentation)
# ---------------------------------------------------------------------------

TORGO_SEVERITY: dict[str, int] = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}

TORGO_GENDER: dict[str, str] = {
    "F01": "F", "F03": "F", "F04": "F",
    "M01": "M", "M02": "M", "M03": "M", "M04": "M", "M05": "M",
    "FC01": "F", "FC02": "F", "FC03": "F",
    "MC01": "M", "MC02": "M", "MC03": "M", "MC04": "M",
}

# Ordered group prefixes used when crawling raw TORGO directories
_TORGO_GROUPS = ("F", "FC", "M", "MC")

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def verify_audio_file(filepath: str) -> tuple[bool, str]:
    """Check if a WAV file is valid and readable.

    Opens the file with the standard library ``wave`` module (zero-copy header
    check) and, if that succeeds, with ``soundfile`` to verify the data block
    is intact.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the WAV file.

    Returns
    -------
    tuple[bool, str]
        ``(True, "")`` when the file is valid, or ``(False, error_message)``
        describing the first failure encountered.
    """
    path = Path(filepath)
    if not path.exists():
        return False, f"File not found: {filepath}"
    if path.stat().st_size == 0:
        return False, "File is empty (0 bytes)"

    # Header check
    try:
        with wave.open(str(path), "r") as wf:
            if wf.getnframes() == 0:
                return False, "WAV reports 0 frames"
    except wave.Error as exc:
        return False, f"wave header error: {exc}"
    except Exception as exc:
        return False, f"Unexpected error reading WAV header: {exc}"

    # Data-block check via soundfile
    try:
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if len(data) == 0:
            return False, "soundfile read returned 0 samples"
    except Exception as exc:
        return False, f"soundfile read error: {exc}"

    return True, ""


def get_audio_info(filepath: str) -> dict:
    """Return audio metadata without loading the full waveform.

    Uses the ``wave`` standard-library module for a fast header-only read.

    Parameters
    ----------
    filepath:
        Path to the WAV file.

    Returns
    -------
    dict
        Keys: ``duration_sec`` (float), ``sample_rate`` (int),
        ``n_channels`` (int), ``n_samples`` (int).

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    wave.Error
        If the file cannot be parsed as a WAV.
    """
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


def resample_and_normalize(
    input_path: str,
    output_path: str,
    target_sr: int = 16000,
    target_db: float = -20.0,
) -> bool:
    """Resample a WAV file to *target_sr* Hz and normalise amplitude.

    Normalisation targets *target_db* dBFS (decibels relative to full scale).
    The output is written as a 16-bit mono WAV using ``soundfile``.

    The function is **idempotent**: if *output_path* already exists it is
    skipped immediately (supports resumable preprocessing runs).

    Parameters
    ----------
    input_path:
        Source WAV file (any sample rate, mono or stereo).
    output_path:
        Destination WAV file path.  Parent directories are created if needed.
    target_sr:
        Target sample rate in Hz.  Default ``16000``.
    target_db:
        Target loudness in dBFS.  Default ``-20.0``.

    Returns
    -------
    bool
        ``True`` on success (including skip), ``False`` if an error occurred.
    """
    out_path = Path(output_path)

    # Resumability: skip if already done
    if out_path.exists() and out_path.stat().st_size > 0:
        return True

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Load & resample (librosa converts stereo→mono automatically)
        y, _ = librosa.load(input_path, sr=target_sr, mono=True)

        if len(y) == 0:
            return False

        # Amplitude normalisation to target_db dBFS
        rms = np.sqrt(np.mean(y ** 2))
        if rms > 0:
            target_rms = 10 ** (target_db / 20.0)
            y = y * (target_rms / rms)
        # Clip to [-1, 1] to avoid clipping artefacts in int16 output
        y = np.clip(y, -1.0, 1.0)

        sf.write(str(out_path), y, target_sr, subtype="PCM_16")
        return True

    except Exception:
        # Remove partial output if it was created
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        return False


def parse_torgo_filename(filepath: str) -> dict:
    """Parse a TORGO WAV filepath into its structural components.

    TORGO file paths follow the pattern::

        {root}/{Group}/{SpeakerID}/{Session}/wav_{micType}/{NNNN}.wav

    Examples::

        data/raw/torgo/F/F01/Session1/wav_headMic/0001.wav
        data/raw/torgo/MC/MC03/Session2/wav_arrayMic/0042.wav

    Parameters
    ----------
    filepath:
        Full path (absolute or relative) to the WAV file.

    Returns
    -------
    dict
        Keys: ``speaker_id`` (str), ``session`` (str), ``mic_type`` (str),
        ``prompt_id`` (str), ``original_path`` (str).
        ``mic_type`` is one of ``"headMic"`` or ``"arrayMic"``.
        ``prompt_id`` is the zero-padded stem (e.g. ``"0001"``).
    """
    path = Path(filepath)
    parts = path.parts

    # Locate the mic-type directory (wav_headMic or wav_arrayMic)
    mic_type_raw: Optional[str] = None
    mic_idx: Optional[int] = None
    for i, part in enumerate(parts):
        if part.startswith("wav_"):
            mic_type_raw = part          # e.g. "wav_headMic"
            mic_idx = i
            break

    if mic_idx is None:
        # Fallback: best-effort extraction
        return {
            "speaker_id": "unknown",
            "session": "unknown",
            "mic_type": "unknown",
            "prompt_id": path.stem,
            "original_path": str(filepath),
        }

    # Parts relative to the mic directory
    # Expected layout: .../{Group}/{SpeakerID}/{Session}/wav_{micType}/{stem}.wav
    session = parts[mic_idx - 1] if mic_idx >= 1 else "unknown"
    speaker_id = parts[mic_idx - 2] if mic_idx >= 2 else "unknown"
    mic_type = mic_type_raw.replace("wav_", "")   # "headMic" or "arrayMic"
    prompt_id = path.stem                          # e.g. "0001"

    return {
        "speaker_id": speaker_id,
        "session": session,
        "mic_type": mic_type,
        "prompt_id": prompt_id,
        "original_path": str(filepath),
    }


# ---------------------------------------------------------------------------
# Metadata builder
# ---------------------------------------------------------------------------

def _read_prompt_text(prompt_dir: Path, file_stem: str) -> str:
    """Return prompt text for *file_stem* from *prompt_dir*, or the stem itself."""
    txt = prompt_dir / f"{file_stem}.txt"
    if txt.exists():
        return txt.read_text(encoding="utf-8", errors="replace").strip()
    return file_stem


def build_torgo_metadata(
    raw_dir: str,
    mic_type: str = "headMic",
    severity_map: dict = TORGO_SEVERITY,
    gender_map: dict = TORGO_GENDER,
) -> pd.DataFrame:
    """Crawl the TORGO raw directory and build a complete metadata DataFrame.

    Only utterances recorded with *mic_type* are included (default ``headMic``).

    The processed audio path (``filepath_processed``) is pre-computed but the
    files are **not** created here — that is the job of
    ``experiments/03_preprocess_audio.py``.

    Parameters
    ----------
    raw_dir:
        Root of the raw TORGO data (contains sub-folders ``F``, ``FC``,
        ``M``, ``MC``).
    mic_type:
        Which microphone condition to include.  ``"headMic"`` (default) gives
        higher-quality recordings.
    severity_map:
        Mapping of speaker ID → severity score (0 = control, 1 = mild/mod,
        2 = severe).
    gender_map:
        Mapping of speaker ID → gender string (``"M"`` or ``"F"``).

    Returns
    -------
    pd.DataFrame
        One row per utterance with columns:
        ``utterance_id``, ``dataset``, ``speaker_id``, ``prompt_id``,
        ``word``, ``filepath_raw``, ``filepath_processed``,
        ``duration_sec``, ``sample_rate_orig``, ``severity_score``,
        ``gender``, ``is_control``, ``session``, ``mic_type``.

    Side Effects
    ------------
    Saves the DataFrame to
    ``data/processed/metadata/torgo_metadata_raw.csv`` relative to the
    project root (two levels above *raw_dir*).
    """
    raw_root = Path(raw_dir)
    project_root = raw_root.parent.parent.parent  # data/raw/torgo → data/raw → data → project root
    processed_audio_root = project_root / "data" / "processed" / "audio_16khz"
    meta_dir = project_root / "data" / "processed" / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)

    mic_dir_name = f"wav_{mic_type}"
    rows: list[dict] = []

    # Counter per (speaker, prompt_id) pair to generate unique utterance IDs
    uid_counter: dict[tuple, int] = {}

    for group in _TORGO_GROUPS:
        group_dir = raw_root / group
        if not group_dir.is_dir():
            continue

        for spk_dir in sorted(group_dir.iterdir()):
            if not spk_dir.is_dir():
                continue
            spk_id = spk_dir.name
            if spk_id not in severity_map:
                continue  # skip Notes or unrecognised folders

            severity = severity_map[spk_id]
            gender   = gender_map.get(spk_id, "U")
            is_ctrl  = severity == 0

            for session_dir in sorted(spk_dir.iterdir()):
                if not session_dir.is_dir() or session_dir.name == "Notes":
                    continue
                session = session_dir.name
                wav_dir = session_dir / mic_dir_name
                prompt_dir = session_dir / "prompts"

                if not wav_dir.is_dir():
                    continue

                for wav_path in sorted(wav_dir.glob("*.wav")):
                    stem = wav_path.stem
                    prompt_text = _read_prompt_text(prompt_dir, stem)

                    # Audio info
                    try:
                        info = get_audio_info(str(wav_path))
                        duration = info["duration_sec"]
                        sr_orig  = info["sample_rate"]
                    except Exception:
                        duration = 0.0
                        sr_orig  = -1

                    # Unique utterance ID
                    key = (spk_id, stem, session)
                    uid_counter[key] = uid_counter.get(key, 0) + 1
                    idx = uid_counter[key]
                    utterance_id = f"TORGO_{spk_id}_{stem}_{session}_{idx:03d}"

                    # Pre-computed processed path
                    processed_path = (
                        processed_audio_root / spk_id / f"{utterance_id}.wav"
                    )

                    rows.append({
                        "utterance_id":      utterance_id,
                        "dataset":           "torgo",
                        "speaker_id":        spk_id,
                        "prompt_id":         stem,
                        "word":              prompt_text,
                        "filepath_raw":      str(wav_path),
                        "filepath_processed": str(processed_path),
                        "duration_sec":      duration,
                        "sample_rate_orig":  sr_orig,
                        "severity_score":    severity,
                        "gender":            gender,
                        "is_control":        is_ctrl,
                        "session":           session,
                        "mic_type":          mic_type,
                    })

    df = pd.DataFrame(rows)

    out_csv = meta_dir / "torgo_metadata_raw.csv"
    df.to_csv(str(out_csv), index=False)
    print(f"[build_torgo_metadata] Saved {len(df)} rows → {out_csv}")

    return df


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------

def create_data_splits(
    metadata_df: pd.DataFrame,
    output_path: str = "data/splits/splits.json",
    random_seed: int = 42,
) -> dict:
    """Create three evaluation splits for the TORGO dataset.

    All splits are defined using speaker IDs only, making them independent of
    utterance count.  The resulting JSON is the single source of truth for
    all experiments — **never regenerate after first creation**.

    Split A — Standard LOSO (primary metric)
        Leave-2-dysarthric-speakers-out cross-validation.
        Every pair of dysarthric speakers is held out exactly once.
        Train set = remaining dysarthric speakers + all controls.
        Produces C(n_dysarthric, 2) folds.

    Split B — Severity generalisation
        Train: control + mild-moderate speakers.
        Test: severe speakers only.
        Answers: *Can a model trained on mild-moderate speech generalise to
        severe dysarthria?*

    Split C — Cross-session
        Train: Session 1 utterances for every speaker.
        Test: all subsequent sessions (Session 2+).
        Speakers with only one session appear only in the train set.
        Answers: *Are features session-independent?*

    Parameters
    ----------
    metadata_df:
        The complete metadata DataFrame (as returned by
        :func:`build_torgo_metadata`).
    output_path:
        Where to save the JSON file.  Parent directories are created if needed.
    random_seed:
        Random seed stored in the JSON for reproducibility bookkeeping.

    Returns
    -------
    dict
        Nested dict with keys ``split_A``, ``split_B``, ``split_C``.
        Each contains the split configuration and fold/speaker definitions.

    Side Effects
    ------------
    Writes the splits to *output_path* as UTF-8 JSON.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    severity_map = (
        metadata_df[["speaker_id", "severity_score"]]
        .drop_duplicates()
        .set_index("speaker_id")["severity_score"]
        .to_dict()
    )

    dysarthric_spks = sorted(
        s for s, sev in severity_map.items() if sev > 0
    )
    control_spks = sorted(
        s for s, sev in severity_map.items() if sev == 0
    )
    mild_mod_spks = sorted(s for s, sev in severity_map.items() if sev == 1)
    severe_spks   = sorted(s for s, sev in severity_map.items() if sev == 2)

    # ── Split A: LOSO ───────────────────────────────────────────────────────
    folds_A = []
    for test_pair in itertools.combinations(dysarthric_spks, 2):
        test_pair = list(test_pair)
        train_dys = [s for s in dysarthric_spks if s not in test_pair]
        folds_A.append({
            "test_speakers":  test_pair,
            "train_speakers": train_dys + control_spks,
            "control_speakers": control_spks,
        })

    split_A = {
        "description": (
            "Leave-2-dysarthric-speakers-out cross-validation. "
            f"{len(folds_A)} folds = C({len(dysarthric_spks)},2)."
        ),
        "n_folds": len(folds_A),
        "all_dysarthric_speakers": dysarthric_spks,
        "all_control_speakers": control_spks,
        "folds": folds_A,
    }

    # ── Split B: Severity generalisation ────────────────────────────────────
    split_B = {
        "description": (
            "Severity generalisation: train on control+mild-moderate, "
            "test on severe only."
        ),
        "train_speakers": mild_mod_spks + control_spks,
        "test_speakers":  severe_spks,
        "control_speakers": control_spks,
        "mild_mod_speakers": mild_mod_spks,
        "severe_speakers": severe_spks,
    }

    # ── Split C: Cross-session ───────────────────────────────────────────────
    # Determine which sessions each speaker has
    session_info = (
        metadata_df.groupby("speaker_id")["session"]
        .apply(lambda s: sorted(s.unique().tolist()))
        .to_dict()
    )

    train_utterances_C: list[str] = []
    test_utterances_C: list[str]  = []
    speakers_train_only_C: list[str] = []
    speakers_both_C: list[str] = []

    for spk_id, sessions in session_info.items():
        # Identify the "first" session (alphabetically first = Session1 or earliest)
        first_session = sessions[0]
        other_sessions = sessions[1:]

        spk_rows = metadata_df[metadata_df["speaker_id"] == spk_id]
        train_utts = spk_rows[spk_rows["session"] == first_session]["utterance_id"].tolist()
        train_utterances_C.extend(train_utts)

        if other_sessions:
            test_utts = spk_rows[spk_rows["session"].isin(other_sessions)]["utterance_id"].tolist()
            test_utterances_C.extend(test_utts)
            speakers_both_C.append(spk_id)
        else:
            speakers_train_only_C.append(spk_id)

    split_C = {
        "description": (
            "Cross-session: train on each speaker's first session, "
            "test on all remaining sessions."
        ),
        "session_map": session_info,
        "speakers_with_multiple_sessions": speakers_both_C,
        "speakers_train_only": speakers_train_only_C,
        "train_utterance_ids": train_utterances_C,
        "test_utterance_ids":  test_utterances_C,
        "n_train": len(train_utterances_C),
        "n_test":  len(test_utterances_C),
    }

    # ── Assemble & save ──────────────────────────────────────────────────────
    splits = {
        "random_seed": random_seed,
        "created_from": "torgo_metadata_raw.csv",
        "dysarthric_speakers": dysarthric_spks,
        "control_speakers": control_spks,
        "split_A": split_A,
        "split_B": split_B,
        "split_C": split_C,
    }

    with open(str(out_path), "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=2)
    print(f"[create_data_splits] Saved splits → {out_path}")
    print(f"  Split A: {len(folds_A)} LOSO folds")
    print(f"  Split B: train={len(split_B['train_speakers'])} speakers, "
          f"test={len(split_B['test_speakers'])} speakers")
    print(f"  Split C: {len(train_utterances_C)} train / "
          f"{len(test_utterances_C)} test utterances")

    return splits
