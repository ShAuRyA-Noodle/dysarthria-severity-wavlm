"""UASpeech loader (English dysarthria; graded severity via intelligibility).

Reuses the existing ``metadata_uaspeech.csv`` (28 speakers: 15 dysarthric +
13 control; per-speaker intelligibility %). Native severity rank is derived
from intelligibility (higher intelligibility -> lower severity), so
``severity_rank = 100 - intelligibility`` (control ~= 0). ``severity_proxy``
(0..3 bands) is kept as ``native_label``.

STATUS ON THIS MACHINE: the metadata is present but the UASpeech audio is NOT
on disk (neither the processed ``data/processed/audio_16khz/uaspeech`` tree nor
the raw ``UASpeech_original_C`` tree). Frozen WavLM features cannot be extracted
without audio, so this corpus SKIPS gracefully and reports exactly what must
appear to activate it. No rows are fabricated.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.veritas.corpora.base import (
    ETIOLOGY_CONTROL,
    ETIOLOGY_DYSARTHRIA,
    PROJECT_ROOT,
    CorpusStatus,
    UtteranceRecord,
    remap_stale_path,
)

NAME = "uaspeech"
LANGUAGE = "en"
ETIOLOGY = ETIOLOGY_DYSARTHRIA

_META = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_uaspeech.csv"


def _frame() -> pd.DataFrame | None:
    if not _META.exists():
        return None
    df = pd.read_csv(_META)
    # Prefer processed path; fall back to raw. Remap stale machine prefixes.
    proc = df["filepath_processed"].map(remap_stale_path)
    raw = df.get("filepath_raw", pd.Series([""] * len(df))).fillna("").map(remap_stale_path)
    df["_proc"] = proc
    df["_raw"] = raw
    df["_exists"] = [
        Path(p).exists() or (bool(r) and Path(r).exists())
        for p, r in zip(proc, raw)
    ]
    df["wav_path"] = [
        p if Path(p).exists() else (r if (bool(r) and Path(r).exists()) else p)
        for p, r in zip(proc, raw)
    ]
    return df


def status() -> CorpusStatus:
    if not _META.exists():
        return CorpusStatus(
            NAME, False, f"UASpeech metadata missing at {_META}",
            activation="build metadata_uaspeech.csv via src.uaspeech_parser",
            label_kind="graded_severity",
        )
    df = _frame()
    n_ok = int(df["_exists"].sum())
    n_spk = int(df["speaker_id"].nunique())
    if n_ok == 0:
        return CorpusStatus(
            NAME, False,
            "metadata present (28 speakers) but NO audio on disk",
            n_speakers_hint=n_spk,
            activation=(
                "restore UASpeech wavs so paths in metadata_uaspeech.csv resolve, e.g. "
                r"data\processed\audio_16khz\uaspeech\<spk>\<utt>.wav (or the raw "
                r"UASpeech_original_C\...\audio\original\<spk>\<file>.wav tree)"
            ),
            label_kind="graded_severity",
        )
    return CorpusStatus(
        NAME, True,
        f"graded severity via intelligibility; {n_ok} utts resolve",
        n_speakers_hint=n_spk,
        label_kind="graded_severity",
    )


def load() -> list[UtteranceRecord]:
    st = status()
    if not st.available:
        return []
    df = _frame()
    df = df[df["_exists"]].copy()
    records: list[UtteranceRecord] = []
    for r in df.itertuples(index=False):
        intel = float(r.intelligibility_score)
        is_ctrl = bool(r.is_control)
        records.append(
            UtteranceRecord(
                speaker_id=str(r.speaker_id),
                corpus=NAME,
                etiology=ETIOLOGY_CONTROL if is_ctrl else ETIOLOGY_DYSARTHRIA,
                language=LANGUAGE,
                severity_rank=100.0 - intel,   # higher = more severe
                wav_path=str(r.wav_path),
                utterance_id=str(r.utterance_id),
                task=str(getattr(r, "word", "") or ""),
                is_control=is_ctrl,
                native_label=float(getattr(r, "severity_proxy", 0.0)),
            )
        )
    return records
