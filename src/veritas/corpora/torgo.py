"""TORGO loader (English dysarthria; graded severity 0/1/2).

Reuses the project's authoritative artefacts — nothing is rebuilt:
  * speaker severity from ``src.constants.TORGO_SEVERITY`` (0=ctrl, 1=mild-mod,
    2=severe);
  * utterance list from ``data/processed/metadata/metadata_unified.csv``
    (the only dataset in that file is TORGO);
  * processed 16 kHz audio at ``data/processed/audio_16khz/<spk>/<utt>.wav``.

The frozen WavLM-L9 mean-pool features for TORGO already exist as a validated
cache (``data/processed/features/torgo_utt_mean_l9.npy``); the feature builder
reuses it directly, so this loader only needs to enumerate speakers/utterances.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.constants import TORGO_SEVERITY
from src.veritas.corpora.base import (
    ETIOLOGY_CONTROL,
    ETIOLOGY_DYSARTHRIA,
    PROJECT_ROOT,
    CorpusStatus,
    UtteranceRecord,
)

NAME = "torgo"
LANGUAGE = "en"
ETIOLOGY = ETIOLOGY_DYSARTHRIA

_META = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
_AUDIO_ROOT = PROJECT_ROOT / "data" / "processed" / "audio_16khz"
_FEATURE_CACHE = PROJECT_ROOT / "data" / "processed" / "features" / "torgo_utt_mean_l9.npy"


def status() -> CorpusStatus:
    if not _META.exists():
        return CorpusStatus(
            NAME, False, f"unified metadata missing at {_META}",
            activation="build data/processed/metadata/metadata_unified.csv",
            label_kind="graded_severity",
        )
    have_audio = _AUDIO_ROOT.exists()
    have_feats = _FEATURE_CACHE.exists()
    if not (have_audio or have_feats):
        return CorpusStatus(
            NAME, False,
            "no TORGO audio and no cached WavLM features on disk",
            activation=f"provide {_AUDIO_ROOT}\\<spk>\\<utt>.wav or the L9 feature cache",
            label_kind="graded_severity",
        )
    return CorpusStatus(
        NAME, True,
        "graded severity 0/1/2; reuses metadata_unified.csv + cached L9 features",
        n_speakers_hint=len(TORGO_SEVERITY),
        label_kind="graded_severity",
    )


def load() -> list[UtteranceRecord]:
    st = status()
    if not st.available:
        return []
    df = pd.read_csv(_META)
    df = df[df["dataset"] == "torgo"].copy()
    records: list[UtteranceRecord] = []
    for r in df.itertuples(index=False):
        spk = str(r.speaker_id)
        if spk not in TORGO_SEVERITY:
            continue  # never invent a label
        sev = TORGO_SEVERITY[spk]
        wav = _AUDIO_ROOT / spk / f"{r.utterance_id}.wav"
        records.append(
            UtteranceRecord(
                speaker_id=spk,
                corpus=NAME,
                etiology=ETIOLOGY_CONTROL if sev == 0 else ETIOLOGY_DYSARTHRIA,
                language=LANGUAGE,
                severity_rank=float(sev),
                wav_path=str(wav),
                utterance_id=str(r.utterance_id),
                task=str(getattr(r, "word", "") or ""),
                is_control=bool(sev == 0),
                native_label=float(sev),
            )
        )
    return records
