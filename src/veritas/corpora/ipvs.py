"""IPVS loader (Italian Parkinson's Voice and Speech; Dimauro et al. 2019).

Reuses the existing parsed metadata ``metadata_ipvs.csv`` (built by
``src.cross_corpus.ipvs_parser``) and the raw audio under ``data/raw/ipvs``.
Metadata paths were captured on an old checkout, so they are remapped to this
repo root (``base.remap_stale_path``).

HONEST LABEL NOTE — the public IPVS release ships NO graded clinical severity
(we inspected the accompanying spreadsheets: they contain only reading-time /
characters-per-second rate, not UPDRS-III or Hoehn & Yahr). Therefore the
severity signal here is BINARY: 0 = healthy control, 1 = Parkinson's disease.
This is a detection contrast, not a graded-severity regression. It is reported
as such and never dressed up as graded severity.

Age confound (flagged): controls include both young and elderly speakers while
all PD are older; ``elderly_only=True`` restricts controls to the age-matched
elderly group for a fairer within-corpus contrast.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.veritas.corpora.base import (
    ETIOLOGY_CONTROL,
    ETIOLOGY_PD,
    PROJECT_ROOT,
    CorpusStatus,
    UtteranceRecord,
    remap_stale_path,
)

NAME = "ipvs"
LANGUAGE = "it"
ETIOLOGY = ETIOLOGY_PD

_META = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_ipvs.csv"


def _remapped_frame() -> pd.DataFrame | None:
    if not _META.exists():
        return None
    df = pd.read_csv(_META)
    df["wav_path"] = df["filepath_processed"].map(remap_stale_path)
    df["_exists"] = df["wav_path"].map(lambda p: Path(p).exists())
    return df


def status() -> CorpusStatus:
    df = _remapped_frame()
    if df is None:
        return CorpusStatus(
            NAME, False, f"IPVS metadata missing at {_META}",
            activation="run src.cross_corpus.ipvs_parser.build_ipvs_metadata over data/raw/ipvs",
            label_kind="binary_detection",
        )
    n_ok = int(df["_exists"].sum())
    if n_ok == 0:
        return CorpusStatus(
            NAME, False,
            "IPVS metadata present but no audio resolves on disk",
            activation="provide the raw IPVS wavs under data/raw/ipvs/italian_parkinson/",
            label_kind="binary_detection",
        )
    n_spk = int(df.loc[df["_exists"], "speaker_id"].nunique())
    return CorpusStatus(
        NAME, True,
        f"binary PD-vs-HC only (no graded UPDRS/H&Y in the public release); "
        f"{n_ok} utts resolve",
        n_speakers_hint=n_spk,
        label_kind="binary_detection",
        extra={"note": "severity is detection (0=HC,1=PD), not graded"},
    )


def load(elderly_only: bool = False) -> list[UtteranceRecord]:
    df = _remapped_frame()
    if df is None:
        return []
    df = df[df["_exists"]].copy()
    if elderly_only:
        # Age-matched: drop young controls; keep PD + elderly controls.
        df = df[df["group"].isin(["PD", "HC_Old"])].copy()
    records: list[UtteranceRecord] = []
    for r in df.itertuples(index=False):
        is_pd = bool(r.group == "PD")
        records.append(
            UtteranceRecord(
                speaker_id=str(r.speaker_id),
                corpus=NAME,
                etiology=ETIOLOGY_PD if is_pd else ETIOLOGY_CONTROL,
                language=LANGUAGE,
                severity_rank=1.0 if is_pd else 0.0,
                wav_path=str(r.wav_path),
                utterance_id=str(r.utterance_id),
                task=str(getattr(r, "task", "") or ""),
                is_control=not is_pd,
                native_label=1.0 if is_pd else 0.0,
            )
        )
    return records
