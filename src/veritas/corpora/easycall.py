"""EasyCall loader (Italian dysarthric voice commands).

The audio package (``EasyCall.zip``) has landed and is valid. It ships a
dysarthric-vs-healthy speaker split (documented in its ReadMe: ``f*``/``m*`` are
dysarthric, ``fc*``/``mc*`` are healthy controls) but NO graded per-speaker
severity — the TOM (Therapy Outcome Measure) severity lives in the source paper
(Turrisi et al. 2021), not the audio zip. So, exactly like IPVS here, EasyCall
participates as a BINARY detection arm (0 = healthy, 1 = dysarthric), clearly
labelled as such; providing the TOM table would upgrade it to graded severity.

To keep frozen-WavLM cost bounded, a deterministic per-speaker sample of the
utterances is materialised once from the zip into
``data/veritas/features/easycall_sample/`` (reproducible: sorted, capped). No
rows are fabricated — every record maps to a real extracted wav.
"""
from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path

from src.veritas.corpora.base import (
    ETIOLOGY_CONTROL,
    ETIOLOGY_DYSARTHRIA,
    PROJECT_ROOT,
    CorpusStatus,
    UtteranceRecord,
)
from src.veritas.corpora.downloads import DOWNLOADS, archive_state

NAME = "easycall"
LANGUAGE = "it"
ETIOLOGY = ETIOLOGY_DYSARTHRIA

_ZIP = DOWNLOADS / "EasyCall.zip"
_SAMPLE_DIR = PROJECT_ROOT / "data" / "veritas" / "features" / "easycall_sample"
_PER_SPEAKER = 48  # bounded, representative speaker-mean sample


def _is_control(spk: str) -> bool:
    return spk.lower().startswith(("fc", "mc"))


def _materialize_sample(per_speaker: int = _PER_SPEAKER, log=print) -> int:
    """Extract a deterministic capped per-speaker sample from the zip once."""
    if _SAMPLE_DIR.exists() and any(_SAMPLE_DIR.rglob("*.wav")):
        return sum(1 for _ in _SAMPLE_DIR.rglob("*.wav"))
    if archive_state("EasyCall.zip") != "ok":
        return 0
    zf = zipfile.ZipFile(_ZIP)
    by_spk: dict[str, list[str]] = defaultdict(list)
    for n in zf.namelist():
        if not n.lower().endswith(".wav"):
            continue
        parts = n.split("/")  # EasyCall/<spk>/<session>/<file>.wav
        if len(parts) >= 4:
            by_spk[parts[1]].append(n)
    n_out = 0
    for spk, lst in by_spk.items():
        for name in sorted(lst)[:per_speaker]:  # deterministic selection
            out = _SAMPLE_DIR / spk / Path(name).name
            out.parent.mkdir(parents=True, exist_ok=True)
            if not out.exists():
                out.write_bytes(zf.read(name))
            n_out += 1
    log(f"[easycall] materialised {n_out} wavs ({len(by_spk)} speakers) -> {_SAMPLE_DIR}")
    return n_out


def status() -> CorpusStatus:
    have_sample = _SAMPLE_DIR.exists() and any(_SAMPLE_DIR.rglob("*.wav"))
    arch = archive_state("EasyCall.zip")
    if not have_sample and arch != "ok":
        reason = ("EasyCall.zip present but unreadable (download incomplete)"
                  if arch == "unreadable" else "EasyCall not downloaded")
        return CorpusStatus(
            NAME, False, reason,
            activation="finish/repair EasyCall.zip under data/veritas/downloads/",
            label_kind="binary_detection",
        )
    n_spk = 0
    if have_sample:
        n_spk = sum(1 for d in _SAMPLE_DIR.iterdir() if d.is_dir())
    return CorpusStatus(
        NAME, True,
        "binary dysarthric-vs-healthy (graded TOM severity not in the audio package)",
        n_speakers_hint=n_spk or 55,
        label_kind="binary_detection",
        extra={"activation_for_graded": "provide per-speaker TOM severity from Turrisi et al. 2021"},
    )


def load() -> list[UtteranceRecord]:
    if not (_SAMPLE_DIR.exists() and any(_SAMPLE_DIR.rglob("*.wav"))):
        if _materialize_sample() == 0:
            return []
    records: list[UtteranceRecord] = []
    for wav in sorted(_SAMPLE_DIR.rglob("*.wav")):
        spk = wav.parent.name
        ctrl = _is_control(spk)
        records.append(
            UtteranceRecord(
                speaker_id=f"EC_{spk}",
                corpus=NAME,
                etiology=ETIOLOGY_CONTROL if ctrl else ETIOLOGY_DYSARTHRIA,
                language=LANGUAGE,
                severity_rank=0.0 if ctrl else 1.0,
                wav_path=str(wav),
                utterance_id=f"EC_{spk}_{wav.stem}",
                task="voice_command",
                is_control=ctrl,
                native_label=0.0 if ctrl else 1.0,
            )
        )
    return records
