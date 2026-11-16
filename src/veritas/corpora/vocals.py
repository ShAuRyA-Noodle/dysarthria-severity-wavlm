"""VOC-ALS loader stub (ALS; ALSFRS-R severity).

Not downloaded on this machine yet. Activates when the extracted corpus plus a
per-speaker ALSFRS-R (or bulbar-subscore) table are present. No fabricated rows.

Language is provisional ("it"; VOC-ALS is an Italian ALS voice corpus) and must
be confirmed against the release when it lands.
"""
from __future__ import annotations

from src.veritas.corpora.base import ETIOLOGY_ALS, CorpusStatus, UtteranceRecord
from src.veritas.corpora.downloads import DOWNLOADS, archive_state, find_extracted_audio

NAME = "vocals"
LANGUAGE = "it"  # provisional; confirm on release
ETIOLOGY = ETIOLOGY_ALS

_EXTRACT_DIRS = (DOWNLOADS / "VOC-ALS", DOWNLOADS / "VOC_ALS", DOWNLOADS / "vocals")


def status() -> CorpusStatus:
    audio = find_extracted_audio(*_EXTRACT_DIRS)
    arch = archive_state("VOC-ALS.zip")
    if not audio:
        if arch == "ok":
            reason = "VOC-ALS.zip is valid but not extracted yet"
        elif arch == "unreadable":
            reason = "VOC-ALS.zip present but unreadable (download incomplete)"
        else:
            reason = "VOC-ALS not downloaded"
        return CorpusStatus(
            NAME, False, reason,
            activation=(
                "download + extract VOC-ALS to data/veritas/downloads/VOC-ALS/, then "
                "provide per-speaker ALSFRS-R (ideally the speech/bulbar subscore); "
                "confirm corpus language; then fill in load()."
            ),
            label_kind="graded_severity",
        )
    return CorpusStatus(
        NAME, False,
        f"{len(audio)} audio files extracted but no ALSFRS-R table wired yet",
        activation="add per-speaker ALSFRS-R map and fill in load()",
        label_kind="graded_severity",
    )


def load() -> list[UtteranceRecord]:
    # Activates only when audio + ALSFRS-R labels are both resolvable; until then, skip.
    return []
