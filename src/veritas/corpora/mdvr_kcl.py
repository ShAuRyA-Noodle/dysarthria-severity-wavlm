"""MDVR-KCL loader stub (Parkinson's; Hoehn & Yahr / UPDRS).

The archive (``MDVR-KCL.zip``) is a VALID zip on this machine and extracts to
``26-29_09_2017_KCL/{ReadText,SpontaneousDialogue}/{HC,PD}/ID##_*.wav`` — but it
ships NO clinical-score file, so only a binary HC/PD split is derivable from the
path. A SEVERITY harness needs the graded H&Y/UPDRS per patient ID (published
with the dataset, not inside the audio zip). This stub therefore activates only
when that score table is provided; until then it SKIPS. No fabricated rows.
"""
from __future__ import annotations

from src.veritas.corpora.base import ETIOLOGY_PD, CorpusStatus, UtteranceRecord
from src.veritas.corpora.downloads import DOWNLOADS, archive_state, find_extracted_audio

NAME = "mdvr_kcl"
LANGUAGE = "en"
ETIOLOGY = ETIOLOGY_PD

_EXTRACT_DIRS = (
    DOWNLOADS / "26-29_09_2017_KCL",
    DOWNLOADS / "MDVR-KCL",
    DOWNLOADS / "MDVR_KCL",
)


def status() -> CorpusStatus:
    audio = find_extracted_audio(*_EXTRACT_DIRS)
    arch = archive_state("MDVR-KCL.zip")
    if not audio:
        if arch == "ok":
            reason = "MDVR-KCL.zip is valid but not extracted yet"
        elif arch == "unreadable":
            reason = "MDVR-KCL.zip present but unreadable (download incomplete)"
        else:
            reason = "MDVR-KCL not downloaded"
        return CorpusStatus(
            NAME, False, reason,
            activation=(
                "extract MDVR-KCL.zip to data/veritas/downloads/, then provide the "
                "H&Y/UPDRS score per patient ID (ID## in filenames). Audio alone gives "
                "only a binary HC/PD detection split, not graded severity."
            ),
            label_kind="graded_severity",
        )
    return CorpusStatus(
        NAME, False,
        f"{len(audio)} wavs extracted (HC/PD in path) but no H&Y/UPDRS score table",
        activation=(
            "provide graded H&Y or UPDRS-III per patient ID and fill in load(); "
            "without it MDVR-KCL is PD-vs-HC detection only, not a severity target."
        ),
        label_kind="graded_severity",
    )


def load() -> list[UtteranceRecord]:
    # Activates only when graded H&Y/UPDRS labels are provided; until then, skip.
    return []
