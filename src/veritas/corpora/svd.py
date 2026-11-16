"""SVD loader stub (Saarbrücken Voice Database; voice-disorder SPECIFICITY arm).

SVD is NOT a severity target. It is a non-dysarthria / non-PD / non-ALS voice
pathology set, used as a specificity / out-of-distribution arm: a severity probe
trained on dysarthria/PD should NOT light up strongly on generic voice disorders.
Hence ``is_severity_target=False`` and it is excluded from the severity LOCO by
the registry's default filter.

Archive on this machine (``SVD_data.zip``) is currently unreadable (incomplete
download). Activates when extracted. No fabricated rows.
"""
from __future__ import annotations

from src.veritas.corpora.base import ETIOLOGY_VOICE_DISORDER, CorpusStatus, UtteranceRecord
from src.veritas.corpora.downloads import DOWNLOADS, archive_state, find_extracted_audio

NAME = "svd"
LANGUAGE = "de"  # Saarbrücken Voice Database is German
ETIOLOGY = ETIOLOGY_VOICE_DISORDER

_EXTRACT_DIRS = (DOWNLOADS / "SVD", DOWNLOADS / "SVD_data", DOWNLOADS / "svd")


def status() -> CorpusStatus:
    audio = find_extracted_audio(*_EXTRACT_DIRS)
    arch = archive_state("SVD_data.zip")
    if not audio:
        if arch == "ok":
            reason = "SVD_data.zip is valid but not extracted yet"
        elif arch == "unreadable":
            reason = "SVD_data.zip present but unreadable (download incomplete)"
        else:
            reason = "SVD not downloaded"
        return CorpusStatus(
            NAME, False, reason,
            activation=(
                "finish/repair SVD_data.zip, extract to data/veritas/downloads/SVD/; "
                "used as a specificity/OOD arm (voice-disorder controls), NOT a "
                "severity target."
            ),
            label_kind="none",
            is_severity_target=False,
        )
    return CorpusStatus(
        NAME, False,
        f"{len(audio)} audio files extracted; specificity arm wiring not implemented",
        activation="fill in load() to emit voice-disorder records for the OOD arm",
        label_kind="none",
        is_severity_target=False,
    )


def load() -> list[UtteranceRecord]:
    # Specificity/OOD arm; activates when extracted. Until then, skip.
    return []
