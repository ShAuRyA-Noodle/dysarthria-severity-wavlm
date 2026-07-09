"""Helpers for the not-yet-ready 'downloading' corpora.

These corpora land under ``data/veritas/downloads/`` as archives that are then
extracted into a sibling directory. A stub loader activates only when its
extracted, loader-ready tree (audio + resolvable labels) is present; otherwise
it SKIPS and reports precisely what must appear. Nothing is ever fabricated.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

from src.veritas.corpora.base import PROJECT_ROOT

DOWNLOADS = PROJECT_ROOT / "data" / "veritas" / "downloads"

_AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg")


def archive_state(zip_name: str) -> str:
    """Return 'absent' | 'unreadable' | 'ok' for an archive under downloads/.

    Cheap check: open the archive and read its central directory (namelist).
    A truncated/partial download raises ``BadZipFile`` here. We deliberately do
    NOT full-CRC (``testzip``) every archive — that would re-hash hundreds of MB
    on every harness run.
    """
    p = DOWNLOADS / zip_name
    if not p.exists():
        return "absent"
    try:
        with zipfile.ZipFile(p) as zf:
            return "ok" if zf.namelist() else "unreadable"
    except zipfile.BadZipFile:
        return "unreadable"
    except Exception:  # noqa: BLE001
        return "unreadable"


def find_extracted_audio(*candidate_dirs: Path) -> list[Path]:
    """Return audio files under the first candidate dir that contains any."""
    for d in candidate_dirs:
        if d.exists():
            hits = [
                p for p in d.rglob("*")
                if p.suffix.lower() in _AUDIO_EXT
            ]
            if hits:
                return hits
    return []
