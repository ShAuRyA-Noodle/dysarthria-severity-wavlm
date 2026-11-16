"""Corpus-agnostic record + loader contract for the VERITAS severity harness.

Every speech corpus (TORGO, UASpeech, IPVS, EasyCall, MDVR-KCL, VOC-ALS, SVD, …)
is exposed through ONE small interface so the cross-corpus experiment never has
to special-case a dataset:

    loader.status() -> CorpusStatus     # is the data on disk? if not, what's needed?
    loader.load()   -> list[UtteranceRecord]

A loader is a plain module (see ``torgo.py``, ``ipvs.py``, …) that defines the
module-level constants ``NAME``, ``LANGUAGE``, ``ETIOLOGY`` and the two functions
``status()`` and ``load()``. The registry (``registry.py``) discovers them.

Design commitments (carried from the project's own hard-won findings):
  * REAL data only. A loader that cannot find its audio/labels returns an empty
    record list and a ``CorpusStatus(available=False, ...)`` explaining exactly
    what must appear on disk to activate it. It NEVER fabricates rows.
  * Severity is stored as a per-corpus *native ordinal rank* (higher = more
    severe). Cross-corpus harmonisation (z-rank) happens centrally at eval time,
    never by naively pooling raw heterogeneous labels.
  * Splits are always speaker-disjoint; ``speaker_id`` is the grouping key.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Repo root = .../src/veritas/corpora/base.py -> parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Etiology vocabulary (kept small + explicit).
ETIOLOGY_CONTROL = "control"
ETIOLOGY_DYSARTHRIA = "dysarthria"
ETIOLOGY_PD = "PD"
ETIOLOGY_ALS = "ALS"
ETIOLOGY_VOICE_DISORDER = "voice_disorder"


@dataclass(frozen=True)
class UtteranceRecord:
    """One utterance from one speaker in one corpus.

    ``severity_rank`` is the *native* per-corpus ordinal (higher = more severe);
    for a binary detection corpus it is 0/1. Harmonisation to a comparable
    cross-corpus scale (z-rank) is done centrally, not here.
    """

    speaker_id: str
    corpus: str
    etiology: str          # one of the ETIOLOGY_* constants
    language: str          # ISO-639-1, e.g. "en", "it"
    severity_rank: float   # native ordinal; higher = more severe
    wav_path: str
    utterance_id: str = ""
    task: str = ""
    is_control: bool = False
    native_label: float | None = None  # the raw clinical value if distinct from rank


@dataclass(frozen=True)
class CorpusStatus:
    """Whether a corpus is loadable right now, and if not, how to activate it."""

    name: str
    available: bool
    reason: str
    n_speakers_hint: int = 0
    activation: str = ""          # human-readable "what must appear on disk"
    label_kind: str = ""          # "graded_severity" | "binary_detection" | "none"
    is_severity_target: bool = True   # SVD (controls-only OOD arm) sets this False
    extra: dict = field(default_factory=dict)


def remap_stale_path(path: str) -> str:
    """Rewrite metadata paths captured on a different machine to this repo.

    Several metadata CSVs store absolute paths from an old checkout
    (``C:\\Users\\Dell\\Desktop\\Projects\\Dysar\\...``). Rewrite the known
    prefixes to the current repo root. Returns the path unchanged if no known
    stale prefix matches.
    """
    root = str(PROJECT_ROOT)
    stale_prefixes = [
        r"C:\Users\Dell\Desktop\Projects\Dysar",
        r"C:\Users\Dell\Desktop\Dysar_001",
    ]
    for pref in stale_prefixes:
        if path.startswith(pref):
            return root + path[len(pref):]
    return path
