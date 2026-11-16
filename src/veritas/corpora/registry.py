"""Registry of VERITAS corpus loaders.

Yields every registered corpus (available or not) so the cross-corpus
experiment can (a) run on whatever is on disk right now and (b) print exactly
what each not-yet-ready corpus needs to activate — without editing the
experiment when a new corpus lands.

Each entry is a loader module exposing ``NAME``, ``LANGUAGE``, ``ETIOLOGY``,
``status()`` and ``load()`` (see ``base.py``).
"""
from __future__ import annotations

from types import ModuleType

from src.veritas.corpora import (
    easycall,
    ipvs,
    mdvr_kcl,
    svd,
    torgo,
    uaspeech,
    vocals,
)
from src.veritas.corpora.base import CorpusStatus, UtteranceRecord

# Order: on-disk-now corpora first, then the downloading/pluggable stubs.
LOADERS: list[ModuleType] = [torgo, uaspeech, ipvs, easycall, mdvr_kcl, vocals, svd]


def all_status() -> list[CorpusStatus]:
    """Status for every registered corpus (available or skipped)."""
    return [m.status() for m in LOADERS]


def available_loaders(severity_targets_only: bool = True) -> list[ModuleType]:
    """Loader modules whose data is on disk and loadable right now.

    ``severity_targets_only`` drops specificity/OOD arms (e.g. SVD) that are not
    severity-prediction targets.
    """
    out = []
    for m in LOADERS:
        st = m.status()
        if not st.available:
            continue
        if severity_targets_only and not st.is_severity_target:
            continue
        out.append(m)
    return out


def load_available(
    severity_targets_only: bool = True,
) -> dict[str, list[UtteranceRecord]]:
    """Map corpus name -> records for every available corpus (real rows only)."""
    result: dict[str, list[UtteranceRecord]] = {}
    for m in available_loaders(severity_targets_only=severity_targets_only):
        recs = m.load()
        if recs:  # never register an empty corpus
            result[m.NAME] = recs
    return result
