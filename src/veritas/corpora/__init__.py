"""VERITAS cross-corpus severity harness — corpus loaders.

Importing this package activates the VERITAS offline bootstrap (via ``src.veritas``)
so any downstream WavLM load uses the local Hugging Face cache only.

Public API::

    from src.veritas.corpora import registry
    registry.all_status()        # what's on disk / what each corpus needs
    registry.load_available()    # {corpus: [UtteranceRecord, ...]} — real rows only
"""
from __future__ import annotations

import src.veritas as _veritas  # noqa: F401  (side effect: enable offline HF cache)

from src.veritas.corpora.base import (  # noqa: F401
    CorpusStatus,
    UtteranceRecord,
)

__all__ = ["CorpusStatus", "UtteranceRecord"]
