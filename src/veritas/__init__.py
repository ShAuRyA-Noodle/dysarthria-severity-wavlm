"""VERITAS — faithfulness-first pathological speech AI.

Importing this package activates the offline bootstrap (see ``_offline``) so
model loads use the local Hugging Face cache only. The dev machine cannot
reliably reach huggingface.co; all required weights (WavLM-Large, SPARC) are
already cached.
"""
from . import _offline as _offline  # noqa: F401  (side effect: enable offline)
