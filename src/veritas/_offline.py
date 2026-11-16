"""Offline bootstrap: force transformers / Hugging Face to use the LOCAL cache.

Why: this machine's Python cannot reliably reach ``huggingface.co`` — HTTPS
connections are reset intermittently (``WinError 10054``). Every weight VERITAS
needs (WavLM-Large, SPARC) is fully cached on disk, so we do NOT need the
network. The one place transformers still probes the network is
``modeling_utils.has_file`` (it checks the Hub for a ``.safetensors`` variant
before falling back to the cached ``pytorch_model.bin``). We neutralise that
probe so loading is 100% offline and deterministic.

Real weights, real cache — nothing mocked. Import this before any
``transformers`` / ``sparc`` model load. Importing :mod:`src.veritas` does so
automatically.
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

_APPLIED = False


def enable_offline() -> None:
    """Neutralise transformers' network probe so cached weights load offline.

    Idempotent. After this call, ``PreTrainedModel.from_pretrained`` loads the
    cached ``pytorch_model.bin`` without any Hub HEAD request.
    """
    global _APPLIED
    if _APPLIED:
        return
    import transformers.modeling_utils as _mu

    # has_file() checks the Hub for a safetensors variant; returning False makes
    # transformers use the cached pytorch_model.bin with no network call.
    _mu.has_file = lambda *a, **k: False  # type: ignore[assignment]
    _APPLIED = True


enable_offline()
