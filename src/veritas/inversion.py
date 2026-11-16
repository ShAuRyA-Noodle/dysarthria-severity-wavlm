"""SPARC articulatory inversion wrapper (single responsibility).

Audio <-> articulatory code. Thin wrapper over the open-source SPARC coder:
``encode`` maps audio to a dict of articulatory features (12 EMA tract
variables + pitch/loudness/periodicity/speaker embedding at 50 fps) and
``decode`` inverts the code back to a waveform. ``resynth`` chains
encode->decode and resamples the output to 16 kHz.

SPARC loads OFFLINE from the local cache because importing :mod:`src.veritas`
activates the offline bootstrap before ``sparc`` is imported here.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
from sparc import load_model


class SPARCInverter:
    """Wrap SPARC: audio -> articulatory code (ema+pitch+loudness+spk) -> audio."""

    def __init__(self, ckpt: str = "en", device: str = "cpu"):
        self.coder = load_model(ckpt, device=device)
        self.sr = int(getattr(self.coder, "sr", 16000))

    def encode(self, wav) -> dict:
        # SPARC encode accepts a file path (verified Task 0).
        return self.coder.encode(wav)

    def decode(self, code: dict) -> np.ndarray:
        w = self.coder.decode(**code)
        return np.asarray(w, dtype=np.float32).squeeze()

    def resynth(self, wav, out_sr: int = 16000) -> np.ndarray:
        code = self.encode(wav)
        w = self.decode(code)
        if self.sr != out_sr:
            w = torchaudio.functional.resample(
                torch.from_numpy(np.ascontiguousarray(w)), self.sr, out_sr
            ).numpy()
        return np.asarray(w, dtype=np.float32)
