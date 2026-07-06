"""GT-free cycle-consistency faithfulness — the novel VERITAS core.

Idea: run audio through the SPARC articulatory bottleneck and back
(encode -> decode), then measure how much the re-synthesis drifts from the
input *in WavLM-Large feature space* (cycle consistency). High faithfulness =
the articulatory code preserved the input; low faithfulness = the invertible
model could not represent this frame and "normalised" it toward its training
manifold. No ground-truth articulography is required.

``fused_faithfulness`` combines the cycle signal with an out-of-distribution
penalty fit on healthy speech (``OODScorer``).
"""
from __future__ import annotations

import numpy as np


def _cos_per_frame(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    T = min(len(a), len(b))
    a, b = a[:T], b[:T]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return num / den  # cosine in [-1, 1]


def _to16k(audio, sr):
    import torch
    import torchaudio

    return torchaudio.functional.resample(
        torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32)), sr, 16000
    ).numpy()


def cycle_faithfulness(audio: np.ndarray, sr: int, inverter, wavlm) -> np.ndarray:
    """GT-free per-frame faithfulness in [0,1]: how much the articulatory
    bottleneck preserves the input, measured in WavLM space after re-synthesis.
    audio->encode->decode->re-encode(WavLM); compare WavLM(x) vs WavLM(x_hat)."""
    xhat = inverter.resynth(audio if sr == 16000 else _to16k(audio, sr))
    fx = wavlm.extract(audio, sr=sr)
    fxh = wavlm.extract(xhat, sr=16000)
    cos = _cos_per_frame(fx, fxh)
    return np.clip((cos + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1] -> [0,1]
