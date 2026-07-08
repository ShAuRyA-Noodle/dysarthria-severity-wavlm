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


class HealthyReference:
    """Healthy-manifold centroid: the mean of control-speaker WavLM frames.

    Anisotropy fix. Raw cosine on WavLM features is saturated by a large shared
    component (every frame points in a similar direction), which compresses all
    faithfulness into a narrow band and destroys discrimination. Subtracting the
    healthy centroid ``mu`` measures cosine on the *deviation-from-healthy*
    subspace — the actual concept — and recovers dynamic range.

    Fit ONCE on control frames extracted at the SAME WavLM layer later used for
    ``E(x)``; ``center`` then subtracts ``mu``. With ``mu is None`` (unfit)
    ``center`` is the identity, i.e. the uncentered fallback.
    """

    def __init__(self, mu: np.ndarray | None = None):
        self.mu = None if mu is None else np.asarray(mu, dtype=np.float32)

    def fit(self, control_frames: np.ndarray) -> "HealthyReference":
        # float64 accumulation for a stable centroid, stored float32 to match feats
        self.mu = np.asarray(control_frames, dtype=np.float64).mean(0).astype(np.float32)
        return self

    def center(self, feats: np.ndarray) -> np.ndarray:
        feats = np.asarray(feats)
        return feats if self.mu is None else feats - self.mu


def frame_faithfulness(
    fx: np.ndarray, fxh: np.ndarray, healthy_ref: "HealthyReference | None" = None
) -> np.ndarray:
    """Per-frame faithfulness in [0,1] from WavLM features of x and x_hat.

    Pure cycle consistency: ``cosine(E(x), E(x_hat))`` mapped [-1,1]->[0,1]. When
    a fitted ``HealthyReference`` is given, both feature sets are centered against
    the healthy manifold first (anisotropy fix). This depends ONLY on how well
    the reconstruction preserved the input (E(x) vs E(x_hat)) — never on how far
    the INPUT itself sits from the healthy manifold (see module docstring)."""
    if healthy_ref is not None:
        fx = healthy_ref.center(fx)
        fxh = healthy_ref.center(fxh)
    cos = _cos_per_frame(np.asarray(fx), np.asarray(fxh))
    return np.clip((cos + 1.0) / 2.0, 0.0, 1.0)  # map [-1,1] -> [0,1]


def cycle_faithfulness(
    audio: np.ndarray, sr: int, inverter, wavlm, healthy_ref: "HealthyReference | None" = None
) -> np.ndarray:
    """GT-free per-frame faithfulness in [0,1]: how much the articulatory
    bottleneck preserves the input, measured in WavLM space after re-synthesis.
    audio->encode->decode->re-encode(WavLM); compare WavLM(x) vs WavLM(x_hat).

    Pass a fitted ``HealthyReference`` to center against the healthy manifold
    (recommended — recovers dynamic range); ``healthy_ref=None`` is the
    uncentered fallback."""
    xhat = inverter.resynth(audio if sr == 16000 else _to16k(audio, sr))
    fx = wavlm.extract(audio, sr=sr)
    fxh = wavlm.extract(xhat, sr=16000)
    return frame_faithfulness(fx, fxh, healthy_ref)


class OODScorer:
    """Diagonal-covariance Mahalanobis distance to the healthy feature manifold.

    Fit on healthy (control) WavLM frames; ``score`` returns a per-frame
    out-of-distribution distance. Disordered speech, being off the healthy
    manifold, scores higher — the anti-normalisation half of the fused signal.
    """

    def fit(self, healthy_feats: np.ndarray):
        self.mu = healthy_feats.mean(0)
        self.var = healthy_feats.var(0) + 1e-6
        return self

    def score(self, feats: np.ndarray) -> np.ndarray:
        d = (feats - self.mu) ** 2 / self.var
        return np.sqrt(d.sum(1))  # per-frame OOD distance


def fused_faithfulness(
    cycle_F: np.ndarray, ood: np.ndarray, ood_ref: float
) -> np.ndarray:
    """Combine cycle faithfulness with an OOD penalty normalised by a healthy
    reference OOD level. Returns per-frame faithfulness in [0,1]."""
    T = min(len(cycle_F), len(ood))
    ood_pen = np.clip(1.0 - (ood[:T] / (ood_ref + 1e-8) - 1.0), 0.0, 1.0)
    return np.clip(0.5 * cycle_F[:T] + 0.5 * ood_pen, 0.0, 1.0)
