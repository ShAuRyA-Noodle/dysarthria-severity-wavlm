"""Cross-corpus normalization: per-corpus CMVN, z-norm, shared-basis PCA.

API is intentionally small: every consumer imports `normalize_for_corpus`,
`zscore_atd`, and `SharedPCABasis`. All state is keyed by `corpus_name`
strings drawn from `cross_corpus.corpus_registry` (TBD).

Design rationale and validation hypotheses are described in docs/veritas/
(Track D). Current UASpeech failure (ρ = -0.22 within-corpus, ρ = -0.07
pooled) is attributed to short-frame DTW + per-corpus PCA basis mismatch;
fix = shared TORGO PCA basis applied to all corpora after per-corpus CMVN.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.constants import RANDOM_SEED


@dataclass
class CorpusStats:
    """First/second moments of frame-level SSL features for one corpus.

    Computed once per (corpus, model, layer) on control speakers only.
    """

    corpus: str
    model: str
    layer: int
    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)
    n_frames: int


@dataclass
class SharedPCABasis:
    """A PCA basis fit on a *donor* corpus and applied to all others.

    The donor is typically TORGO controls — the corpus whose articulatory
    manifold has been validated to encode dysarthric severity.
    """

    source_corpus: str
    n_components: int
    pca: PCA
    feature_mean: np.ndarray  # (D,)
    feature_std: np.ndarray  # (D,)

    def transform(self, frames: np.ndarray, target_stats: CorpusStats) -> np.ndarray:
        """Apply target-corpus CMVN, then re-anchor to source distribution, then PCA."""
        x = (frames - target_stats.mean) / np.maximum(target_stats.std, 1e-6)
        # Re-center to source mean/std so basis sees a comparable distribution.
        x = x * self.feature_std + self.feature_mean
        return self.pca.transform(x)


def fit_corpus_stats(
    control_frames: np.ndarray, corpus: str, model: str, layer: int
) -> CorpusStats:
    """Compute mean/std over concatenated control frames (one corpus)."""
    if control_frames.ndim != 2:
        raise ValueError(f"expected (N, D); got {control_frames.shape}")
    return CorpusStats(
        corpus=corpus,
        model=model,
        layer=layer,
        mean=control_frames.mean(axis=0),
        std=control_frames.std(axis=0),
        n_frames=int(control_frames.shape[0]),
    )


def fit_shared_pca(
    source_frames: np.ndarray,
    source_corpus: str,
    n_components: int = 64,
) -> SharedPCABasis:
    """Fit a PCA basis once on the donor corpus (typically TORGO controls)."""
    fmean = source_frames.mean(axis=0)
    fstd = source_frames.std(axis=0)
    centered = (source_frames - fmean) / np.maximum(fstd, 1e-6)
    pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
    pca.fit(centered)
    return SharedPCABasis(
        source_corpus=source_corpus,
        n_components=n_components,
        pca=pca,
        feature_mean=fmean,
        feature_std=fstd,
    )


def normalize_for_corpus(
    frames: np.ndarray,
    stats: CorpusStats,
    mode: str = "cmvn",
) -> np.ndarray:
    """Apply per-corpus normalization.

    Modes
    -----
    cmvn : (x - mu_C) / sigma_C  (channel-mean-and-variance normalization)
    cmn  : (x - mu_C)             (mean-only, keeps amplitude)
    none : identity                (sanity baseline)
    """
    if mode == "none":
        return frames
    if mode == "cmn":
        return frames - stats.mean
    if mode == "cmvn":
        return (frames - stats.mean) / np.maximum(stats.std, 1e-6)
    raise ValueError(f"unknown mode: {mode!r}")


def zscore_atd(atd_scores: np.ndarray, control_atd: np.ndarray) -> np.ndarray:
    """Per-corpus z-score of ATD using control mean/std as anchor."""
    mu = float(control_atd.mean())
    sd = float(control_atd.std() or 1.0)
    return (atd_scores - mu) / sd


def zrank_severity(severities: np.ndarray) -> np.ndarray:
    """Z-rank severity within a corpus.

    Invariant to scale and to monotone transforms; preserves the exact
    within-corpus ordering used by Spearman ρ. The unified scale used in
    pooled mixed-effects models (`y ~ atd + (1|corpus)`).
    """
    severities = np.asarray(severities, dtype=float)
    ranks = np.argsort(np.argsort(severities)) + 1.0
    if len(ranks) < 2:
        return np.zeros_like(ranks)
    return (ranks - ranks.mean()) / (ranks.std() or 1.0)


def save_stats(stats: CorpusStats, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_stats(path: Path) -> CorpusStats:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_basis(basis: SharedPCABasis, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(basis, f)


def load_basis(path: Path) -> SharedPCABasis:
    with open(path, "rb") as f:
        return pickle.load(f)
