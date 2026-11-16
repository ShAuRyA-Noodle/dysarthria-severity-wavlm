"""Smoke tests for Track A/C/D prototype scaffolds.

Run with::

    pytest tests/test_track_prototypes.py -q

Verifies imports and minimal correctness of the new prototype modules
without requiring real audio.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_track_a_online_dtw_imports() -> None:
    from src.streaming.online_dtw import OnlineSubseqDTW  # noqa: F401


def test_track_a_online_dtw_runs() -> None:
    from src.streaming.online_dtw import OnlineSubseqDTW, exact_dtw_full

    rng = np.random.default_rng(42)
    ref = rng.standard_normal((40, 64)).astype(np.float32)
    qry = rng.standard_normal((30, 64)).astype(np.float32)

    # Default (unbanded) classic DTW: every running cost to the reference end
    # is finite because the closed-start full path can always reach row |R|-1.
    dtw = OnlineSubseqDTW(ref)
    scores = [dtw.extend(f) for f in qry]

    assert len(scores) == 30
    assert all(np.isfinite(s) and s >= 0 for s in scores)
    # Final ATD should be the running cost normalized by max(t+1, R).
    assert scores[-1] == dtw.last_atd
    # Final accumulated cost must equal the exact full-DP DTW (no free column).
    expected = exact_dtw_full(qry, ref)
    assert dtw.accumulated_cost == pytest.approx(expected, rel=1e-9, abs=1e-9)
    assert dtw.last_atd == pytest.approx(expected / max(len(qry), len(ref)), rel=1e-9)


def test_track_a_online_dtw_reproducible_extension() -> None:
    """Extending one frame at a time matches the running internal state."""
    from src.streaming.online_dtw import OnlineSubseqDTW

    rng = np.random.default_rng(0)
    ref = rng.standard_normal((20, 16)).astype(np.float32)
    qry = rng.standard_normal((15, 16)).astype(np.float32)

    dtw = OnlineSubseqDTW(ref, radius=8)
    [dtw.extend(f) for f in qry]
    final_state = dtw.last_atd

    # Re-run from scratch on the same query.
    dtw2 = OnlineSubseqDTW(ref, radius=8)
    [dtw2.extend(f) for f in qry]
    assert dtw2.last_atd == pytest.approx(final_state, rel=1e-6)


def test_track_d_normalization_imports() -> None:
    from src.cross_corpus.normalization import (  # noqa: F401
        CorpusStats,
        SharedPCABasis,
        fit_corpus_stats,
        fit_shared_pca,
        normalize_for_corpus,
        zrank_severity,
        zscore_atd,
    )


def test_track_d_corpus_stats_shape() -> None:
    from src.cross_corpus.normalization import fit_corpus_stats

    rng = np.random.default_rng(7)
    ctrl = rng.standard_normal((300, 1024)).astype(np.float32)
    s = fit_corpus_stats(ctrl, "torgo", "wavlm_large", 9)
    assert s.mean.shape == (1024,)
    assert s.std.shape == (1024,)
    assert s.n_frames == 300


def test_track_d_shared_pca_transforms() -> None:
    from src.cross_corpus.normalization import fit_corpus_stats, fit_shared_pca

    rng = np.random.default_rng(11)
    ctrl_torgo = rng.standard_normal((500, 1024)).astype(np.float32) + 0.5
    basis = fit_shared_pca(ctrl_torgo, "torgo", n_components=64)

    ctrl_target = rng.standard_normal((200, 1024)).astype(np.float32) - 0.3
    target_stats = fit_corpus_stats(ctrl_target, "uaspeech", "wavlm_large", 9)

    test = rng.standard_normal((50, 1024)).astype(np.float32)
    out = basis.transform(test, target_stats)
    assert out.shape == (50, 64)
    assert np.isfinite(out).all()


def test_track_d_zrank_severity_invariants() -> None:
    from src.cross_corpus.normalization import zrank_severity

    zr = zrank_severity([0, 0, 0, 1, 1, 2, 2, 2])
    assert zr.mean() == pytest.approx(0.0, abs=1e-9)
    assert zr.std() > 0.9
    # Monotone transform leaves z-rank unchanged.
    zr_log = zrank_severity([0, 0, 0, 1, 1, 100, 100, 100])
    np.testing.assert_array_almost_equal(zr, zr_log)


def test_track_c_perturbations_pure_python_only() -> None:
    """Smoke-import perturbations module. Praat-dependent functions only fail
    at call-time if parselmouth is missing — registry should always import."""
    from src.perturbations import PERTURBATIONS, perturb  # noqa: F401

    assert set(PERTURBATIONS.keys()) == {
        "time_stretch", "pink_noise", "formant_shift",
        "partial_mute", "reverberate", "pitch_shift",
    }
    # Each entry is (callable, list_of_intensities).
    for fn, intensities in PERTURBATIONS.values():
        assert callable(fn)
        assert len(intensities) >= 5


def test_track_c_pink_noise_runs_without_praat() -> None:
    """pink_noise + partial_mute + reverberate use only numpy/scipy — must work."""
    from src.perturbations import partial_mute, pink_noise, reverberate

    rng = np.random.default_rng(3)
    wave = rng.standard_normal(16000 * 2).astype(np.float32)

    out_pn = pink_noise(wave, 16000, snr_db=10, key="utt1")
    assert out_pn.shape == wave.shape
    assert np.isfinite(out_pn).all()

    out_mute = partial_mute(wave, 16000, drop_frac=0.1, key="utt1")
    assert out_mute.shape == wave.shape
    assert np.isfinite(out_mute).all()

    out_rev = reverberate(wave, 16000, rt60_s=0.3, key="utt1")
    assert out_rev.shape == wave.shape
    assert np.isfinite(out_rev).all()
