import glob

import librosa
import numpy as np
import torch

from src.perturbations import formant_shift  # real DSP on real audio
from src.veritas.faithfulness import (
    HealthyReference,
    OODScorer,
    cycle_faithfulness,
    frame_faithfulness,
)
from src.veritas.inversion import SPARCInverter
from src.veritas.wavlm_features import WavLMFeatures

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _ctrl_wavs(grp, spk, lo=1.5, hi=3.5, n=8):
    """Deterministic short head-mic control clips (real TORGO audio)."""
    ws = sorted(
        glob.glob(rf"C:/Users/Dell/Desktop/Dysar/{grp}/{spk}/**/wav_headMic/*.wav", recursive=True)
    )
    out = []
    for w in ws:
        x, sr = librosa.load(w, sr=16000)
        if lo <= len(x) / sr <= hi:
            out.append((x, sr))
        if len(out) >= n:
            break
    return out


def _fit_healthy_reference(fx):
    """Fit the healthy-manifold centroid on control frames DISJOINT from the
    eval speakers used below (fit: FC01+MC01; eval: FC02/MC02/FC03)."""
    fit = _ctrl_wavs("FC", "FC01", n=6) + _ctrl_wavs("MC", "MC01", n=6)
    frames = np.concatenate([fx.extract(x, sr=sr) for x, sr in fit], 0)
    return HealthyReference().fit(frames)


def test_centered_faithfulness_monotone_dose_response():
    """Discrimination: under a formant-shift dose sweep the CENTERED metric must
    drop MONOTONICALLY, and centering (against the healthy manifold) must give a
    CLEARLY larger clean->distorted separation than the raw/uncentered cosine
    (the anisotropy fix recovers dynamic range). Averaged over >=2 real control
    wavs; fit frames disjoint from eval wavs."""
    inv, fx = SPARCInverter(device=DEVICE), WavLMFeatures(device=DEVICE)
    ref = _fit_healthy_reference(fx)
    # Average over several real control wavs (>=2 required by the plan; we use 8
    # across 4 held-out control speakers so the small middle-dose steps resolve
    # above per-utterance noise — the dose-response is genuinely monotone, it
    # just needs enough averaging to see it, not a weaker assertion).
    ev = (
        _ctrl_wavs("FC", "FC02", n=2)
        + _ctrl_wavs("MC", "MC02", n=2)
        + _ctrl_wavs("FC", "FC03", n=2)
        + _ctrl_wavs("MC", "MC03", n=2)
    )
    assert len(ev) >= 6, "need several real control wavs for a stable dose-response average"

    alphas = [1.0, 1.1, 1.2, 1.3]
    unc, cen = [], []
    for a in alphas:
        u, c = [], []
        for x, sr in ev:
            xa = x if a == 1.0 else formant_shift(x, sr, a)  # clean baseline at 1.0
            xhat = inv.resynth(xa)
            e_x = fx.extract(xa, sr=sr)
            e_h = fx.extract(xhat, sr=16000)
            # one resynth+extract feeds BOTH metrics — same cycle, differ only in centering
            u.append(float(frame_faithfulness(e_x, e_h).mean()))
            c.append(float(frame_faithfulness(e_x, e_h, ref).mean()))
        unc.append(float(np.mean(u)))
        cen.append(float(np.mean(c)))
    unc, cen = np.array(unc), np.array(cen)

    # centered dose-response is monotone non-increasing (tol only for float noise)
    assert np.all(np.diff(cen) <= 1e-3), f"centered not monotone: {cen}"
    cen_sep, unc_sep = cen[0] - cen[-1], unc[0] - unc[-1]
    assert cen_sep > 0.02, f"centered clean->distorted separation too small: {cen_sep:.4f}"
    # anisotropy fix: centered separation is CLEARLY larger than uncentered
    assert cen_sep > 1.2 * unc_sep, (
        f"centered sep {cen_sep:.4f} not clearly > uncentered sep {unc_sep:.4f}"
    )
    assert (cen >= 0).all() and (cen <= 1).all()


def _feats(pattern, fx, sl=slice(0, 3)):
    out = []
    for w in sorted(glob.glob(pattern, recursive=True))[sl]:
        x, sr = librosa.load(w, sr=16000)
        out.append(fx.extract(x, sr=16000))
    return np.concatenate(out, 0)


def test_ood_higher_for_dysarthric():
    fx = WavLMFeatures(device=DEVICE)
    fit_ctrl = _feats(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", fx, sl=slice(0, 3))
    eval_ctrl = _feats(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", fx, sl=slice(3, 6))  # DISJOINT
    dys = _feats(r"C:/Users/Dell/Desktop/Dysar/F/**/*.wav", fx, sl=slice(0, 3))  # dysarthric
    ood = OODScorer().fit(fit_ctrl)
    assert ood.score(dys).mean() > ood.score(eval_ctrl).mean()
