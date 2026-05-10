"""DSP perturbation library for Track C causal validation.

Six families that distort neurotypical control speech in known ways. Each
function takes (waveform, sample_rate, intensity) and returns a perturbed
waveform of the same shape and dtype. All perturbations are deterministic
given (family, intensity, key).

Predicted ATD response (locked pre-registration, see paper/appendix/
track_c_preregistration.md):
    β(pink_noise) < β(time_stretch) ≈ β(reverberate) ≈ β(pitch_shift)
                  ≪ β(formant_shift) ≈ β(partial_mute)
"""
from __future__ import annotations

import hashlib

import numpy as np

try:
    import parselmouth
    from parselmouth.praat import call as _praat_call
    _HAS_PRAAT = True
except ImportError:
    _HAS_PRAAT = False

from scipy.signal import fftconvolve


SR_DEFAULT = 16000


def _rng(family: str, intensity: float, key: str = "") -> np.random.Generator:
    """Deterministic per-(family, intensity, utterance) RNG."""
    h = hashlib.md5(f"{family}|{intensity}|{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(h[:8], "little"))


def _to_snd(wave: np.ndarray, sr: int):
    if not _HAS_PRAAT:
        raise RuntimeError("parselmouth not installed (pip install praat-parselmouth)")
    return parselmouth.Sound(values=wave.astype(np.float64), sampling_frequency=sr)


def _peak_norm(x: np.ndarray, target_dbfs: float = -20.0) -> np.ndarray:
    rms = np.sqrt(np.mean(x ** 2) + 1e-12)
    target_rms = 10 ** (target_dbfs / 20.0)
    return (x * (target_rms / rms)).astype(np.float32, copy=False)


def time_stretch(wave: np.ndarray, sr: int, rate: float, key: str = "") -> np.ndarray:
    """PSOLA time-stretch preserving pitch + formants. ATD should be ~invariant."""
    snd = _to_snd(wave, sr)
    out = _praat_call(snd, "Lengthen (overlap-add)", 75.0, 600.0, 1.0 / rate)
    return _peak_norm(np.asarray(out.values).squeeze())


def pink_noise(wave: np.ndarray, sr: int, snr_db: float, key: str = "") -> np.ndarray:
    """Add 1/f pink noise at target SNR. WavLM denoising-pretrained, expect small ΔATD."""
    rng = _rng("pink", snr_db, key)
    n = len(wave)
    white = rng.standard_normal(n)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    f[0] = f[1]
    spec = np.fft.rfft(white) / np.sqrt(f)
    pink = np.fft.irfft(spec, n=n)
    rms_s = np.sqrt(np.mean(wave ** 2) + 1e-12)
    rms_n = np.sqrt(np.mean(pink ** 2) + 1e-12)
    scale = rms_s / (rms_n * 10 ** (snr_db / 20.0))
    return _peak_norm(wave + (pink * scale).astype(wave.dtype))


def formant_shift(wave: np.ndarray, sr: int, alpha: float, key: str = "") -> np.ndarray:
    """LPC-resynth formant shift (alpha > 1 raises formants). True articulatory distortion."""
    snd = _to_snd(wave, sr)
    out = _praat_call(snd, "Change gender", 75.0, 600.0, alpha, 0.0, 1.0, 1.0)
    return _peak_norm(np.asarray(out.values).squeeze())


def partial_mute(wave: np.ndarray, sr: int, drop_frac: float, key: str = "") -> np.ndarray:
    """Zero out a Bernoulli(drop_frac) sample of 50 ms windows with 5 ms ramps."""
    rng = _rng("mute", drop_frac, key)
    win = int(0.050 * sr)
    ramp = int(0.005 * sr)
    out = wave.astype(np.float32, copy=True)
    env = np.ones(win, dtype=np.float32)
    r = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, ramp)))
    env[:ramp] = 1.0 - r
    env[-ramp:] = 1.0 - r[::-1]
    for start in range(0, len(out) - win, win):
        if rng.random() < drop_frac:
            out[start : start + win] *= (1.0 - env)
    return _peak_norm(out)


def reverberate(wave: np.ndarray, sr: int, rt60_s: float, key: str = "") -> np.ndarray:
    """Synthetic exponential-decay IR convolution. Channel-only distortion."""
    rng = _rng("reverb", rt60_s, key)
    L = int(1.5 * rt60_s * sr) + 1
    decay = np.exp(-6.9078 * np.arange(L) / (rt60_s * sr))
    h = rng.standard_normal(L) * decay
    h /= np.sqrt(np.sum(h ** 2) + 1e-12)
    y = fftconvolve(wave, h)[: len(wave)]
    return _peak_norm(y.astype(np.float32))


def pitch_shift(wave: np.ndarray, sr: int, semitones: float, key: str = "") -> np.ndarray:
    """PSOLA pitch shift with formant ratio = 1.0 (vowel quality preserved)."""
    snd = _to_snd(wave, sr)
    factor = 2 ** (semitones / 12.0)
    base = _praat_call(snd, "Get mean pitch", 0.0, 0.0, "Hertz") or 150.0
    out = _praat_call(snd, "Change gender", 75.0, 600.0, 1.0, base * factor, 1.0, 1.0)
    return _peak_norm(np.asarray(out.values).squeeze())


PERTURBATIONS = {
    "time_stretch": (time_stretch, [0.6, 0.8, 1.0, 1.25, 1.5, 1.75]),
    "pink_noise": (pink_noise, [30, 20, 10, 5, 0, -5]),
    "formant_shift": (formant_shift, [0.85, 0.90, 0.95, 1.05, 1.10, 1.15]),
    "partial_mute": (partial_mute, [0.02, 0.05, 0.10, 0.20, 0.30]),
    "reverberate": (reverberate, [0.1, 0.3, 0.6, 1.0, 1.5]),
    "pitch_shift": (pitch_shift, [-6, -3, -1, 1, 3, 6]),
}


def perturb(
    wave: np.ndarray, sr: int, family: str, intensity: float, key: str = ""
) -> np.ndarray:
    """Dispatch a single perturbation by family name."""
    if family not in PERTURBATIONS:
        raise KeyError(f"unknown family {family!r}; available: {list(PERTURBATIONS)}")
    fn, _ = PERTURBATIONS[family]
    return fn(wave, sr, intensity, key=key)
