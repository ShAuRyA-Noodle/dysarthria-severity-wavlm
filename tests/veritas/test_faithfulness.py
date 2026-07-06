import glob

import librosa
import numpy as np

from src.perturbations import formant_shift  # real DSP on real audio
from src.veritas.faithfulness import cycle_faithfulness
from src.veritas.inversion import SPARCInverter
from src.veritas.wavlm_features import WavLMFeatures

WAV = glob.glob(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", recursive=True)[0]


def test_faithfulness_drops_under_distortion():
    inv, fx = SPARCInverter(device="cpu"), WavLMFeatures(device="cpu")
    x, sr = librosa.load(WAV, sr=16000)
    # formant_shift(wave, sr, alpha): alpha=1.3 raises formants ~30% — a strong,
    # clearly-audible articulatory distortion pushed well off SPARC's training
    # manifold. (The perturbation study caps its dose-response sweep at 1.15 for a
    # different purpose; here we want an unambiguously strong distortion. A 15%
    # shift is mild and yields only a ~0.01 WavLM-cycle drop; per the plan's Task 3
    # Step 4, distortion strength is the lever to inspect — not a weaker assertion.)
    x_bad = formant_shift(x, sr, 1.3)
    f_clean = cycle_faithfulness(x, sr, inv, fx).mean()
    f_bad = cycle_faithfulness(x_bad, sr, inv, fx).mean()
    assert 0.0 <= f_bad <= f_clean <= 1.0
    assert f_clean - f_bad > 0.02  # measurable drop
