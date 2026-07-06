import numpy as np
from src.veritas.wavlm_features import WavLMFeatures


def test_features_shape_and_rate():
    fx = WavLMFeatures(layer=9, device="cpu")
    audio = np.zeros(16000, dtype=np.float32)  # 1.0 s of silence @ 16 kHz
    feats = fx.extract(audio, sr=16000)
    assert feats.ndim == 2 and feats.shape[1] == 1024
    # WavLM-Large is ~50 fps -> ~49-50 frames for 1 s
    assert 45 <= feats.shape[0] <= 51
