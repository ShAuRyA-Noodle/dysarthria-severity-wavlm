import glob

import numpy as np

from src.veritas.inversion import SPARCInverter

WAV = glob.glob(r"C:/Users/Dell/Desktop/Dysar/FC/**/*.wav", recursive=True)[0]


def test_encode_decode_roundtrip():
    inv = SPARCInverter(device="cpu")
    code = inv.encode(WAV)
    assert code["ema"].shape[1] == 12
    audio16k = inv.resynth(WAV)  # decode + resample to 16 kHz
    assert audio16k.ndim == 1 and audio16k.dtype == np.float32
    assert audio16k.shape[0] > 1600  # >0.1 s of audio
