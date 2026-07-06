"""Frozen WavLM-Large frame-feature extractor (single responsibility).

Audio array -> per-frame hidden states at a chosen transformer layer.
WavLM-Large runs at ~50 fps and emits 1024-d features. The project's headline
analysis layer is 9. Model loads are OFFLINE from the local HF cache because
importing :mod:`src.veritas` activates the offline bootstrap first.

Preprocessing note: WavLM-Large uses ``do_normalize=True`` (zero-mean,
unit-variance waveform normalisation). We apply that normalisation directly
rather than via ``AutoFeatureExtractor`` — only the model weights are in the
local cache (``preprocessor_config.json`` is not), and this matches the
preprocessing already used across the repo (``src.ssl_extractor.SSLExtractor``),
so features are identical to the project's headline pipeline.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio
from transformers import AutoModel


class WavLMFeatures:
    """Frozen WavLM-Large frame features at a chosen transformer layer (50 fps)."""

    def __init__(
        self,
        layer: int = 9,
        device: str = "cpu",
        model_name: str = "microsoft/wavlm-large",
    ):
        self.layer = layer
        self.device = device
        self.model = (
            AutoModel.from_pretrained(model_name, output_hidden_states=True)
            .to(device)
            .eval()
        )
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def extract(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32)
        if sr != 16000:
            audio = torchaudio.functional.resample(
                torch.from_numpy(audio), sr, 16000
            ).numpy()
        wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)  # (1, T)
        # WavLM do_normalize=True: zero-mean, unit-variance (matches SSLExtractor).
        mean = wav.mean()
        std = wav.std()
        if std > 0:
            wav = (wav - mean) / std
        wav = wav.to(self.device)
        hs = self.model(wav).hidden_states  # tuple: [cnn_out, layer1..layer24]
        return hs[self.layer].squeeze(0).cpu().numpy()  # (T, 1024)
