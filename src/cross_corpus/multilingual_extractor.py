"""Multilingual SSL extractor wrapper (Track D).

Single API over WavLM-Large (English-pretrained) and XLS-R-300m / XLS-R-1B
(128-language multilingual). All models loaded frozen. Per the roadmap,
XLS-R-300m is the default cross-lingual workhorse; WavLM-Large is the
English-only baseline.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    from transformers import AutoModel
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


MODEL_REGISTRY = {
    "wavlm_large":   {"hf_name": "microsoft/wavlm-large",                "n_layers": 24, "hidden_dim": 1024},
    "xlsr_300m":     {"hf_name": "facebook/wav2vec2-xls-r-300m",         "n_layers": 24, "hidden_dim": 1024},
    "xlsr_1b":       {"hf_name": "facebook/wav2vec2-xls-r-1b",           "n_layers": 48, "hidden_dim": 1280},
    "hubert_large":  {"hf_name": "facebook/hubert-large-ll60k",          "n_layers": 24, "hidden_dim": 1024},
}


class MultilingualExtractor:
    """Frozen multilingual SSL feature extractor.

    Drop-in replacement for `src.ssl_extractor.SSLExtractor` when crossing
    languages. Uses the same forward + `output_hidden_states` API.
    """

    def __init__(
        self,
        model_name: str = "xlsr_300m",
        device: str = "auto",
        use_fp16: bool = True,
    ) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch + transformers required")
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"unknown model {model_name!r}; known: {sorted(MODEL_REGISTRY)}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg = MODEL_REGISTRY[model_name]
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16 and device == "cuda"
        self.cfg = cfg

        self.model = AutoModel.from_pretrained(cfg["hf_name"], output_hidden_states=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        if self.use_fp16:
            self.model = self.model.half()
        self.model = self.model.to(device)

    @torch.no_grad()
    def extract_all_layers(
        self,
        waveform: np.ndarray,
        sample_rate: int = 16000,
        target_layers: Optional[list[int]] = None,
    ) -> dict[int, np.ndarray]:
        if target_layers is None:
            target_layers = list(range(1, self.cfg["n_layers"] + 1))
        wav = torch.from_numpy(waveform).float()
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        # Per-utterance z-norm (matches src/ssl_extractor.py behavior).
        m, s = wav.mean(), wav.std()
        if s > 0:
            wav = (wav - m) / s
        if self.use_fp16:
            wav = wav.half()
        wav = wav.to(self.device)
        out = self.model(wav, output_hidden_states=True)
        hs = out.hidden_states
        result = {}
        for L in target_layers:
            if 0 <= L < len(hs):
                result[L] = hs[L].squeeze(0).float().cpu().numpy()
        return result
