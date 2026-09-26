"""Exercise the frozen WavLM loading and inference path without TORGO data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
from transformers import AutoModel, WavLMConfig

from src.ssl_extractor import MODELS_CONFIG, SSLExtractor, load_processed_audio


class SSLExtractorSmokeTest(unittest.TestCase):
    """Verify that supported Transformers can load and run a WavLM model."""

    def test_processed_mono_wav_loading(self) -> None:
        """Read the PCM WAV format produced by the preprocessing stage."""
        with tempfile.TemporaryDirectory() as audio_dir:
            audio_path = Path(audio_dir) / "sample.wav"
            sf.write(audio_path, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
            waveform, sample_rate = load_processed_audio(audio_path)

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(waveform.shape, (16000,))
        self.assertEqual(waveform.dtype, np.float32)

    def test_frozen_wavlm_hidden_states(self) -> None:
        """Load a tiny local checkpoint and extract both transformer layers."""
        config = WavLMConfig(
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8, 8, 8, 8, 8, 8),
            num_conv_pos_embeddings=16,
            num_conv_pos_embedding_groups=2,
            mask_time_prob=0.0,
        )

        with tempfile.TemporaryDirectory() as model_dir:
            AutoModel.from_config(config).save_pretrained(model_dir)
            test_config = {
                "hf_name": model_dir,
                "hidden_dim": 16,
                "n_layers": 2,
                "feature_layer": "hidden_states",
            }
            with patch.dict(MODELS_CONFIG, {"tiny_wavlm": test_config}):
                extractor = SSLExtractor("tiny_wavlm", device="cpu", use_fp16=False)
                self.assertFalse(extractor.model.training)
                self.assertTrue(
                    all(not parameter.requires_grad for parameter in extractor.model.parameters())
                )

                embeddings = extractor.extract_all_layers(
                    np.zeros(16000, dtype=np.float32), target_layers=[1, 2]
                )

        self.assertEqual(set(embeddings), {1, 2})
        for layer in embeddings.values():
            self.assertEqual(layer.ndim, 2)
            self.assertEqual(layer.shape[1], 16)
            self.assertGreater(layer.shape[0], 0)


if __name__ == "__main__":
    unittest.main()
