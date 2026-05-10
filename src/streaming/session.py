"""StreamingSession — bundles ring_buffer + ssl streamer + reference + DTW (Track A)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.constants import FASTDTW_RADIUS, TARGET_SAMPLE_RATE
from src.streaming.online_dtw import OnlineSubseqDTW
from src.streaming.reference_cache import ReferenceCache
from src.streaming.ring_buffer import AudioRingBuffer


WAVLM_FRAME_HOP_MS = 20  # WavLM-Large output frame rate ≈ 50 fps
CONTEXT_MS = 250
HOP_MS = 50


@dataclass
class StreamingSession:
    """Per-connection state for streaming ATD inference.

    The server creates one StreamingSession per WebSocket connection. The
    inference worker (decoupled via asyncio queue) calls `consume_chunk()`
    when a new audio chunk lands.
    """

    word: str
    cache: ReferenceCache
    extractor: object  # SSLExtractor — typed loosely to avoid heavy import in tests
    layer: int = 9

    sample_rate: int = TARGET_SAMPLE_RATE
    context_ms: int = CONTEXT_MS
    hop_ms: int = HOP_MS

    ring: AudioRingBuffer = field(init=False)
    dtw: Optional[OnlineSubseqDTW] = field(default=None, init=False)
    last_frame_idx: int = field(default=-1, init=False)
    started_at: float = field(default_factory=time.perf_counter, init=False)

    def __post_init__(self) -> None:
        # Hold ~10 s rolling audio.
        self.ring = AudioRingBuffer(capacity_samples=self.sample_rate * 10)
        ref = self.cache.get(self.word)
        if ref is None:
            raise ValueError(f"no reference for word {self.word!r}")
        self.dtw = OnlineSubseqDTW(ref["reference"], radius=FASTDTW_RADIUS)
        self._pca = ref["pca_model"]

    def consume_chunk(self, samples: np.ndarray) -> list[float]:
        """Feed an audio chunk; return list of new ATD scores (one per new WavLM frame)."""
        self.ring.write(samples.astype(np.float32, copy=False))
        ctx_n = int(self.sample_rate * self.context_ms / 1000)
        if len(self.ring) < ctx_n:
            return []
        # Pull last context window for WavLM forward.
        wav = self.ring.peek(ctx_n)
        feats = self.extractor.extract_all_layers(wav, self.sample_rate, [self.layer])[self.layer]
        # Project into reference PCA space.
        pca64 = self._pca.transform(feats)  # (T_ctx, 64)
        # Emit ATD updates only for *new* frames since last call (rough alignment to hop).
        n_new = max(1, int(len(pca64) * self.hop_ms / self.context_ms))
        new_frames = pca64[-n_new:]
        scores = []
        for f in new_frames:
            scores.append(self.dtw.extend(f))
            self.last_frame_idx += 1
        return scores

    def latency_ms(self) -> float:
        return (time.perf_counter() - self.started_at) * 1000.0
