"""Lock-free ring buffer for streaming audio + WavLM frames (Track A)."""
from __future__ import annotations

import numpy as np


class AudioRingBuffer:
    """Fixed-capacity ring buffer for raw int16 / float32 audio samples.

    Single-producer (network thread) / single-consumer (inference thread).
    Designed for 50 ms PCM chunks at 16 kHz ⇒ 800 samples per chunk.
    """

    def __init__(self, capacity_samples: int, dtype=np.float32) -> None:
        self.capacity = int(capacity_samples)
        self.dtype = dtype
        self.buffer = np.zeros(self.capacity, dtype=dtype)
        self.write_idx = 0
        self.read_idx = 0
        self.size = 0

    def write(self, samples: np.ndarray) -> int:
        """Append samples. Returns count actually written (may drop on overflow)."""
        n = len(samples)
        if n == 0:
            return 0
        free = self.capacity - self.size
        if n > free:
            n = free  # drop tail; producer should have flushed
        end = self.write_idx + n
        if end <= self.capacity:
            self.buffer[self.write_idx:end] = samples[:n].astype(self.dtype, copy=False)
        else:
            tail = self.capacity - self.write_idx
            self.buffer[self.write_idx:] = samples[:tail].astype(self.dtype, copy=False)
            self.buffer[: end - self.capacity] = samples[tail:n].astype(self.dtype, copy=False)
        self.write_idx = end % self.capacity
        self.size += n
        return n

    def peek(self, n: int) -> np.ndarray:
        """Read n most-recent samples without advancing read pointer."""
        n = min(n, self.size)
        if n == 0:
            return np.zeros(0, dtype=self.dtype)
        start = (self.write_idx - n) % self.capacity
        if start + n <= self.capacity:
            return self.buffer[start:start + n].copy()
        head = self.capacity - start
        out = np.empty(n, dtype=self.dtype)
        out[:head] = self.buffer[start:]
        out[head:] = self.buffer[:n - head]
        return out

    def consume(self, n: int) -> np.ndarray:
        """Read and advance. Returns up to n samples."""
        n = min(n, self.size)
        if n == 0:
            return np.zeros(0, dtype=self.dtype)
        if self.read_idx + n <= self.capacity:
            out = self.buffer[self.read_idx:self.read_idx + n].copy()
        else:
            head = self.capacity - self.read_idx
            out = np.empty(n, dtype=self.dtype)
            out[:head] = self.buffer[self.read_idx:]
            out[head:] = self.buffer[:n - head]
        self.read_idx = (self.read_idx + n) % self.capacity
        self.size -= n
        return out

    def __len__(self) -> int:
        return self.size
