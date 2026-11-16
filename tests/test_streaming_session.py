"""Smoke tests for Track A streaming session + ring buffer."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_ring_buffer_write_read_roundtrip() -> None:
    from src.streaming.ring_buffer import AudioRingBuffer

    buf = AudioRingBuffer(capacity_samples=1000)
    samples = np.arange(800, dtype=np.float32)
    n = buf.write(samples)
    assert n == 800
    assert len(buf) == 800

    peek = buf.peek(800)
    np.testing.assert_array_equal(peek, samples)

    consumed = buf.consume(400)
    np.testing.assert_array_equal(consumed, samples[:400])
    assert len(buf) == 400


def test_ring_buffer_wraps_correctly() -> None:
    from src.streaming.ring_buffer import AudioRingBuffer

    buf = AudioRingBuffer(capacity_samples=10)
    buf.write(np.arange(8, dtype=np.float32))
    buf.consume(6)
    # Should wrap on the next write of 6 samples.
    buf.write(np.arange(100, 106, dtype=np.float32))
    out = buf.consume(8)
    expected = np.array([6, 7, 100, 101, 102, 103, 104, 105], dtype=np.float32)
    np.testing.assert_array_equal(out, expected)


def test_ring_buffer_overflow_drops_tail() -> None:
    from src.streaming.ring_buffer import AudioRingBuffer

    buf = AudioRingBuffer(capacity_samples=5)
    written = buf.write(np.arange(10, dtype=np.float32))
    assert written == 5  # only the first 5 fit


def test_streaming_schemas_validate() -> None:
    from src.streaming.schemas import (
        ATDUpdate, AudioChunk, SessionEnd, SessionStart, StreamError, WordEvent,
    )

    SessionStart(word="banana", client_ts=0)
    AudioChunk(seq=0, pcm16_b64="abc", client_ts=0)
    ATDUpdate(seq=0, frame_idx=0, atd=0.0, path_tip=(0, 0), server_ts=0)
    WordEvent(word="banana", start_frame=0, end_frame=10, conf=0.9)
    SessionEnd(final_atd=0.5, n_frames=100, latency_p50_ms=120.0)
    StreamError(code="NO_REFERENCE", msg="x")


def test_streaming_online_dtw_short_run_matches_manual() -> None:
    """Reproducibility: same RNG -> identical final ATD."""
    from src.streaming.online_dtw import OnlineSubseqDTW

    rng = np.random.default_rng(7)
    ref = rng.standard_normal((10, 4)).astype(np.float32)
    qry = rng.standard_normal((8, 4)).astype(np.float32)
    a = OnlineSubseqDTW(ref, radius=20)
    for f in qry:
        a.extend(f)
    b = OnlineSubseqDTW(ref, radius=20)
    for f in qry:
        b.extend(f)
    assert a.last_atd == pytest.approx(b.last_atd, rel=1e-9)
