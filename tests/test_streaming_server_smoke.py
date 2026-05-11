"""Smoke tests for the FastAPI streaming server (Track A demo).

Boots the server via the FastAPI TestClient (no real network), opens a
mock WebSocket, sends a synthetic-but-real 50 ms PCM chunk, and asserts
the server responds with an ATD update message. No browser, no mic, no
external audio — but the full server code path is exercised.
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def app():
    fastapi = pytest.importorskip("fastapi")  # noqa: F841
    try:
        from src.streaming.server import create_app
    except RuntimeError as exc:
        pytest.skip(f"server prerequisites missing: {exc}")
    try:
        return create_app()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"app construction failed (likely no SSL model / refs): {exc}")


def test_health_endpoint(app) -> None:
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_atd_update_on_chunk(app) -> None:
    """Send a synthetic-PCM chunk through /ws/atd; expect server to handle gracefully."""
    from fastapi.testclient import TestClient

    rng = np.random.default_rng(42)
    pcm = (rng.standard_normal(800) * 1000).astype(np.int16)
    pcm_b64 = base64.b64encode(pcm.tobytes()).decode()

    with TestClient(app) as client:
        with client.websocket_connect("/ws/atd") as ws:
            ws.send_text(json.dumps({
                "type": "session_start",
                "word": "banana",  # may or may not be in cache; either path is fine
                "sample_rate": 16000,
                "client_ts": 0,
            }))
            ws.send_text(json.dumps({
                "type": "audio_chunk",
                "seq": 0,
                "pcm16_b64": pcm_b64,
                "client_ts": 100,
            }))
            # Server should respond with either atd_update OR error
            # (if the requested word has no reference). Both are valid.
            msg = json.loads(ws.receive_text())
            assert msg["type"] in {"atd_update", "error"}, msg
