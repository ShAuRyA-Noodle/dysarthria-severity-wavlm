"""FastAPI WebSocket server for streaming ATD inference (Track A scaffold).

Run with::

    uvicorn src.streaming.server:app --host 0.0.0.0 --port 8765

Browser frontend: realtime_demo/index.html (uses MediaRecorder + WSS).
"""
from __future__ import annotations

import base64
import json
import time
from pathlib import Path

import numpy as np

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles
    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def create_app():
    """Create the FastAPI app. Lazy import keeps fastapi optional for tests."""
    if not _HAS_FASTAPI:
        raise RuntimeError("fastapi not installed — pip install fastapi uvicorn[standard]")

    from src.streaming.reference_cache import ReferenceCache
    from src.streaming.session import StreamingSession
    from src.ssl_extractor import SSLExtractor

    app = FastAPI(title="ATD streaming demo")

    # Lazy-init shared state.
    cache = ReferenceCache(
        manifest_path=PROJECT_ROOT / "data/processed/references/reference_manifest.json",
        project_root=PROJECT_ROOT,
        layer=9,
    )
    extractor = SSLExtractor(model_name="wavlm_large", device="auto", use_fp16=True)

    @app.get("/health")
    def health():
        return {"status": "ok", "device": extractor.device, "cache_words": len(cache._manifest.get("word_layer_ref", {}))}

    @app.websocket("/ws/atd")
    async def ws_atd(ws: WebSocket):
        await ws.accept()
        session: StreamingSession | None = None
        try:
            while True:
                msg_text = await ws.receive_text()
                msg = json.loads(msg_text)
                t = msg.get("type")
                if t == "session_start":
                    word = msg["word"]
                    if not cache.has(word):
                        await ws.send_json({"type": "error", "code": "NO_REFERENCE",
                                            "msg": f"word {word!r} not in reference cache"})
                        continue
                    session = StreamingSession(word=word, cache=cache, extractor=extractor)
                elif t == "audio_chunk" and session is not None:
                    pcm16 = np.frombuffer(base64.b64decode(msg["pcm16_b64"]), dtype=np.int16)
                    audio = pcm16.astype(np.float32) / 32768.0
                    scores = session.consume_chunk(audio)
                    if scores:
                        await ws.send_json({
                            "type": "atd_update",
                            "seq": msg.get("seq", -1),
                            "frame_idx": session.last_frame_idx,
                            "atd": float(scores[-1]),
                            "ci_low": None,
                            "ci_high": None,
                            "path_tip": [int(session.dtw.band_center), session.dtw.t],
                            "server_ts": int(time.time() * 1000),
                        })
                elif t == "session_end" and session is not None:
                    await ws.send_json({
                        "type": "session_end",
                        "final_atd": float(session.dtw.last_atd) if session.dtw else float("nan"),
                        "n_frames": session.last_frame_idx + 1,
                        "latency_p50_ms": session.latency_ms(),
                    })
                    session = None
        except WebSocketDisconnect:
            return

    # Serve the demo frontend.
    demo_dir = PROJECT_ROOT / "realtime_demo"
    if demo_dir.exists():
        app.mount("/demo", StaticFiles(directory=str(demo_dir), html=True), name="demo")

    return app


# Convenience module-level instance for `uvicorn src.streaming.server:app`.
if _HAS_FASTAPI:
    app = create_app()
