"""WebSocket message schemas for streaming ATD (Track A).

Pydantic models — FastAPI auto-validates on receipt. See WORLD_CLASS_ROADMAP.md
section "Track A — WebSocket Schema".
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SessionStart(BaseModel):
    type: Literal["session_start"] = "session_start"
    word: str = Field(..., description="Word/prompt being uttered (drives reference selection)")
    sample_rate: int = 16000
    client_ts: int = Field(..., description="Client epoch ms at session-start")


class AudioChunk(BaseModel):
    type: Literal["audio_chunk"] = "audio_chunk"
    seq: int
    pcm16_b64: str = Field(..., description="Base64-encoded PCM16 samples (~50 ms)")
    client_ts: int


class ATDUpdate(BaseModel):
    type: Literal["atd_update"] = "atd_update"
    seq: int
    frame_idx: int
    atd: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    path_tip: tuple[int, int] = Field(..., description="(reference_row, query_col)")
    server_ts: int


class WordEvent(BaseModel):
    type: Literal["word_event"] = "word_event"
    word: str
    start_frame: int
    end_frame: int
    conf: float


class SessionEnd(BaseModel):
    type: Literal["session_end"] = "session_end"
    final_atd: float
    n_frames: int
    latency_p50_ms: float


class StreamError(BaseModel):
    type: Literal["error"] = "error"
    code: str
    msg: str
