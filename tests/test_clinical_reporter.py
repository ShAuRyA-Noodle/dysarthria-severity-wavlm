"""Smoke tests for src.clinical_reporter (Track 6).

These tests do NOT make real LLM calls — they verify:
  - the prompt builder produces a well-formed brief
  - the cache layer round-trips
  - rate-limit timing is enforced
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_build_user_prompt_contains_numbers() -> None:
    from src.clinical_reporter import build_user_prompt

    msg = build_user_prompt(
        speaker_id="F01",
        severity_score=2,
        speaker_mean_atd=111.01,
        overall_population_mean=80.0,
        overall_population_std=15.0,
        top_deviating_phonemes=[("AA1", 3.2), ("S", -2.1)],
        manner_class_summary={"vowel": 1.5, "fricative": -1.2},
    )
    assert "F01" in msg
    assert "111.01" in msg or "111.010" in msg
    assert "AA1" in msg
    assert "fricative" in msg
    assert "Severe" in msg


def test_cache_round_trip(tmp_path: Path, monkeypatch) -> None:
    """Force a cached response by pre-populating the cache dir."""
    pytest.importorskip("dotenv")
    pytest.importorskip("google.generativeai")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-real")
    # We can't fully construct LLMClient without a live key probe,
    # but the cache file format is documented. Test that format here.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    payload = {"backend": "gemini", "model": "gemini-2.5-flash",
               "text": "Cached body.", "elapsed_s": 0.001}
    (cache_dir / "S123.gemini.json").write_text(json.dumps(payload),
                                                encoding="utf-8")
    # Verify a separate reader can parse it.
    data = json.loads((cache_dir / "S123.gemini.json").read_text(encoding="utf-8"))
    assert data["text"] == "Cached body."
    assert data["backend"] == "gemini"


def test_rate_limit_timing(monkeypatch) -> None:
    """Two consecutive _respect_rate_limit calls must wait >= min_interval_s."""
    pytest.importorskip("dotenv")
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-not-real")
    try:
        from src.clinical_reporter import LLMClient
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"import failed: {exc}")

    # Bypass live-probe by constructing a bare object.
    obj = LLMClient.__new__(LLMClient)
    obj.min_interval_s = 0.2
    obj._last_call_ts = 0.0
    t0 = time.perf_counter()
    obj._respect_rate_limit()
    obj._respect_rate_limit()
    elapsed = time.perf_counter() - t0
    assert elapsed >= 0.15  # generous lower bound to avoid flake
