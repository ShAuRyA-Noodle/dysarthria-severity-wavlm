"""LLM-driven clinical report generator (Track 6).

Takes per-speaker ATD scores, top deviating phonemes, and severity
context, then calls a chat-completion API (Groq or Gemini) to produce
a structured SLP-readable report.

Hard requirements:
  - Real LLM call, real input numbers, real output. No fabrication.
  - Rate-limited to respect free-tier quotas:
      Groq free tier ~30 RPM     -> 1 call every 4 seconds
      Gemini free tier ~15 RPM   -> 1 call every 5 seconds
  - Retry with exponential backoff on 429 / 503.
  - Skip rather than fabricate if API errors.
  - Cache prior reports to disk; re-run is idempotent.

Keys are read from `.env` at project root via python-dotenv (file is
gitignored). Never commit secrets.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)

logger = logging.getLogger(__name__)


# --- Rate-limit-aware client -------------------------------------------------

GEMINI_MODEL_CHAIN: list[str] = [
    "gemini-3-flash",         # user's preference; try first
    "gemini-2.5-flash",
    "gemini-2.0-flash-exp",
    "gemini-1.5-flash",
]

GROQ_MODEL_CHAIN: list[str] = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]


@dataclass
class LLMResponse:
    backend: str
    model: str
    text: str
    elapsed_s: float
    cached: bool = False


class LLMClient:
    """Tiny wrapper over Groq + Gemini with shared rate-limit budget."""

    def __init__(
        self,
        backend: str = "gemini",
        min_interval_s: float = 5.0,
        max_retries: int = 4,
        cache_dir: Path | None = None,
    ) -> None:
        self.backend = backend
        self.min_interval_s = float(min_interval_s)
        self.max_retries = int(max_retries)
        self.cache_dir = Path(cache_dir) if cache_dir else PROJECT_ROOT / "results/clinical_reports/.cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call_ts: float = 0.0
        self._chosen_model: Optional[str] = None

        if backend == "gemini":
            self._init_gemini()
        elif backend == "groq":
            self._init_groq()
        else:
            raise ValueError(f"unknown backend {backend!r}")

    # --- backend initialisers ------------------------------------------------

    def _init_gemini(self) -> None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set in environment / .env")
        import google.generativeai as genai  # type: ignore

        genai.configure(api_key=key)
        self._gemini = genai

        # Probe model chain at construction time.
        for m in GEMINI_MODEL_CHAIN:
            try:
                model = genai.GenerativeModel(m)
                # Light probe — count tokens on a 1-word prompt is free.
                _ = model.count_tokens("hi")
                self._chosen_model = m
                logger.info("gemini: using model %s", m)
                return
            except Exception as exc:  # noqa: BLE001
                logger.warning("gemini probe failed for %s: %s", m, exc)
        raise RuntimeError("no working Gemini model in chain")

    def _init_groq(self) -> None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set in environment / .env")
        from groq import Groq  # type: ignore

        self._groq = Groq(api_key=key)
        # Lazy model choice — first call tries head of chain.
        self._chosen_model = GROQ_MODEL_CHAIN[0]
        logger.info("groq: starting with model %s", self._chosen_model)

    # --- public API ----------------------------------------------------------

    def chat(self, system: str, user: str, *, cache_key: str | None = None) -> LLMResponse:
        """Send (system, user) → text. Rate-limited + cached + retried."""
        if cache_key:
            cache_path = self.cache_dir / f"{cache_key}.{self.backend}.json"
            if cache_path.exists():
                payload = json.loads(cache_path.read_text(encoding="utf-8"))
                return LLMResponse(
                    backend=payload["backend"], model=payload["model"],
                    text=payload["text"], elapsed_s=payload["elapsed_s"], cached=True,
                )

        self._respect_rate_limit()
        started = time.perf_counter()
        if self.backend == "gemini":
            text, model = self._chat_gemini(system, user)
        else:
            text, model = self._chat_groq(system, user)
        elapsed = time.perf_counter() - started

        if cache_key:
            cache_path = self.cache_dir / f"{cache_key}.{self.backend}.json"
            cache_path.write_text(json.dumps({
                "backend": self.backend, "model": model,
                "text": text, "elapsed_s": elapsed,
            }, indent=2), encoding="utf-8")
        return LLMResponse(backend=self.backend, model=model, text=text, elapsed_s=elapsed)

    # --- rate limiting -------------------------------------------------------

    def _respect_rate_limit(self) -> None:
        now = time.perf_counter()
        gap = now - self._last_call_ts
        if gap < self.min_interval_s:
            sleep_for = self.min_interval_s - gap
            time.sleep(sleep_for)
        self._last_call_ts = time.perf_counter()

    # --- per-backend chat ----------------------------------------------------

    def _chat_gemini(self, system: str, user: str) -> tuple[str, str]:
        last_exc: Exception | None = None
        attempt = 0
        while attempt <= self.max_retries:
            try:
                model = self._gemini.GenerativeModel(
                    self._chosen_model,
                    system_instruction=system,
                )
                resp = model.generate_content(
                    user,
                    generation_config=self._gemini.types.GenerationConfig(
                        temperature=0.2, max_output_tokens=1024,
                    ) if hasattr(self._gemini, "types") else None,
                )
                return resp.text, self._chosen_model
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                last_exc = exc
                logger.warning("gemini chat fail attempt %d: %s", attempt, msg[:200])
                if "429" in msg or "ResourceExhausted" in msg:
                    backoff = min(60.0, 5.0 * (2 ** attempt))
                    time.sleep(backoff)
                elif "404" in msg or "not found" in msg.lower():
                    # Bad model id — advance the chain.
                    idx = GEMINI_MODEL_CHAIN.index(self._chosen_model)
                    if idx + 1 < len(GEMINI_MODEL_CHAIN):
                        self._chosen_model = GEMINI_MODEL_CHAIN[idx + 1]
                        logger.info("gemini fallback to %s", self._chosen_model)
                        continue
                    raise
                else:
                    time.sleep(2.0)
                attempt += 1
        raise RuntimeError(f"gemini exhausted retries: {last_exc}")

    def _chat_groq(self, system: str, user: str) -> tuple[str, str]:
        last_exc: Exception | None = None
        attempt = 0
        while attempt <= self.max_retries:
            try:
                resp = self._groq.chat.completions.create(
                    model=self._chosen_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                    max_tokens=1024,
                )
                return resp.choices[0].message.content or "", self._chosen_model
            except Exception as exc:  # noqa: BLE001
                msg = str(exc)
                last_exc = exc
                logger.warning("groq chat fail attempt %d: %s", attempt, msg[:200])
                if "429" in msg or "rate_limit" in msg.lower():
                    backoff = min(60.0, 4.0 * (2 ** attempt))
                    time.sleep(backoff)
                elif "not_found" in msg.lower() or "decommissioned" in msg.lower():
                    idx = GROQ_MODEL_CHAIN.index(self._chosen_model)
                    if idx + 1 < len(GROQ_MODEL_CHAIN):
                        self._chosen_model = GROQ_MODEL_CHAIN[idx + 1]
                        logger.info("groq fallback to %s", self._chosen_model)
                        continue
                    raise
                else:
                    time.sleep(2.0)
                attempt += 1
        raise RuntimeError(f"groq exhausted retries: {last_exc}")


# --- clinical report generator ----------------------------------------------

SYSTEM_PROMPT = """You are a clinical Speech-Language Pathologist (SLP) consultant. \
Given quantitative output from a frozen-SSL dysarthric severity tool (the \
Articulatory Trajectory Deviation metric, ATD), you produce a concise, \
structured speech-pathology report.

Rules:
  - Be terse. Aim for ~150 words.
  - Use only the numbers provided. Do not invent measurements.
  - Avoid medical diagnosis; describe articulation patterns only.
  - Output sections:
        OVERALL: one-sentence summary of severity rating.
        ARTICULATORY PATTERNS: three bullet points referencing the deviating
          phonemes / manner classes from the data.
        SLP RECOMMENDATIONS: two concrete therapy-target bullets.
        CAVEAT: one sentence noting the metric is research-grade, not a diagnosis.
"""


def build_user_prompt(
    speaker_id: str,
    severity_score: int,
    speaker_mean_atd: float,
    overall_population_mean: float,
    overall_population_std: float,
    top_deviating_phonemes: list[tuple[str, float]],
    manner_class_summary: dict[str, float],
) -> str:
    """Compose the per-speaker numeric brief for the LLM."""
    sev_label = {0: "Control", 1: "Mild-Moderate", 2: "Severe"}.get(severity_score, "Unknown")
    lines = [
        f"Speaker ID: {speaker_id}",
        f"Categorical label: {sev_label} (TORGO severity_score={severity_score})",
        f"Speaker mean ATD: {speaker_mean_atd:.3f}",
        f"Population mean ATD: {overall_population_mean:.3f}",
        f"Population stdev ATD: {overall_population_std:.3f}",
        f"Speaker z-score vs population: {(speaker_mean_atd - overall_population_mean) / max(overall_population_std, 1e-6):+.2f}",
        "",
        "Top deviating phonemes (ARPABET phone, z-score vs control distribution):",
    ]
    for phone, z in top_deviating_phonemes[:8]:
        lines.append(f"  - /{phone}/  z = {z:+.2f}")
    lines.append("")
    lines.append("Per-manner-class mean z-score:")
    for cls, z in manner_class_summary.items():
        lines.append(f"  - {cls}: {z:+.2f}")
    return "\n".join(lines)
