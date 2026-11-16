"""VERITAS Phase-1 — Honest Articulatory Mirror: core pipeline (no Gradio).

Real audio in -> your vocal tract out. The pipeline is:

    audio (any sr / mono-stereo)
        -> preprocess to 16 kHz mono float32
        -> SPARC acoustic-to-articulatory inversion  (encode -> 12-ch EMA @ 50fps)
        -> render midsagittal articulator MP4 (src.veritas.render), audio muxed in
        -> GT-free per-frame cycle-consistency "reconstruction confidence"
           (encode->decode->re-encode in WavLM space; src.veritas.faithfulness)
        -> faithfulness line plot + text summary

Everything is REAL: real SPARC inversion on the real waveform, real WavLM
features, weights from the local HF cache (offline) or downloaded on a Space
(online). Nothing mocked.

This module deliberately has NO Gradio dependency so it can be unit-run on a
wav directly. ``app.py`` wraps ``analyze`` in a Gradio UI.

IMPORTANT: the per-frame "reconstruction confidence" is EXPERIMENTAL. It
reflects how well the invertible articulatory model preserved the input after
re-synthesis; it is NOT a validated clinical measure of anything.
"""
from __future__ import annotations

import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("veritas_mirror")

# ---------------------------------------------------------------------------
# Online vs offline weight loading.
#
# Local dev machine: cannot reach huggingface.co, but every weight is cached.
# Importing ``src.veritas`` activates an offline bootstrap that forces the
# transformers/HF cache-only path (see src/veritas/_offline.py).
#
# HF Space: HAS internet but starts with an EMPTY cache, so we must be ONLINE
# to download the weights. We detect a Space via the ``SPACE_ID`` env var (set
# by HF) or an explicit ``VERITAS_FORCE_ONLINE`` and switch the offline env
# vars OFF *before* importing src.veritas — its bootstrap uses ``setdefault``,
# so a value we set here wins.
# ---------------------------------------------------------------------------
ONLINE = bool(os.environ.get("SPACE_ID") or os.environ.get("VERITAS_FORCE_ONLINE"))
if ONLINE:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    logger.info("VERITAS mirror: ONLINE mode (Space) — weights will download if absent.")


def _find_repo_root() -> Path | None:
    """Locate the repo root that contains ``src/veritas/inversion.py``.

    Works both in-repo (walk up from this file) and on a Space where the
    ``src/veritas`` package sits alongside the app. ``VERITAS_REPO_ROOT`` env
    var overrides the search.
    """
    cands: list[Path] = []
    env = os.environ.get("VERITAS_REPO_ROOT")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    cands.extend(here.parents)
    cands.append(Path.cwd())
    for c in cands:
        if (c / "src" / "veritas" / "inversion.py").exists():
            return c
    return None


_ROOT = _find_repo_root()
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    logger.info("VERITAS mirror: repo root = %s", _ROOT)
else:
    logger.warning(
        "VERITAS mirror: could not locate src/veritas near %s; "
        "set VERITAS_REPO_ROOT or vendor the src/veritas package.", Path(__file__).resolve()
    )

# import src.veritas FIRST (activates offline bootstrap unless we set ONLINE)
import src.veritas  # noqa: E402,F401
from src.veritas.faithfulness import HealthyReference, frame_faithfulness  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.render import render_articulators  # noqa: E402
from src.veritas.wavlm_features import WavLMFeatures  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET_SR = 16000
FPS = 50
LAYER = 9
MAX_SECONDS = 15.0          # cap render length (keeps it real-time + snappy)
MIN_SAMPLES = TARGET_SR // 5  # 0.2 s minimum
SILENCE_RMS = 1e-4


class AudioError(ValueError):
    """User-facing, friendly audio problem (silent / too short / unreadable)."""


# ---------------------------------------------------------------------------
# lazy singletons (models loaded once, on first use)
# ---------------------------------------------------------------------------
_MODELS: dict[str, object] = {}


def pick_device() -> str:
    import torch

    if os.environ.get("VERITAS_FORCE_CPU"):
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def bootstrap_weights() -> None:
    """On a Space (ONLINE), pre-download the weights into the HF cache.

    No-op locally (offline cache already populated). Resilient: a partial or
    failed pre-fetch is not fatal — the model loaders will try again.
    """
    if not ONLINE:
        return
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        logger.info("bootstrap_weights: fetching WavLM-Large + SPARC checkpoints…")
        snapshot_download(
            "microsoft/wavlm-large",
            allow_patterns=["config.json", "pytorch_model.bin", "*.json"],
        )
        for fn in ("model_english_1500k.yaml", "model_english_1500k.ckpt"):
            hf_hub_download("cheoljun95/Speech-Articulatory-Coding", filename=fn)
        logger.info("bootstrap_weights: done.")
    except Exception as exc:  # noqa: BLE001 — startup resilience
        logger.warning("bootstrap_weights failed (%s); loaders will retry.", exc)


def _load_healthy_ref() -> HealthyReference:
    """Optional healthy-manifold centroid (recovers faithfulness dynamic range).

    Bundled as ``references/healthy_mu_layer9.npy``. Absent -> uncentered
    fallback (still valid, just a narrower band)."""
    mu_path = HERE / "references" / f"healthy_mu_layer{LAYER}.npy"
    if mu_path.exists():
        try:
            mu = np.load(mu_path)
            logger.info("loaded healthy centroid %s (dim=%d)", mu_path.name, mu.shape[-1])
            return HealthyReference(mu)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load healthy centroid (%s); uncentered.", exc)
    else:
        logger.info("no healthy centroid bundled; using uncentered faithfulness.")
    return HealthyReference(None)


def get_models() -> tuple[SPARCInverter, WavLMFeatures, HealthyReference]:
    if "inverter" not in _MODELS:
        bootstrap_weights()
        device = pick_device()
        t0 = time.perf_counter()
        logger.info("loading SPARC inverter + WavLM-Large on %s …", device)
        _MODELS["inverter"] = SPARCInverter(device=device)
        _MODELS["wavlm"] = WavLMFeatures(layer=LAYER, device=device)
        _MODELS["healthy_ref"] = _load_healthy_ref()
        _MODELS["device"] = device
        logger.info("models ready in %.1fs (device=%s)", time.perf_counter() - t0, device)
    return _MODELS["inverter"], _MODELS["wavlm"], _MODELS["healthy_ref"]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# audio preprocessing (robust to mono/stereo, int/float, any sr, silence)
# ---------------------------------------------------------------------------
def preprocess(audio, sr: int) -> np.ndarray:
    """Return a clean 16 kHz mono float32 waveform, or raise ``AudioError``."""
    if audio is None:
        raise AudioError("No audio was supplied. Record or upload a short clip and try again.")
    wav = np.asarray(audio)
    if wav.size == 0:
        raise AudioError("The audio is empty. Please record or upload a short clip.")
    # stereo / multi-channel -> mono
    if wav.ndim > 1:
        wav = wav.mean(axis=1) if wav.shape[1] <= wav.shape[0] else wav.mean(axis=0)
    wav = wav.astype(np.float32, copy=False)
    # integer PCM (or int16 delivered as float) -> [-1, 1]
    if np.issubdtype(np.asarray(audio).dtype, np.integer):
        wav = wav / 32768.0
    elif np.max(np.abs(wav)) > 1.5:
        wav = wav / 32768.0
    # sanitise NaN/inf
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    # resample to 16 kHz
    if sr != TARGET_SR:
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"Could not resample the audio from {sr} Hz ({exc}).") from exc
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if len(wav) < MIN_SAMPLES:
        raise AudioError(
            f"Clip is too short ({len(wav) / TARGET_SR:.2f}s). "
            "Please provide at least ~0.2 s of speech."
        )
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < SILENCE_RMS:
        raise AudioError(
            "The audio looks silent (near-zero energy). "
            "Please speak a little louder or check your microphone."
        )
    # peak-normalise for consistent inversion behaviour
    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav = wav / peak * 0.95
    return wav


def _faithfulness_plot(faith: np.ndarray, fps: int, out_png: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(faith)) / fps
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=120)
    ax.plot(t, faith, color="#6a5acd", lw=1.8)
    ax.fill_between(t, 0, faith, color="#6a5acd", alpha=0.15)
    m = float(np.mean(faith))
    ax.axhline(m, color="#c0392b", ls="--", lw=1.0, alpha=0.8, label=f"mean = {m:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1] if len(t) > 1 else 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("recon. confidence")
    ax.set_title("EXPERIMENTAL per-frame reconstruction confidence "
                 "(not clinically validated)", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def analyze(
    audio,
    sr: int,
    out_dir: str | None = None,
    max_seconds: float = MAX_SECONDS,
) -> dict:
    """Run the full mirror on one clip. Returns a result dict.

    Keys: ``ok`` (bool), ``video_path``, ``faith_plot_path``, ``summary``
    (markdown), ``stats`` (dict), and on failure ``error`` (friendly message).

    Raises only ``AudioError`` (friendly) for bad input; model/render failures
    are caught and returned as ``ok=False`` with a message so the app never
    crashes on user input.
    """
    t_start = time.perf_counter()
    out_dir_p = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="veritas_mirror_"))
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    video_path = str(out_dir_p / f"vocal_tract_{stamp}.mp4")
    faith_png = str(out_dir_p / f"faithfulness_{stamp}.png")

    # preprocess raises AudioError (friendly) — let it propagate to the app.
    wav = preprocess(audio, sr)
    dur = len(wav) / TARGET_SR

    try:
        inverter, wavlm, healthy_ref = get_models()

        # 1) SPARC inversion -> EMA (T, 12) @ 50 fps
        code = inverter.encode(wav)
        ema = np.asarray(code["ema"], dtype=np.float32)
        if ema.ndim != 2 or ema.shape[1] != 12 or ema.shape[0] < 2:
            return {
                "ok": False,
                "error": "Inversion produced no usable articulatory frames for this clip.",
            }

        # 2) GT-free per-frame reconstruction confidence (cycle consistency)
        #    reuse `code` (avoid a second SPARC encode): decode -> re-encode WavLM
        try:
            xhat = inverter.decode(code)
            fx = wavlm.extract(wav, sr=TARGET_SR)
            fxh = wavlm.extract(np.asarray(xhat, dtype=np.float32), sr=TARGET_SR)
            faith = frame_faithfulness(fx, fxh, healthy_ref)
        except Exception as exc:  # noqa: BLE001 — confidence is optional overlay
            logger.warning("faithfulness computation failed (%s); rendering without it.", exc)
            faith = None

        # 3) render the midsagittal MP4 (audio muxed, faithfulness strip synced)
        title = "Your vocal tract — SPARC inversion"
        render_articulators(
            ema, video_path, fps=FPS, audio=wav, sr=TARGET_SR,
            faithfulness=faith, title=title, max_seconds=max_seconds,
        )

        # 4) standalone faithfulness plot + summary
        stats = {
            "duration_s": round(dur, 2),
            "n_ema_frames": int(ema.shape[0]),
            "fps": FPS,
            "device": _MODELS.get("device", "?"),
            "rendered_seconds": round(min(dur, max_seconds), 2),
        }
        if faith is not None and len(faith) >= 2:
            _faithfulness_plot(faith, FPS, faith_png)
            stats["faith_mean"] = round(float(np.mean(faith)), 4)
            stats["faith_min"] = round(float(np.min(faith)), 4)
            stats["faith_max"] = round(float(np.max(faith)), 4)
        else:
            faith_png = None

        elapsed = time.perf_counter() - t_start
        stats["elapsed_s"] = round(elapsed, 2)

        summary = _summary_markdown(stats)
        return {
            "ok": True,
            "video_path": video_path,
            "faith_plot_path": faith_png,
            "summary": summary,
            "stats": stats,
        }
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001 — never crash the app on user input
        logger.exception("analyze failed")
        return {
            "ok": False,
            "error": (
                "Sorry — something went wrong processing this clip "
                f"({type(exc).__name__}: {exc}). Try a different / shorter recording."
            ),
        }


def _summary_markdown(stats: dict) -> str:
    lines = [
        "### What you are seeing",
        "A **midsagittal (side-view) animation of your own vocal tract**, driven "
        "frame-by-frame by **SPARC acoustic-to-articulatory inversion** of your "
        "recording. The three red points are the tongue (tip / body / dorsum) with "
        "a smooth contour; blue are the lips; green is the jaw. The clip plays in "
        "real time (50 fps) with your audio.",
        "",
        f"- Duration analysed: **{stats['rendered_seconds']}s** "
        f"({stats['n_ema_frames']} articulatory frames @ {stats['fps']} fps)",
        f"- Compute device: `{stats['device']}` · elapsed `{stats['elapsed_s']}s`",
    ]
    if "faith_mean" in stats:
        lines += [
            f"- Reconstruction confidence (EXPERIMENTAL): mean **{stats['faith_mean']}** "
            f"(range {stats['faith_min']}–{stats['faith_max']})",
        ]
    lines += [
        "",
        "> **Experimental overlay.** The reconstruction-confidence trace shows how "
        "faithfully the invertible articulatory model reproduced your speech after "
        "re-synthesis (cycle-consistency in WavLM feature space). It is a research "
        "signal only — **not** a clinical or diagnostic measure, and not yet validated.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: run the core pipeline on a wav directly (the local validation path)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    ap = argparse.ArgumentParser(description="Run the VERITAS mirror on a wav.")
    ap.add_argument("wav", help="path to an input .wav")
    ap.add_argument("--out", default=None, help="output directory")
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = ap.parse_args()

    data, in_sr = sf.read(args.wav, dtype="float32", always_2d=False)
    res = analyze(data, in_sr, out_dir=args.out, max_seconds=args.max_seconds)
    if res["ok"]:
        print("OK")
        print("video:", res["video_path"])
        print("faith plot:", res["faith_plot_path"])
        print("stats:", res["stats"])
    else:
        print("FAILED:", res["error"])
        sys.exit(1)
