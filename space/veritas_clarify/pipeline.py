"""VERITAS Phase-2 — Articulatory-Bottleneck Clarifier: core pipeline (no Gradio).

Real dysarthric audio in -> clearer resynthesis out. The pipeline is:

    audio (any sr / mono-stereo)
        -> preprocess to 16 kHz mono float32
        -> SPARC acoustic-to-articulatory inversion  (encode -> 12-ch EMA @ 50 fps)
        -> OPTIONAL projection of the EMA trajectory toward the healthy-control
           articulatory manifold (src.veritas.clarify.ArticulatoryClarifier), with
           a strength in [0, 1]; pitch / loudness / speaker identity untouched
        -> SPARC HiFi-GAN decode -> clarified 16 kHz waveform
        -> EXPERIMENTAL per-frame reconstruction-confidence overlay
           (cycle-consistency; src.veritas.clarify.experimental_faithfulness_gate)

Everything is REAL: real SPARC inversion, a real closed-form articulatory
projection fit on real TORGO control statistics, real WavLM features. Weights come
from the local HF cache (offline) or download on a Space (online). Nothing mocked.

The clarifier ENCODES ONCE and reuses the SPARC code across strengths, so a given
input yields a bit-exact clarified output (see determinism note in clarify.py).

IMPORTANT — HONESTY. This is a research proof-of-concept, NOT a validated therapy
or medical device. The articulatory projection is a plausible de-centralisation of
vocal-tract motion; whether it improves human intelligibility is UNPROVEN (the
offline objective eval uses a healthy-manifold *typicality* proxy, not ASR/WER).
The per-frame reconstruction-confidence overlay is EXPERIMENTAL / UNVALIDATED —
VERITAS H1 found cycle-consistency does NOT predict true inversion error.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("veritas_clarify")

# Online (HF Space) vs offline (local cache) weight loading — mirrors veritas_mirror.
ONLINE = bool(os.environ.get("SPACE_ID") or os.environ.get("VERITAS_FORCE_ONLINE"))
if ONLINE:
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    logger.info("VERITAS clarify: ONLINE mode (Space) — weights will download if absent.")


def _find_repo_root() -> Path | None:
    cands: list[Path] = []
    env = os.environ.get("VERITAS_REPO_ROOT")
    if env:
        cands.append(Path(env))
    here = Path(__file__).resolve()
    cands.extend(here.parents)
    cands.append(Path.cwd())
    for c in cands:
        if (c / "src" / "veritas" / "clarify.py").exists():
            return c
    return None


_ROOT = _find_repo_root()
if _ROOT is not None and str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    logger.info("VERITAS clarify: repo root = %s", _ROOT)
else:
    logger.warning(
        "VERITAS clarify: could not locate src/veritas near %s; "
        "set VERITAS_REPO_ROOT or vendor the src/veritas package.", Path(__file__).resolve()
    )

import src.veritas  # noqa: E402,F401  (offline bootstrap unless ONLINE)
from src.veritas.clarify import (  # noqa: E402
    ArticulatoryClarifier,
    HealthyEMAStats,
    experimental_faithfulness_gate,
)
from src.veritas.faithfulness import HealthyReference  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.wavlm_features import WavLMFeatures  # noqa: E402

HERE = Path(__file__).resolve().parent
TARGET_SR = 16000
FPS = 50
LAYER = 9
MAX_SECONDS = 15.0
MIN_SAMPLES = TARGET_SR // 5  # 0.2 s
SILENCE_RMS = 1e-4


class AudioError(ValueError):
    """User-facing, friendly audio problem (silent / too short / unreadable)."""


_MODELS: dict[str, object] = {}


def pick_device() -> str:
    import torch

    if os.environ.get("VERITAS_FORCE_CPU"):
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def bootstrap_weights() -> None:
    """On a Space (ONLINE), pre-download WavLM-Large + SPARC. No-op locally."""
    if not ONLINE:
        return
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        snapshot_download("microsoft/wavlm-large",
                          allow_patterns=["config.json", "pytorch_model.bin", "*.json"])
        for fn in ("model_english_1500k.yaml", "model_english_1500k.ckpt"):
            hf_hub_download("cheoljun95/Speech-Articulatory-Coding", filename=fn)
    except Exception as exc:  # noqa: BLE001 — startup resilience
        logger.warning("bootstrap_weights failed (%s); loaders will retry.", exc)


def _load_healthy_stats() -> HealthyEMAStats | None:
    """Healthy-control EMA statistics = the projection target. Bundled as
    ``references/healthy_ema_stats.npz`` (built by experiments/veritas/P2_clarify_eval.py).
    Absent -> clarifier degrades to pure resynthesis (strength has no effect)."""
    p = HERE / "references" / "healthy_ema_stats.npz"
    if p.exists():
        try:
            stats = HealthyEMAStats.load(str(p))
            logger.info("loaded healthy EMA stats %s", p.name)
            return stats
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load healthy EMA stats (%s); pure resynth only.", exc)
    else:
        logger.warning("no healthy EMA stats bundled at %s; pure resynth only "
                       "(strength slider will have no effect).", p)
    return None


def _load_healthy_ref() -> HealthyReference:
    """Optional WavLM healthy centroid for the EXPERIMENTAL faithfulness overlay."""
    p = HERE / "references" / f"healthy_mu_layer{LAYER}.npy"
    if p.exists():
        try:
            return HealthyReference(np.load(p))
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to load healthy centroid (%s); uncentered.", exc)
    return HealthyReference(None)


def get_models():
    if "clarifier" not in _MODELS:
        bootstrap_weights()
        device = pick_device()
        t0 = time.perf_counter()
        logger.info("loading SPARC + WavLM-Large + healthy manifold on %s …", device)
        inverter = SPARCInverter(device=device)
        _MODELS["clarifier"] = ArticulatoryClarifier(
            inverter, healthy_stats=_load_healthy_stats(), mode="per_tract", seed=42
        )
        _MODELS["wavlm"] = WavLMFeatures(layer=LAYER, device=device)
        _MODELS["healthy_ref"] = _load_healthy_ref()
        _MODELS["device"] = device
        logger.info("models ready in %.1fs (device=%s)", time.perf_counter() - t0, device)
    return _MODELS["clarifier"], _MODELS["wavlm"], _MODELS["healthy_ref"]


def preprocess(audio, sr: int) -> np.ndarray:
    """Return a clean 16 kHz mono float32 waveform, or raise ``AudioError``."""
    if audio is None:
        raise AudioError("No audio was supplied. Record or upload a short clip and try again.")
    wav = np.asarray(audio)
    if wav.size == 0:
        raise AudioError("The audio is empty. Please record or upload a short clip.")
    if wav.ndim > 1:
        wav = wav.mean(axis=1) if wav.shape[1] <= wav.shape[0] else wav.mean(axis=0)
    wav = wav.astype(np.float32, copy=False)
    if np.issubdtype(np.asarray(audio).dtype, np.integer):
        wav = wav / 32768.0
    elif np.max(np.abs(wav)) > 1.5:
        wav = wav / 32768.0
    wav = np.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    if sr != TARGET_SR:
        try:
            import librosa

            wav = librosa.resample(wav, orig_sr=sr, target_sr=TARGET_SR)
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"Could not resample the audio from {sr} Hz ({exc}).") from exc
    wav = np.ascontiguousarray(wav, dtype=np.float32)
    if len(wav) < MIN_SAMPLES:
        raise AudioError(f"Clip is too short ({len(wav) / TARGET_SR:.2f}s). "
                         "Please provide at least ~0.2 s of speech.")
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < SILENCE_RMS:
        raise AudioError("The audio looks silent (near-zero energy). "
                         "Please speak a little louder or check your microphone.")
    peak = float(np.max(np.abs(wav)))
    if peak > 0:
        wav = wav / peak * 0.95
    return wav


def _faith_plot(faith: np.ndarray, flagged: np.ndarray, fps: int, out_png: str) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(len(faith)) / fps
    fig, ax = plt.subplots(figsize=(8, 2.8), dpi=120)
    ax.plot(t, faith, color="#6a5acd", lw=1.8)
    ax.fill_between(t, 0, faith, color="#6a5acd", alpha=0.15)
    if flagged is not None and flagged.any():
        ax.fill_between(t, 0, 1, where=flagged, color="#c0392b", alpha=0.15,
                        step="mid", label="flagged (low confidence)")
    m = float(np.mean(faith))
    ax.axhline(m, color="#c0392b", ls="--", lw=1.0, alpha=0.8, label=f"mean = {m:.2f}")
    ax.set_ylim(0, 1)
    ax.set_xlim(t[0], t[-1] if len(t) > 1 else 1)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("recon. confidence")
    ax.set_title("EXPERIMENTAL / UNVALIDATED reconstruction confidence — research "
                 "overlay, does NOT measure clarification quality", fontsize=9)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return out_png


def clarify(audio, sr: int, strength: float = 1.0, out_dir: str | None = None,
            max_seconds: float = MAX_SECONDS) -> dict:
    """Run the clarifier on one clip. Returns a result dict.

    Keys: ``ok``, ``original`` (sr, np), ``clarified`` (sr, np), ``faith_png``,
    ``summary`` (markdown), ``stats``. On failure: ``ok=False`` + ``error``.
    Only ``AudioError`` (friendly) propagates for bad input; model failures are
    caught and returned as ``ok=False`` so the app never crashes.
    """
    import tempfile

    t_start = time.perf_counter()
    out_dir_p = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="veritas_clarify_"))
    out_dir_p.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%H%M%S")
    faith_png = str(out_dir_p / f"faith_{stamp}.png")

    wav = preprocess(audio, sr)  # AudioError propagates
    # cap length for snappy interaction
    if len(wav) > int(max_seconds * TARGET_SR):
        wav = wav[: int(max_seconds * TARGET_SR)]
    dur = len(wav) / TARGET_SR
    strength = float(np.clip(strength, 0.0, 1.0))

    try:
        clarifier, wavlm, healthy_ref = get_models()

        # ENCODE ONCE, reuse for resynth (strength 0) + clarified (deterministic).
        code = clarifier.encode(wav)
        ema = np.asarray(code["ema"], dtype=np.float32)
        if ema.ndim != 2 or ema.shape[1] != 12 or ema.shape[0] < 2:
            return {"ok": False,
                    "error": "Inversion produced no usable articulatory frames for this clip."}

        clarified, meta = clarifier.clarify_with_meta(strength=strength, code=code)
        resynth = np.asarray(clarifier.inverter.decode(code), dtype=np.float32).squeeze()

        # EXPERIMENTAL faithfulness overlay: input vs its pure SPARC resynth.
        faith_path = None
        gate_stats = {}
        try:
            fx = wavlm.extract(wav, sr=TARGET_SR)
            fxh = wavlm.extract(resynth, sr=TARGET_SR)
            gate = experimental_faithfulness_gate(fx, fxh, healthy_ref)
            if len(gate.faith) >= 2:
                _faith_plot(gate.faith, gate.flagged_mask, FPS, faith_png)
                faith_path = faith_png
                gate_stats = {
                    "faith_mean": round(float(np.mean(gate.faith)), 4),
                    "frac_flagged": round(gate.frac_flagged, 4),
                    "gate_label": gate.label,
                }
        except Exception as exc:  # noqa: BLE001 — overlay is optional
            logger.warning("faithfulness overlay failed (%s); omitting it.", exc)

        stats = {
            "duration_s": round(dur, 2),
            "strength": strength,
            "projected": meta["projected"],
            "mean_abs_ema_displacement": round(meta["mean_abs_ema_displacement"], 4),
            "n_ema_frames": int(ema.shape[0]),
            "device": _MODELS.get("device", "?"),
            "elapsed_s": round(time.perf_counter() - t_start, 2),
            **gate_stats,
        }
        return {
            "ok": True,
            "original": (TARGET_SR, wav.astype(np.float32)),
            "clarified": (TARGET_SR, clarified.astype(np.float32)),
            "faith_png": faith_path,
            "summary": _summary_markdown(stats),
            "stats": stats,
        }
    except AudioError:
        raise
    except Exception as exc:  # noqa: BLE001 — never crash the app on user input
        logger.exception("clarify failed")
        return {"ok": False,
                "error": (f"Sorry — something went wrong processing this clip "
                          f"({type(exc).__name__}: {exc}). Try a different / shorter recording.")}


def _summary_markdown(stats: dict) -> str:
    proj = ("projected toward the healthy articulatory manifold"
            if stats.get("projected") else "pure SPARC resynthesis (no projection)")
    lines = [
        "### What you are hearing",
        f"The **clarified** clip is your speech run through the SPARC articulatory "
        f"bottleneck, with the vocal-tract trajectory **{proj}** at "
        f"**strength = {stats['strength']:.2f}** (0 = pure resynthesis, 1 = full "
        f"projection), then resynthesised. Your pitch, loudness and voice identity "
        f"are left unchanged.",
        "",
        f"- Duration: **{stats['duration_s']}s** ({stats['n_ema_frames']} articulatory "
        f"frames @ {FPS} fps) · device `{stats['device']}` · elapsed `{stats['elapsed_s']}s`",
        f"- Mean articulatory displacement applied: **{stats['mean_abs_ema_displacement']}** "
        f"(z-units; 0 at strength 0)",
    ]
    if "faith_mean" in stats:
        lines.append(
            f"- Reconstruction confidence (EXPERIMENTAL): mean **{stats['faith_mean']}**, "
            f"{100 * stats['frac_flagged']:.0f}% of frames flagged low-confidence"
        )
    lines += [
        "",
        "> **Honesty.** This is a research proof-of-concept, not a therapy or medical "
        "device. Whether the projection improves human **intelligibility** is UNPROVEN — "
        "the objective eval uses a healthy-manifold *typicality* proxy (no offline ASR "
        "for WER). The confidence overlay is **EXPERIMENTAL / UNVALIDATED**: VERITAS H1 "
        "found cycle-consistency does not predict true inversion error.",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI: run the clarifier on a wav directly (the local validation / smoke path)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import argparse

    import soundfile as sf

    ap = argparse.ArgumentParser(description="Run the VERITAS clarifier on a wav.")
    ap.add_argument("wav", help="path to an input .wav")
    ap.add_argument("--strength", type=float, default=1.0, help="clarification strength [0,1]")
    ap.add_argument("--out", default=None, help="output .wav path for the clarified audio")
    ap.add_argument("--max-seconds", type=float, default=MAX_SECONDS)
    args = ap.parse_args()

    data, in_sr = sf.read(args.wav, dtype="float32", always_2d=False)
    res = clarify(data, in_sr, strength=args.strength, max_seconds=args.max_seconds)
    if not res["ok"]:
        print("FAILED:", res["error"])
        sys.exit(1)
    sr, w = res["clarified"]
    out_path = args.out or str(Path(args.wav).with_name(Path(args.wav).stem + "_clarified.wav"))
    sf.write(out_path, w, sr, subtype="PCM_16")
    info = sf.info(out_path)
    print("OK")
    print("clarified wav:", out_path)
    print(f"  samplerate={info.samplerate} frames={info.frames} "
          f"duration={info.duration:.3f}s subtype={info.subtype}")
    print("faith overlay:", res["faith_png"])
    print("stats:", res["stats"])
