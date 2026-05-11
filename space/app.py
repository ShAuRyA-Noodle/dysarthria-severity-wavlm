"""Gradio frontend for ATD — zero-shot dysarthric severity demo on HF Space.

Pipeline (CPU only, free HF Space tier):
  1. User uploads or records a 16 kHz WAV (resampled automatically).
  2. User picks the target word from a dropdown of 15 demo words.
  3. WavLM-Large layer-9 frame embeddings extracted (frozen, FP32 on CPU).
  4. Projected through the per-word PCA-64 basis bundled in references/.
  5. fastdtw against the control-built reference trajectory.
  6. ATD = DTW distance / max(T_utt, T_ref) — length-normalised.
  7. Returns ATD score + interpretation banner + tiny matplotlib plot
     showing where the user trajectory sits relative to control range.

Note: this Space is a research demo. Not a clinical instrument.
"""
from __future__ import annotations

import logging
import os
import pickle
import time
from pathlib import Path

import gradio as gr
import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from transformers import AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
TARGET_SR = 16000
LAYER = 9
FASTDTW_RADIUS = 30


# --- bundled reference cache ------------------------------------------------

def _load_references() -> dict[str, dict]:
    refs: dict[str, dict] = {}
    for p in sorted((HERE / "references").glob("*_layer9.pkl")):
        word = p.stem.replace("_layer9", "")
        with open(p, "rb") as f:
            refs[word] = pickle.load(f)
    return refs


REFS = _load_references()
WORDS = sorted(REFS.keys())
logger.info("loaded %d reference words: %s", len(REFS), WORDS)


# --- WavLM (loaded once at startup) -----------------------------------------

logger.info("loading WavLM-Large (frozen, CPU)…")
_t0 = time.perf_counter()
MODEL = AutoModel.from_pretrained("microsoft/wavlm-large", output_hidden_states=True)
for p in MODEL.parameters():
    p.requires_grad = False
MODEL.eval()
logger.info("WavLM-Large ready in %.1fs", time.perf_counter() - _t0)


# --- core inference ---------------------------------------------------------

def _extract_layer9(wave: np.ndarray) -> np.ndarray:
    wav = torch.from_numpy(wave).float().unsqueeze(0)
    m, s = wav.mean(), wav.std()
    if s > 0:
        wav = (wav - m) / s
    with torch.no_grad():
        out = MODEL(wav, output_hidden_states=True)
    return out.hidden_states[LAYER].squeeze(0).cpu().numpy()


def _compute_atd(wave: np.ndarray, word: str) -> tuple[float, np.ndarray, np.ndarray]:
    ref = REFS[word]
    pca = ref["pca_model"]
    reference = ref["reference"]
    feats = _extract_layer9(wave)
    utt_pca = pca.transform(feats)
    distance, path = fastdtw(utt_pca, reference, radius=FASTDTW_RADIUS, dist=euclidean)
    atd = float(distance / max(len(utt_pca), len(reference)))
    return atd, utt_pca, reference


# --- severity interpretation ------------------------------------------------

# Control mean ATD ~62, severe mean ~100 in TORGO (from results/speaker_atd.csv).
CONTROL_BAND = (50.0, 75.0)
MILD_BAND = (75.0, 90.0)
SEVERE_BAND = (90.0, 130.0)


def _interpret(atd: float) -> tuple[str, str]:
    if atd < CONTROL_BAND[1]:
        return "Control range", "🟢"
    if atd < MILD_BAND[1]:
        return "Mild–Moderate range", "🟠"
    return "Severe range", "🔴"


# --- gradio handler ---------------------------------------------------------

def run(audio: tuple[int, np.ndarray] | None, word: str) -> tuple[str, str, plt.Figure | None]:
    if audio is None:
        return "No audio supplied.", "", None
    if word not in REFS:
        return f"Word {word!r} not in reference cache.", "", None

    sr, wave = audio
    if wave.ndim > 1:
        wave = wave.mean(axis=1)
    wave = wave.astype(np.float32)
    if wave.max() > 1.5:  # int16 came in as float
        wave = wave / 32768.0
    if sr != TARGET_SR:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
    if len(wave) < TARGET_SR // 4:
        return "Audio too short (need at least 0.25 s).", "", None

    started = time.perf_counter()
    atd, utt_pca, reference = _compute_atd(wave, word)
    elapsed = time.perf_counter() - started

    band_label, dot = _interpret(atd)
    headline = (
        f"# {dot}  ATD = **{atd:.2f}**\n"
        f"### Severity band: {band_label}\n\n"
        f"`Word: {word}`  ·  `WavLM layer {LAYER}`  ·  "
        f"`PCA-64`  ·  `n_frames_utt = {len(utt_pca)}`  ·  "
        f"`elapsed = {elapsed:.1f}s`"
    )

    detail = (
        "**How to read.** ATD measures the length-normalised dynamic-time-"
        "warping distance between your WavLM-Large layer-9 trajectory and a "
        "reference trajectory averaged from 7 healthy control speakers in "
        "the TORGO corpus. Lower = closer to typical articulatory "
        "execution. Higher = more deviated.\n\n"
        "**Bands** (empirical from TORGO):\n"
        "- 🟢 50–75 : Control range\n"
        "- 🟠 75–90 : Mild–Moderate range\n"
        "- 🔴 90+ : Severe range\n\n"
        "Bands are research-grade only; do not use as a clinical diagnosis."
    )

    # Compact viz: 2-D projection of your trajectory + reference (first 2 PCA dims).
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(reference[:, 0], reference[:, 1], "g-", linewidth=2,
            alpha=0.7, label="reference (control mean)")
    ax.scatter(reference[:, 0], reference[:, 1], c="g", s=14, alpha=0.6)
    ax.plot(utt_pca[:, 0], utt_pca[:, 1], "r-", linewidth=1.5,
            alpha=0.7, label="your utterance")
    ax.scatter(utt_pca[:, 0], utt_pca[:, 1], c="r", s=10, alpha=0.5)
    ax.set_xlabel("PCA-1"); ax.set_ylabel("PCA-2")
    ax.set_title(f"Trajectory geometry — word /{word}/")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()

    return headline, detail, fig


# --- UI ---------------------------------------------------------------------

with gr.Blocks(title="ATD — Articulatory Trajectory Deviation") as demo:
    gr.Markdown(
        "# Articulatory Trajectory Deviation (ATD) — zero-shot dysarthric severity\n"
        "Upload or record a short clip (≥ 0.25 s, 16 kHz preferred) saying "
        "**one of the words below**, then click *Compute ATD*.\n\n"
        "Pipeline: frozen WavLM-Large → PCA-64 → fastdtw vs control reference "
        "→ length-normalised distance. CPU-only on free HF Space hardware "
        "(~30 s WavLM forward; thanks for your patience).\n\n"
        "**Not a clinical instrument.** Research demo only. Code: "
        "https://github.com/ShAuRyA-Noodle/ShAuRyA_Dysar"
    )

    with gr.Row():
        audio_in = gr.Audio(label="Speak the chosen word",
                            sources=["microphone", "upload"],
                            type="numpy")
        word_in = gr.Dropdown(WORDS, label="Target word",
                              value=WORDS[0] if WORDS else None)

    run_btn = gr.Button("Compute ATD", variant="primary")
    headline_out = gr.Markdown()
    detail_out = gr.Markdown()
    plot_out = gr.Plot()

    run_btn.click(
        fn=run,
        inputs=[audio_in, word_in],
        outputs=[headline_out, detail_out, plot_out],
    )

    gr.Markdown(
        "### Reference\n"
        "Paper draft + results: see GitHub repo. Headline TORGO speaker-level "
        "Spearman $\\rho = 0.756$ ($p = 0.0011$, n=15), no fine-tuning. "
        "Cross-language transfer to Italian Parkinson's: BH-FDR-significant "
        "on all five sustained vowels at XLS-R-300m Layer 1."
    )

if __name__ == "__main__":
    demo.launch()
