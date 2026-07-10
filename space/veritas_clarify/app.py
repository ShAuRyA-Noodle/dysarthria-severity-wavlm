"""VERITAS — Articulatory-Bottleneck Clarifier (Gradio app).

Upload or record dysarthric speech and hear a *clearer* resynthesis: your audio is
run through SPARC acoustic-to-articulatory inversion, its vocal-tract trajectory is
projected toward the healthy-control articulatory manifold (strength slider), and
resynthesised — same voice, same pitch, de-centralised articulation. An A/B player
compares original vs clarified, with an EXPERIMENTAL reconstruction-confidence overlay.

All heavy lifting is in ``pipeline.py`` (no Gradio there, so it runs on a wav via
CLI). This file is the UI + a defensive handler that never crashes on user input.
On a HF Space (``SPACE_ID`` set) the pipeline downloads weights on first use;
locally it loads them from the offline HF cache.
"""
from __future__ import annotations

import logging

import gradio as gr

import pipeline
from pipeline import AudioError, clarify

logger = logging.getLogger("veritas_clarify.app")

TITLE = "VERITAS — Articulatory-Bottleneck Clarifier"

INTRO = """
# VERITAS — Articulatory-Bottleneck Clarifier
**Dysarthric → clearer speech, through the vocal tract.** Your audio is run through
**SPARC acoustic-to-articulatory inversion** to recover the motion of the tongue,
lips and jaw. Dysarthria tends to *centralise and shrink* that motion; this demo
optionally projects the articulatory trajectory back toward a **healthy-control
manifold** (estimated from TORGO controls) and **resynthesises** it — keeping your
own pitch, loudness and voice identity. Use the **strength** slider to blend from
pure resynthesis (0) to full projection (1), then A/B the result.

> ⚠️ **Research proof-of-concept — not a therapy or medical device.** Whether this
> improves human **intelligibility is UNPROVEN**: the objective evaluation uses a
> healthy-manifold *typicality* proxy, not ASR word-error-rate (no offline ASR was
> available). Nothing here is a diagnosis, treatment, or a substitute for a
> Speech-Language Pathologist.
"""

HOWTO = """
**How to use.** Record or upload 1–8 s of speech, set the **clarification strength**,
and press **Clarify**. First run loads the models (a few seconds on GPU). Long clips
are capped to the first 15 s. Compare the two players; the clarified clip is the same
speaker with articulation nudged toward the healthy manifold.
"""

GATE_WARN = """
### ⚠️ Experimental reconstruction-confidence overlay — UNVALIDATED
The trace below shows per-frame **cycle-consistency** (how faithfully the invertible
articulatory model reproduced *your input* after re-synthesis, in WavLM feature
space). It is a **research overlay only**: the VERITAS **H1** experiment found that
cycle-consistency does **NOT** predict true articulatory inversion error against
measured TORGO EMA (energy-controlled partial correlation ≈ 0). **Do not** read it as
a measure of clarification quality, intelligibility, or reliability. Shaded regions
are merely the lowest-confidence frames for that clip.
"""

FOOTER = """
---
**Method.** Frozen WavLM-Large → SPARC linear articulatory inversion (12 EMA tract
variables @ 50 fps) → closed-form projection of the trajectory toward healthy-control
EMA statistics (per-tract moment matching; all parameters estimated from TORGO
controls, no hand-tuned constants) → SPARC HiFi-GAN resynthesis. Pitch, loudness and
speaker embedding are untouched (clarification, not voice conversion).

**Honest evaluation.** On real TORGO dysarthric audio we measure a **healthy-manifold
typicality** proxy (WavLM-L9 out-of-distribution distance to control frames), with a
strength-0 resynthesis control and a content-preservation guard, reported per severity
level. This is a *typicality* proxy, **not** intelligibility/WER — no ASR model was
available offline. See ``results/veritas/p2_clarify_eval.json``.

Part of the VERITAS project. Research use only.
"""


def run(audio, strength):
    """Gradio handler → (original_audio, clarified_audio, faith_image, summary_md)."""
    if audio is None:
        return None, None, None, "⚠️ **No audio.** Please record or upload a clip first."
    try:
        sr, wave = audio  # gr.Audio(type="numpy") -> (sample_rate, np.ndarray)
    except Exception:
        return None, None, None, "⚠️ **Could not read the audio input.** Try re-recording."
    try:
        result = clarify(wave, sr, strength=float(strength))
    except AudioError as e:
        return None, None, None, f"⚠️ **{e}**"
    except Exception as e:  # noqa: BLE001 — last-resort guard
        logger.exception("handler failed")
        return None, None, None, (
            f"⚠️ **Sorry, that clip could not be processed** ({type(e).__name__}). "
            "Please try a different or shorter recording.")
    if not result.get("ok"):
        return None, None, None, f"⚠️ **{result.get('error', 'Processing failed.')}**"
    return (result["original"], result["clarified"], result.get("faith_png"),
            result["summary"])


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(HOWTO)
                audio_in = gr.Audio(
                    label="Your speech (record or upload)",
                    sources=["microphone", "upload"], type="numpy",
                )
                strength = gr.Slider(
                    minimum=0.0, maximum=1.0, value=1.0, step=0.05,
                    label="Clarification strength (0 = pure resynthesis, 1 = full projection)",
                )
                run_btn = gr.Button("Clarify", variant="primary")
                gr.Examples(
                    examples=[
                        ["examples/torgo_dysarthric_M03.wav", 1.0],
                        ["examples/torgo_dysarthric_M04.wav", 1.0],
                        ["examples/torgo_control_MC02.wav", 0.5],
                    ],
                    inputs=[audio_in, strength],
                    label="Example clips (TORGO: two dysarthric speakers and a control)",
                    cache_examples=False,
                )
            with gr.Column(scale=2):
                gr.Markdown("### A / B — listen and compare")
                original_out = gr.Audio(label="Original (your input, 16 kHz)", type="numpy")
                clarified_out = gr.Audio(label="Clarified (articulatory projection)", type="numpy")
                summary_out = gr.Markdown()
                gr.Markdown(GATE_WARN)
                faith_out = gr.Image(
                    label="EXPERIMENTAL / UNVALIDATED reconstruction confidence (research overlay)",
                    type="filepath",
                )
        gr.Markdown(FOOTER)

        run_btn.click(
            fn=run, inputs=[audio_in, strength],
            outputs=[original_out, clarified_out, faith_out, summary_out],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    try:
        pipeline.get_models()  # warm the models so the first click is fast
    except Exception:  # noqa: BLE001
        logger.warning("model warm-up deferred to first request.")
    demo.queue(max_size=8).launch()
