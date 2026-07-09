"""VERITAS — Honest Articulatory Mirror (Gradio app).

Record or upload speech and watch your own vocal tract move: a midsagittal
articulator animation driven by SPARC acoustic-to-articulatory inversion, plus
an EXPERIMENTAL per-frame reconstruction-confidence trace.

All heavy lifting lives in ``pipeline.py`` (no Gradio there, so it can be
unit-run on a wav). This file is only the UI + a defensive handler that never
crashes on user input.

Deployment notes are in README.md. On a HF Space (``SPACE_ID`` set) the pipeline
switches to ONLINE mode and downloads WavLM-Large + SPARC weights on first use;
locally it loads them from the offline HF cache.
"""
from __future__ import annotations

import logging

import gradio as gr

import pipeline
from pipeline import AudioError, analyze

logger = logging.getLogger("veritas_mirror.app")

TITLE = "VERITAS — Honest Articulatory Mirror"

INTRO = """
# VERITAS — Honest Articulatory Mirror

**Record or upload a short clip of speech and *see your own vocal tract move.***
Your audio is run through **SPARC acoustic-to-articulatory inversion** to
recover the motion of six articulators — tongue tip / body / dorsum, the two
lips, and the jaw — which are animated on a midsagittal (side-view) head, one
frame per 20 ms, playing back in real time with your audio.

Below the animation, an **experimental reconstruction-confidence** trace shows
how faithfully an invertible articulatory model could reproduce your speech
(cycle-consistency in a self-supervised speech-model feature space).

> ⚠️ **Research demo — not a medical device.** The articulator motion is a
> *model estimate*, not a measurement of your anatomy, and the confidence trace
> is **experimental and not clinically validated**. Nothing here is a diagnosis
> or a substitute for a Speech-Language Pathologist.
"""

HOWTO = """
**How to use.** Record 1–6 seconds of speech (a word or short sentence works
well), or upload a `.wav`, then press **Reveal my vocal tract**. First run loads
the models (a few seconds on GPU; longer on a free CPU Space). Long clips are
capped to the first 15 s to keep playback real-time.
"""

FOOTER = """
---
**What the picture means.** The three red points are the tongue (tip, body,
dorsum) joined by a smooth contour; blue points are the upper and lower lips;
the green point is the jaw (lower incisor). The gray arch is the palate; the
tongue rising toward it corresponds to high / constricted sounds. Positions are
z-normalised model estimates placed at anatomical home locations — a faithful
picture of the inferred *motion*, not calibrated millimetres.

**Method.** Frozen WavLM-Large → SPARC linear articulatory inversion (12 EMA
tract variables @ 50 fps) → matplotlib animation. Confidence = per-frame cosine
cycle-consistency between WavLM features of your audio and of its
articulatory re-synthesis, centered on a healthy-speaker manifold.

Part of the VERITAS project. Research use only.
"""


def reveal(audio):
    """Gradio handler. Returns (video, faithfulness_image, summary_markdown).

    Never raises: friendly messages for bad input, caught errors otherwise.
    """
    if audio is None:
        return None, None, "⚠️ **No audio.** Please record or upload a clip first."
    try:
        sr, wave = audio  # gr.Audio(type="numpy") -> (sample_rate, np.ndarray)
    except Exception:
        return None, None, "⚠️ **Could not read the audio input.** Try re-recording."

    try:
        result = analyze(wave, sr)
    except AudioError as e:
        return None, None, f"⚠️ **{e}**"
    except Exception as e:  # noqa: BLE001 — last-resort guard
        logger.exception("handler failed")
        return None, None, (
            f"⚠️ **Sorry, that clip could not be processed** "
            f"({type(e).__name__}). Please try a different or shorter recording."
        )

    if not result.get("ok"):
        return None, None, f"⚠️ **{result.get('error', 'Processing failed.')}**"
    return result["video_path"], result.get("faith_plot_path"), result["summary"]


def build_demo() -> gr.Blocks:
    with gr.Blocks(title=TITLE, theme=gr.themes.Soft()) as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(HOWTO)
                audio_in = gr.Audio(
                    label="Your speech (record or upload)",
                    sources=["microphone", "upload"],
                    type="numpy",
                )
                run_btn = gr.Button("Reveal my vocal tract", variant="primary")
                gr.Examples(
                    examples=[
                        ["examples/torgo_control_MC02.wav"],
                        ["examples/torgo_dysarthric_M03.wav"],
                        ["examples/torgo_dysarthric_M04.wav"],
                    ],
                    inputs=[audio_in],
                    label="Example clips (TORGO corpus: a healthy control and "
                          "two dysarthric speakers)",
                    cache_examples=False,
                )
            with gr.Column(scale=2):
                video_out = gr.Video(
                    label="Your vocal tract in motion (plays with your audio)",
                    autoplay=True,
                )
                faith_out = gr.Image(
                    label="Experimental reconstruction-confidence (not clinically validated)",
                    type="filepath",
                )
                summary_out = gr.Markdown()
        gr.Markdown(FOOTER)

        run_btn.click(
            fn=reveal,
            inputs=[audio_in],
            outputs=[video_out, faith_out, summary_out],
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    # Warm the models at startup so the first user click is fast (and, on a
    # Space, downloads happen during boot). Non-fatal if it fails.
    try:
        pipeline.get_models()
    except Exception:  # noqa: BLE001
        logger.warning("model warm-up deferred to first request.")
    demo.queue(max_size=8).launch()
