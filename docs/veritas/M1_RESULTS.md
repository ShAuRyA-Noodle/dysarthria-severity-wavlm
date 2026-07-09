# VERITAS Phase 1 · Milestone 1 — Results (honest)

**Date:** 2026-07-08 · **Data:** real TORGO audio + real TORGO AG500 EMA (local) · **Seed:** 42 · Weights: frozen WavLM-Large + SPARC, offline from cache. Every number traces to `results/veritas/*.json`.

## The question
Does a **GT-free cycle-consistency "faithfulness" metric** — encode audio → SPARC articulatory code → re-synthesize `x̂` → compare `WavLM(x)` vs `WavLM(x̂)` — measure whether the acoustic-to-articulatory **inversion was reliable** at each frame? The VERITAS thesis needs "low faithfulness = the model is not faithfully representing this speech" to hold.

## Answer: NO. The metric, as formulated, does not work. (Honest negative result.)

### H2 — faithfulness vs dysarthria severity (`faithfulness_severity.json`)
Secondary sanity check. **Confounded both ways, not a valid test:**
- Uncentered cosine: ρ = **−0.12** (p=0.66) — null. Cause: WavLM embedding **anisotropy** saturates cosine (all values crushed into 0.83–0.88).
- Healthy-manifold-centered cosine: ρ = **+0.47** (p=0.08) — *wrong direction*, and **sign-flipped** vs a smaller sample. Cause: a **deviation-magnitude confound** (controls sit near the healthy centroid → tiny, noisy deviation vectors; dysarthric deviate far → larger, stabler cosine). So it measures "distance from healthy," not reconstruction fidelity.

### H1 — faithfulness vs REAL per-frame inversion error (`h1_faithfulness_vs_inversion_error.json`)  ← decisive
Correlate per-frame faithfulness with the **actual** articulatory inversion error `‖SPARC_pred − affine(measured_EMA)‖` (tongue channels) on healthy TORGO controls. Rigorous: per-speaker lag + 3D→2D affine fit on a **held-out** slice, applied to disjoint eval frames (no circularity); faithfulness never enters the alignment fit.
- **n = 5 control speakers** (FC03, MC02 skipped — >2 dead AG500 coils, objective session-validity gate), **22,854 frames, 150 utterances**.
- Raw pooled Spearman = **+0.13** (p≈1e-85) — but this is a **speech-energy artifact**: faith-vs-energy = **−0.47**, err-vs-energy = **−0.23** (silent frames are trivially faithful *and* articulatorily unconstrained).
- **Energy-controlled partial Spearman = −0.001** (~ zero). Median per-utterance Spearman = +0.07. Even on the clearest (top-25% energy) frames, only −0.09.
- **Verdict: NO** — once the silence/energy confound is removed, faithfulness carries **no** information about real inversion error.

### Why it fails (post-mortem)
Cycle-consistency in WavLM space is dominated by (1) embedding anisotropy and (2) speech energy — not by articulatory fidelity. SPARC's re-synthesis preserves gross acoustic/linguistic content regardless of how accurate the *articulatory* inversion was, so `WavLM(x) ≈ WavLM(x̂)` even when the inversion is wrong. The metric cannot see the thing it was meant to measure.

## What DID work, and ships: the Articulatory Mirror
`space/veritas_mirror/` — a real, deployable Gradio demo: speak/upload audio → **animated midsagittal vocal tract** (tongue/lips/jaw) driven by SPARC inversion, synced to the audio. Real inference on real audio, offline weights; verified it produces a genuine MP4 (`sample_output/`). This is engineering on top of others' frozen models (WavLM, SPARC), not a novel result — but it is a working, impressive artifact.
- The **faithfulness overlay in the demo is EXPERIMENTAL and UNVALIDATED** (H1 = NO). It is labeled as such and makes **no** reliability or clinical claim.

## Honest scope
- **Contribution:** a working real-time articulatory visualizer + a *rigorously executed negative result* on a novel reliability metric. Not a validated scientific contribution; not a clinical tool.
- **Ceiling:** solid solo-student engineering + honest science. Not top-lab research (SPARC/RT-VC/etc. exist; the metric didn't validate).
- **Reproducible:** seed 42, offline weights, real data, all numbers in `results/veritas/`.

## Future work (deferred — Phases 2 & 3 not built)
If revisited: reformulate faithfulness in **articulatory space** directly (not WavLM cosine); try ensemble/MC-dropout inversion uncertainty; energy-normalize before comparison; acquire more (esp. dysarthric) EMA. Phase 2 (safety-gated clarifier) and Phase 3 (benchmark) remain design-only.
