---
title: ATD — Articulatory Trajectory Deviation
emoji: 🗣️
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
short_description: Zero-shot dysarthric speech severity via frozen WavLM-Large
---

# Articulatory Trajectory Deviation (ATD)

Zero-shot dysarthric speech severity estimation via the geometry of a frozen
self-supervised speech model. **No fine-tuning, no labelled training data.**

## How it works

```
Raw audio (16 kHz)
      ↓
WavLM-Large (frozen, layer 9)
      ↓
Frame-level embeddings (T × 1024)
      ↓
PCA → 64 dimensions (same basis as control reference)
      ↓
fastdtw alignment to control reference trajectory R_w
      ↓
ATD = DTW distance / max(T_utt, T_ref)
```

Lower ATD = closer to typical articulatory execution. Higher ATD = more
deviated from the control manifold.

## Results

- TORGO speaker-level Spearman ρ = 0.756 (p = 0.0011, n = 15, 10k bootstrap
  CI [0.38, 0.92])
- Leave-one-speaker-out range [0.724, 0.855], every fold p < 0.005
- Cross-language transfer to Italian Parkinsonian speech (IPVS) — all five
  sustained vowels BH-FDR-significant at XLS-R-300m Layer 1
- 16.3% RMSE reduction over MFCC + SVR baseline

## Demo limitations

- 15 demo words bundled (TORGO single-word stimuli). Pipeline runs only on
  these — Space app is a proof-of-concept, not a general clinical tool.
- CPU-only inference. Each clip takes ~30 seconds for WavLM-Large forward
  pass. Hardware upgrade to GPU would drop this to ~1 second.
- ATD severity bands (50–75 Control, 75–90 Mild, 90+ Severe) are empirical
  from TORGO controls + dysarthric speakers, not clinically calibrated.

## Not a clinical instrument

This is a research demo. Outputs **must not** be used as the basis for
diagnosis, treatment decisions, or substitute for a Speech-Language
Pathologist's evaluation.

## Reproducibility

Code, paper draft, full results: https://github.com/ShAuRyA-Noodle/ShAuRyA_Dysar
