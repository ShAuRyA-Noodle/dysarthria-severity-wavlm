# When Frozen SSL Speech Metrics Don't Measure What They Claim: A Systematic Negative-Results Study on Dysarthric Speech

*Draft — Interspeech-style short/main paper. Every number below traces to a committed `results/` file; nothing is massaged. This is an honest negative-results + methodology contribution, not a new SOTA claim.*

---

## Abstract

Frozen self-supervised (SSL) speech models are increasingly used to build *training-free* clinical-speech metrics — severity scores, articulatory probes, reliability measures. We stress-test this paradigm on real dysarthric speech (TORGO, n=15, with AG500 EMA), Parkinsonian speech (IPVS, n=61), and Italian dysarthria (EasyCall, n=55), against real articulatory ground truth. We report five reproducible negative findings that recur across popular metric families: (1) frozen-WavLM severity prediction is strong **within** corpus (Spearman ρ=0.74–0.85) but **fails to generalise across corpora/etiologies** (graded transfer to TORGO ρ=0.43, *n.s.*, p=0.11; n=131 pooled); (2) reference-based severity metrics are **inflated by speaker-identity leakage** — a naive early-layer score of ρ=0.775 collapses to 0.687 under a leave-one-control-out (LOCO) reference protocol, while the mid-layer score is leakage-robust (0.756→0.756); (3) an SSL "articulatory trajectory" severity signal is **dominated by silence/pausing, not articulation** — removing silence changes the correlation structure and the raw signal is a speech-energy artifact; (4) a GT-free **cycle-consistency "faithfulness" metric does not predict real articulatory-inversion error** (energy-controlled partial Spearman = −0.001, n=5 speakers / 22,854 frames of EMA); (5) acoustic-to-articulatory **inversion is unstable on disordered input**, so inversion-derived kinematics are artifacts (EMA-grounding ρ=0.39 *n.s.*; a "significant" severity correlation of +0.52 is in the clinically *wrong* direction). We also document that **public dysarthric articulatory (EMA) ground truth is effectively n≈2** — a hard ceiling for validating pathological inversion. We release a reproducible harness, the leakage/silence/energy diagnostics, and a working real-time articulatory-visualisation demo. Our recommendation: **frozen-SSL clinical metrics must be validated with LOCO references, silence controls, cross-corpus tests, and — where possible — real articulatory ground truth**, or they report confounds rather than pathology.

---

## 1. Introduction

Frozen SSL representations (WavLM, HuBERT) encode rich phonetic/articulatory structure, motivating a wave of *training-free* clinical-speech metrics: pooled-embedding severity probes, reference-distance scores, and articulatory-inversion analyses. Their appeal is obvious — no labels, no fine-tuning, apparently interpretable geometry. This paper asks a blunt question: **do these metrics measure the disorder, or do they measure confounds that happen to correlate with it on a single corpus?**

We answer with a systematic negative-results study. Working from real dysarthric/Parkinsonian speech and — crucially — real electromagnetic articulography (EMA), we test four popular metric families and find each fails a rigorous validity check. The failures are not exotic; they are the confounds any deployment would hit: speaker identity, silence, speech energy, corpus/task mismatch, and the near-absence of pathological articulatory ground truth. We report them honestly, with the diagnostics that expose each, so the field can avoid them.

**Contributions.** (i) A reproducible, corpus-agnostic frozen-SSL severity harness with within-corpus and leave-one-corpus-out evaluation (n=131, 3 corpora, 2 languages). (ii) Five reproducible negative findings, each with a concrete diagnostic. (iii) A leave-one-control-out (LOCO) reference protocol that removes speaker-identity leakage from reference-based severity metrics. (iv) A real-time, audio-only articulatory-visualisation demo, released as a working artifact.

## 2. Data

- **TORGO** (Rudzicz et al., 2012): 15 English speakers (8 dysarthric CP/ALS + 7 control), audio + **AG500 3D EMA** (200 Hz) + Frenchay assessments. Graded severity {0,1,2}. The only corpus here with graded severity *and* measured articulation.
- **IPVS** (Italian Parkinson's Voice & Speech): 61 speakers (PD/HC). Public package carries **binary** PD/HC labels (reading-rate metrics, not UPDRS), so treated as a detection arm.
- **EasyCall** (Turrisi et al., 2021): 55 Italian speakers (31 dysarthric + 24 control). Public package carries **binary** dysarthric/control labels (per-speaker TOM severity is not distributed), so treated as a detection arm.
- **Articulatory ground truth**: TORGO AG500 EMA. Of 7 dysarthric speakers, **only 2 have usable sessions** (≤2 dead coils) — and both are *mild*. Every severe dysarthric EMA session is degenerate (3–8 dead coils). We verified no additional adult dysarthric EMA is available by direct download.

All features: frozen WavLM-Large, layer 9, mean-pooled per utterance → speaker mean, seed 42, speaker-disjoint splits throughout.

## 3. Finding 1 — Within-corpus success, cross-corpus failure

A fairly-tuned linear probe (Ridge / LinearSVR, nested CV) on frozen WavLM-L9 recovers severity **within** corpus:

| Corpus | n | ρ (LOSO) | label |
|---|---|---|---|
| TORGO | 15 | **0.74** | graded severity |
| IPVS | 61 | 0.85 | binary PD/HC |
| EasyCall | 55 | 0.84 | binary dys/HC |

But **leave-one-corpus-out** transfer collapses:

| Held-out (train on others) | ρ (Ridge / LinSVR) | sig. |
|---|---|---|
| TORGO (graded) | **0.43 / 0.34** | **n.s. (p=0.11)** |
| IPVS | 0.27 / 0.41 | — |
| EasyCall | 0.64 / 0.44 | — |

The signal that looks like "severity" within a corpus does not transfer across etiologies/languages. This is the failure mode a real deployment hits first, and it is invisible to single-corpus evaluation. *(`results/veritas/cross_corpus_severity.json`)*

## 4. Finding 2 — Speaker-identity leakage inflates reference-based metrics

Reference-based severity metrics (distance of a speaker's SSL trajectory to a healthy-control reference) leak speaker identity when the reference includes the test speaker's own data. On TORGO, a naive per-word reference gives an early-layer (Layer 1) score of ρ=**0.775**; under a strict **leave-one-control-out (LOCO)** reference it falls to **0.687** (Δ=−0.089). The mid-network layer (Layer 9) is leakage-robust (**0.756→0.756**). Early SSL layers encode speaker identity (cf. WavRx, 2024); we contribute the LOCO reference-construction fix and quantify the inflation it removes. *(`results/deleak_report_fast.json`)*

## 5. Finding 3 — The "articulatory" signal is largely silence/pausing

Decomposing the reference-distance metric by MFA phone alignment shows **~70–76% of the distance mass falls in silence / unattributed frames**, not phones. Recomputing on silence-trimmed speech changes the correlation structure; the raw per-frame association with severity is dominated by speech energy (silent frames are trivially "close"). A metric named for *articulation* is substantially measuring *pausing*. *(`results/silence_confound.json`)*

## 6. Finding 4 — Cycle-consistency "faithfulness" does not predict inversion error

A natural GT-free reliability metric: invert audio to articulation (SPARC), re-synthesise, and measure cycle-consistency in WavLM space — low consistency ought to flag unreliable inversion. Tested directly against **real EMA inversion error** on TORGO controls (n=5 usable speakers, **22,854 frames**, held-out affine alignment, no circularity): the raw pooled Spearman is +0.13 (p≈1e-85) but is a **speech-energy artifact** (faithfulness-vs-energy = −0.47, error-vs-energy = −0.23); the **energy-controlled partial Spearman = −0.001**. Cycle-consistency carries **no** information about real inversion error once silence/energy is removed. *(`results/veritas/h1_faithfulness_vs_inversion_error.json`)*

## 7. Finding 5 — Inversion is unstable on disordered input; kinematics are artifacts

Interpretable articulatory-kinematic features (tongue range, velocity) derived from SSL inversion do **not** track the real EMA features (EMA-grounding Spearman = 0.39, *n.s.*, n=7). A "significant" range-vs-severity correlation (ρ=+0.52, p=0.048) is in the clinically **wrong direction** (dysarthric > control range), consistent with the inverter producing *more erratic* trajectories on disordered input, not more movement. Combined with the **n≈2 usable dysarthric EMA** ceiling, this closes the door on validating pathological inversion with public data. *(`results/veritas/m2_articulatory_kinematics.json`, `h1_dysarthric_inversion.json`)*

## 8. A working artifact: the real-time articulatory Mirror

Independently of the (failed) metrics, frozen-SSL inversion (SPARC) supports a real, deployable demo: speak → an animated midsagittal vocal tract driven by audio-only inversion, with an explicitly-labelled **experimental** reliability overlay (Finding 4 shows the overlay is unvalidated). Released as a Gradio app. This separates what works (engineering on frozen models) from what does not (the reliability claim).

## 9. Recommendations

For anyone building frozen-SSL clinical-speech metrics: (1) **LOCO references** — never score a speaker against a reference containing its own data; prefer mid-network layers. (2) **Silence/energy controls** — report VAD-trimmed and energy-partialled results; silence is the default confound. (3) **Cross-corpus / leave-one-corpus-out** evaluation — single-corpus success is not generalisation. (4) **Ground truth where possible** — validate inversion/articulatory claims against real EMA; be explicit that pathological EMA is nearly non-existent publicly. (5) **Fair, tuned baselines** — a properly-regularised pooled-embedding probe ties trajectory-geometry methods; report it.

## 10. Conclusion

Frozen-SSL clinical-speech metrics are easy to build and easy to fool. Across four metric families and real articulatory ground truth, the apparent signal is repeatedly a confound — speaker identity, silence, energy, or corpus overfitting — and the one regime that would validate pathological articulatory inversion (dysarthric EMA) barely exists in public data. We release the harness, diagnostics, and demo so these pitfalls are checkable, not repeated. Honest negative results, rigorously obtained, are the contribution.

---

### Reproducibility
Seed 42; frozen weights; every headline number → a committed `results/veritas/*.json` or `results/*.json`; speaker-disjoint splits; harness auto-scales as corpora are added (`experiments/veritas/CX_cross_corpus_severity.py`, `src/veritas/corpora/`).

### Limitations / honest scope
n is small for graded severity (TORGO=15; IPVS/EasyCall are binary detection arms). The largest graded corpora (SAP; VOC-ALS graded ALS) are DUA-gated or pending. Results are a *cautionary study*, not a new metric; a positive cross-corpus result would require the gated large corpora and is left as the falsifiable next test.
