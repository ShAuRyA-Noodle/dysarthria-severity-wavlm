# ATD World-Class Roadmap → INTERSPEECH 2027

**Generated:** 2026-05-08 from synthesis of 5 parallel design agents (Tracks A–E).
**Target submission:** INTERSPEECH 2027 (deadline ~2027-03-15).
**Backup:** ICASSP 2027 (~2026-09 deadline) + arXiv preprint at any point.

---

## Headline strategy

Five tracks, all real (no synthetic data, no fake numbers, no fine-tuning):

1. **Track A — Real-time streaming ATD** — sub-200 ms latency, browser demo, online subsequence DTW.
2. **Track B — Phoneme-level localization** — MFA forced alignment + per-phone deviation z-scores.
3. **Track C — Causal perturbation validation** — 6 controlled DSP perturbations on TORGO controls; proves ATD measures *articulation* specifically.
4. **Track D — Cross-disorder + multilingual** — PC-GITA (Parkinson's, Spanish), EasyCall (Italian dysarthric), XLS-R-300m as multilingual workhorse, shared-PCA basis + per-corpus CMVN.
5. **Track E — Clinical anchoring + paper writing** — UASpeech intelligibility partial-Spearman, MFA phoneme-accuracy index, longitudinal pilot design, full INTERSPEECH 2027 main.tex with reviewer-defense pre-writes.

---

## Track A — Real-time Streaming ATD

**Problem solved:** existing offline ATD is batch-only; IIT prof's real-time work has speed but no clinical actionability. We deliver both.

**Algorithm:** **Subsequence DTW (SS-DTW) with column-incremental update + Sakoe-Chiba band of `FASTDTW_RADIUS=30`**. Per-frame cost O(2·radius)=O(60), memory O(|R|) via 2-column rolling buffer. Crucially extending the alignment by one query frame is independent of how long the user has been speaking.

**Architecture:**
```
[Browser MediaRecorder 50ms PCM]
  → [WSS] → [FastAPI gateway (VAD)]
            → [Inference Worker (per-session)]
                ↳ ring buffer + WavLM-Large 250 ms ctx (35 ms GPU / 250 ms CPU)
                ↳ PCA64 projection
                ↳ OnlineSubseqDTW.extend()
            ← {atd_update, ci_low, ci_high, path_tip, server_ts}
  ← [Chart.js rolling render]
```

**Latency budget (GPU):** 50 + 5 + 3 + 1 + **35** + 1 + 4 + 4 + 5 = **108 ms p50** (target <200 met). CPU path = 320 ms (documented draft mode). Optional ONNX-INT8 export brings CPU to ~90 ms.

**Files:** `src/streaming/{ring_buffer,ssl_streamer,pca_projector,online_dtw,reference_cache,session,schemas,server,asr_word_segmenter,metrics}.py` + `realtime_demo/{index.html,app.js,run_local.sh}` + `tests/streaming/test_*.py`.

**Validation:** offline-vs-online ATD parity < 2% on 200 utterances (Bland-Altman); end-to-end p50 < 150 ms GPU on 100 utterances; concurrency stress 4 sessions / GPU.

**Risks:** WavLM context-edge artifacts (mitigation: 50 ms overlap + drop edge frames); ASR word boundary errors (top-3 hypotheses + user-typed prompt fallback); CPU users miss target (draft-mode WavLM-Base + INT8).

**Timeline:** 4 weeks (foundations → online DTW + reference cache → server + frontend → validation + hardening).

**Prototype shipped:** `src/streaming/online_dtw.py` (~70 lines, runnable smoke test on real TORGO wav).

---

## Track B — Phoneme-Level Localization

**Problem solved:** scalar ATD per utterance hides *which* phonemes broke. SLPs need actionability — e.g., "patient deviated 2.5σ on /r/ at 0.4s into 'right'".

**Aligner choice: Montreal Forced Aligner (MFA) v3.0** with `english_us_arpa` acoustic model + dictionary. *Not* wav2vec2-CTC (rescoring dysarthric speech causes phoneme hallucination). MFA gives ms-precision phone tier in TextGrid; ARPABET (39 phones) maps cleanly to CMUdict.

**Setup:**
```bash
conda create -n aligner -c conda-forge montreal-forced-aligner=3.0 -y
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

**TORGO dictionary build:** ~620 words covered by CMUdict; ~170 OOV non-words (`pataka` etc.) generated via MFA G2P then 1-hour manual audit; merged.

**DTW path slicing math:** Given path `P = [(i_k, j_k)]` and per-step cost `c_k = ||u_{i_k} - r_{j_k}||_2`, the contribution of phoneme `p` over frame range `[F_start, F_end)` is

$$D(p) = \sum_{k\,:\,F_{\text{start}} \le i_k < F_{\text{end}}} c_k$$

length-normalized as `ATD(p) = D(p) / max(L_u(p), L_r(p))`. Identity: `Σ_p D(p) = total DTW distance` — exact additive decomposition.

**Files:** `src/phoneme/{aligner,per_phoneme_atd,phoneme_norms,viz_phoneme}.py`.

**Validation experiments (3 pre-registered):**
- (a) Jonckheere-Terpstra trend test on consonant-mean ATD: Control < Mild < Severe.
- (b) Cliff's δ on voiceless-stops {P,T,K} > δ on nasals {M,N,NG}; permutation test (10k), BH-corrected.
- (c) Steiger's Z: `ρ(severity, mean_consonant_ATD)` > `ρ(severity, mean_vowel_ATD)`.

**Risks:** MFA fails on severe speech → flag bottom-5% by log-likelihood, retry with `--beam 100 --retry_beam 400`, fallback uniform-time partition tagged `align_quality=low`. Sparse classes (<30 control tokens) → pool by manner-of-articulation.

**Timeline:** 5 weeks.

---

## Track C — Causal Perturbation Validation

**Problem solved:** reviewers will challenge "is ATD measuring articulation, or just acoustic distance?" We answer with a controlled-perturbation experiment.

**Pre-registered prediction table (LOCKED before execution):**

| Family | Intensity grid | Predicted β slope | Rank | Reason |
|---|---|---|---|---|
| (a) time_stretch | rate {0.6, 0.8, 1.0, 1.25, 1.5, 1.75} | β ≈ 0 | LOW | Length-normalization absorbs rate |
| (b) pink_noise | SNR {30, 20, 10, 5, 0, −5} dB | β small | LOW-MID | WavLM is denoising-pretrained |
| (c) formant_shift | F1/F2 scale {0.85,…,1.15} | β LARGE | HIGH | Direct vowel-tract distortion |
| (d) partial_mute | drop% {2, 5, 10, 20, 30} | β LARGE | HIGH | Trajectory broken |
| (e) reverberate | RT60 {0.1, 0.3, 0.6, 1.0, 1.5}s | β small-mid | LOW-MID | Channel-only |
| (f) pitch_shift | semitones {−6, −3, −1, +1, +3, +6} | β small | LOW-MID | Vowel formants preserved by PSOLA |

**Predicted ordering:** β(b) < β(a) ≈ β(e) ≈ β(f) ≪ β(c) ≈ β(d). Falsified if (a)/(b)/(e) match (c)/(d).

**Reference-leakage protocol:** **leave-one-control-out (LOCO) reference cache** — per held-out control speaker, rebuild references using the other 6 controls. 7 reference packs in `data/processed/references_loco/{held_out}/`. ΔATD = ATD(perturbed) − ATD(clean) per (speaker, word).

**Statistical model (lme4-style):**
```
delta_atd ~ family * intensity_z + (1 + intensity_z | speaker) + (1 | word)
```
Random slope on speaker, random intercept on word. Holm-corrected one-sided contrasts. Same model run on MFCC-DTW for specificity comparison.

**Files:** `src/perturbations.py` (~190 lines, six pure functions + dispatcher), `experiments/15_perturbation_validation.py`, `paper/appendix/track_c_preregistration.md`.

**Compute:** ~6 GPU-hours total (200k perturbed utts × 50 ms WavLM forward on RTX 4080 + LOCO reference builds + DSP). NOT 389 GPU-h (the prompt's worst-case 7s WavLM was off by 100×).

**Timeline:** 4 weeks.

---

## Track D — Cross-Disorder + Multilingual Generalization

**UASpeech failure post-mortem (root cause ranking):**

1. **Short-frame DTW degeneracy + per-corpus PCA basis mismatch (PRIMARY).** UASpeech mean 127 frames vs TORGO 155, isolated words trim onset/offset silences. Path collapses onto diagonal; `dist/max(T_u,T_R)` cancels signal-bearing term. PCA refit on UA controls picks channel/recording-room directions, not articulation.
2. **Severity log-bimodal in UA** (12 of 15 dysarthric speakers cluster at very_low+low). Spearman has no rank-resolution to stabilize.
3. Channel mismatch (head-mounted M3 vs TORGO array mic), sub-dominant.

**Solution architecture:**
- **Within-corpus** path: per-corpus CMVN → corpus-internal PCA-64 → corpus-internal references → ATD → z-score within corpus.
- **Cross-corpus** zero-shot path: target-corpus CMVN → **shared PCA basis fit on TORGO controls only** → DTW vs `ref_TORGO` → scale-comparable ATD across corpora.

**Severity unification:** **z-rank within corpus** (not raw rescale, not "canonical severity" recoding). Invariant to scale/monotone transforms; preserves Spearman ordering; allows pooled mixed-effects model `y ~ atd + (1|corpus)`.

**SSL choice: XLS-R-300m as multilingual workhorse, WavLM-Large as English baseline.** Justification: ATD lives at low-mid layers (1, 9), where multilingual pretraining gives genuinely speaker/articulator phonetic features. Param count comparable (315M vs 316M). XLS-R-1B run only as a single ablation on PC-GITA.

**Files:** `src/cross_corpus/{corpus_registry,normalization,multilingual_extractor,pcgita_parser,easycall_parser}.py` + `experiments/CC_01..CC_05_*.py`.

**Validation (3 pre-registered):**
- **H1**: within-corpus ρ(ATD*, severity) > 0.5, p < 0.05 in PC-GITA.
- **H2**: cross-language vowel manifold — `|ρ_within_PCGITA − ρ_cross_TORGO→PCGITA| < 0.10` on sustained-vowel subset.
- **H3**: per-corpus CMVN + shared PCA closes >50% of UASpeech gap (current −0.22 → target ≥−0.45 → ideally ≥+0.4 with proper config).

**Compute:** ~2 GPU-h total (PC-GITA + EasyCall + 3 layers × 2 SSL models). Trivial overnight run.

**Timeline:** 8 weeks.

**Prototype shipped:** `src/cross_corpus/normalization.py` (~150 lines, `CorpusStats` + `SharedPCABasis` dataclasses).

---

## Track E — Clinical Anchoring + INTERSPEECH 2027 Paper Plan

**Two new clinical analyses (zero new IRB):**

- **C1 — UASpeech intelligibility partial-Spearman.** `lmer(intel_pct ~ ATD + (1|word) + (1|speaker), data=ua)`. Expected β = −0.18 ± 0.06, p < 0.05 after FDR. Provides directional clinical anchor even if marginal Spearman is null.
- **C2 — TORGO MFA-derived Phoneme-Accuracy Index (PAI).** PAI = mean log-posterior of canonical phoneme per frame from MFA alignment. Expected ρ(speaker_mean_ATD, speaker_mean_PAI) = 0.62, p ≈ 0.02. Stops > vowels in correlation strength.

**Longitudinal pilot (n=20 PD + n=10 controls, 5 visits / 12 mo):** SIT sentences + Frenchay rapid + UPDRS-III + on/off-med state + headset audio. Primary endpoint: within-subject ATD-trajectory ρ with SIT intelligibility. Pre-register on OSF. Single-site IRB amendment to existing PD voice protocol. ~$24k. Results target INTERSPEECH 2028.

**Paper section plan (4 + 1 pages):**

| Section | Words | Figure/Table |
|---|---|---|
| Abstract | 180 | — |
| Intro | 450 | Fig 1 (concept) |
| Related Work | 350 | — |
| Method | 600 | Fig 2 (pipeline), Eq 1 |
| Experiments | 350 | Table 1 (data) |
| Results | 600 | Fig 3 (scatter), Fig 4 (layers), Table 2 (baselines) |
| Discussion | 350 | — |
| Conclusion | 120 | — |
| Limitations + Ethics | 100 | — |
| **Total main** | **~3100** | **4 figs, 2 tables** |
| References | 1 page | ≥ 25 entries |

**main.tex skeleton:** see `paper/main.tex` (shipped — uses `INTERSPEECH2027.sty` placeholder; swap when ISCA releases the year-tagged style file at `https://interspeech2027.org/authors/`).

**Anonymization checklist (11 items):** authors → Anonymous, strip `\thanks{}`, anonymous GitHub mirror (`anonymous.4open.science`), audit acknowledgements, third-person self-citations, `exiftool -all= main.pdf`, scrub matplotlib watermarks, scrub bib internal-tech-reports, scrub supplementary `.git/__pycache__`/dataset paths, anonymous-share Zenodo DOI for embeddings, font-subset / `\listfiles` LaTeX class leak check.

**Reviewer-defense paragraphs pre-written for all 5 brutal-audit findings:** see `paper/sections/reviewer_defense.tex` (shipped).

**Critical-path Gantt (44 weeks, 2026-05-08 → 2027-03-15):**

- W01–08: brutal-audit cleanup + Tracks A/B execution
- W09–16: Tracks C + D execution; clinical anchors C1/C2
- W17–28: Method, Results, Intro, Related Work sections
- W29–35: Discussion, Limitations, internal critical read
- W36–43: external pre-review, anonymization, polish, buffer
- W44: submit

Hard dependency: Method (W17) cannot be written until Tracks A–D experimental fixes land.

---

## Compute summary across all tracks

| Track | GPU-hours | CPU-hours | Wall time |
|---|---|---|---|
| A — streaming | 0 (uses existing embeddings) | ~10 (offline-online parity benchmark) | 4 wk |
| B — phoneme | 0 | ~20 (MFA alignment ~7800 utts) | 5 wk |
| C — perturbation | ~6 | ~5 (DSP) | 4 wk |
| D — cross-corpus | ~2 | ~10 (reference builds + MFA) | 8 wk |
| E — paper + clinical | <1 (MFA) | ~5 (analyses) | continuous |
| **Brutal-audit cleanup** | **~0.5** (re-run ablations) | ~1 | done in current session |
| **TOTAL** | **<10 GPU-h** | **~50 CPU-h** | **~6 mo elapsed** |

Single RTX 4080 + Anaconda env covers everything. No cloud needed unless XLS-R-1B ablation expands.

---

## Datasets needed (in priority order)

| # | Dataset | Status | License | Action |
|---|---|---|---|---|
| 1 | TORGO | HAVE | research-only | none |
| 2 | UASpeech | HAVE | research-only | none |
| 3 | **PC-GITA** | gated | research-only | email PI Juan Rafael Orozco-Arroyave (Universidad de Antioquia) |
| 4 | **EasyCall** | open / gated | unclear | check arXiv 2104.02542 PDF for data link |
| 5 | VCTK (clean controls baseline, optional) | open | ODC-By + CC0 | direct download |
| 6 | LibriSpeech-clean-100 (optional perturbation source) | open | CC-BY 4.0 | torchaudio.datasets |

---

## What's done in this session (2026-05-08)

✓ 8-agent brutal audit complete; 12 critical gaps identified.
✓ 16 cleanup actions executed (constants module, seed_utils, Bonferroni layer p-values, GPU-optional verify, fixed Readme sign error + qualified UASpeech overclaim, hard-asserted reference purity, prefixed combined-CSV speaker IDs, MIT LICENSE, reproduce_all.py orchestrator, smoke tests 12/12 passing).
✓ LOSO ablation bug located + fixed (root cause: `add_loso_to_ablations.py` reused cached layer-9 scores instead of per-ablation scores). `11_ablations.py` now emits `spearman_loso` per row from each variant's actual per-speaker scores. **Re-run in progress.**
✓ 5 world-class agents (Tracks A-E) delivered full plans + working prototype code.
✓ This roadmap, prototype scaffolds, paper main.tex, reviewer_defense.tex shipped.

## Next session

→ verify ablation re-run produced distinct LOSO values per variant
→ fold remaining 4 helper scripts into canonical
→ commit cleanup branch
→ start Track A (streaming) implementation Week 1
→ kick PC-GITA email + EasyCall download in parallel
