# VERITAS — Design Spec (Programme + Phase 1 detailed)

**Date:** 2026-07-07
**Status:** Design — pending user review, then implementation planning.
**One-line thesis:** VERITAS is the first pathological-speech AI built on a *faithfulness* principle — it refuses to hallucinate the disorder away. It measures its own fidelity to the real, impaired signal and surfaces that uncertainty instead of silently normalizing it toward "healthy."

This document specifies the **programme** (three phases unified by one novel core) and designs **Phase 1** in detail. Phases 2 and 3 are outlined here and receive their own full spec → plan → build cycles when reached.

---

## 0. Why this exists (the problem, honestly)

Every serious system in pathological-speech AI shares one failure mode: **hallucination toward "normal."**
- Dysarthric→clear *clarifiers* fabricate words the person never said (severe input lacks the acoustic information to recover intent → the model must invent or fail).
- Acoustic-to-articulatory *inverters* trained on healthy speech pull the impaired tongue back toward a normal trajectory (correlation drops ~0.78 healthy → ~0.42 unseen dysarthric), **masking the very deviation a clinician needs to see.**
- ASR (Whisper) hallucinates transcripts on dysarthric input.

None of these systems know *when they are being unfaithful to the disorder*. That missing self-knowledge is the white space. It is also a safety issue: a confident-but-wrong output for a disabled user is worse than an honest "I can't represent this faithfully."

VERITAS makes faithfulness a first-class, measured quantity across three settings: **SHOW** the disorder (Phase 1), **SPEAK** for it (Phase 2), **MEASURE** it for the field (Phase 3).

---

## 1. The shared core — the faithfulness metric (the breakthrough)

The single idea that unifies the trilogy and is novel on its own:

**GT-free cycle-consistency faithfulness.** Given input audio `x`:
1. Extract frozen-SSL features `E(x)` (WavLM-Large).
2. Invert to an interpretable articulatory / tract-variable representation `A = Inv(E(x))` (SPARC-style: 12–14 tract variables — tongue tip/dorsum, lips, jaw, velum, glottis).
3. Re-synthesize audio `x̂ = Synth(A)` (SPARC is invertible: articulation → speech).
4. Faithfulness `F = sim(E(x), E(x̂))` per frame — how much of the *actual* input the articulatory bottleneck preserved.

When `Inv` normalizes a disordered input toward healthy, `x̂` reverts toward typical speech, so `E(x̂)` diverges from `E(x)` → **F drops exactly where the model is being unfaithful to the disorder.** Crucially, computing `F` needs **no articulatory ground truth** — it is self-supervised on audio alone. This is what lets VERITAS operate in the pathological regime where EMA/MRI ground truth barely exists (~2 dysarthric TORGO EMA speakers worldwide).

Supporting signals (fused into a calibrated per-frame confidence):
- **OOD score**: distance of `E(x)` from the healthy training manifold (Mahalanobis / density) — dysarthric input is out-of-distribution.
- **Inversion ensemble variance / MC-dropout**: epistemic uncertainty of `Inv`.
- Fused `→ Faithfulness ∈ [0,1]`, calibrated on held-out data (reliability diagram).

**Validated claim (to be tested, not assumed):** `F` correlates with true inversion error where GT exists (healthy EMA), and drops monotonically with dysarthria severity — i.e., it detects the normalization failure it is designed to catch.

Everything else in the programme is an application of this measured faithfulness.

---

## 2. Phase 1 — "Honest Articulatory Mirror" (DETAILED)

### 2.1 What it is
A real-time, **audio-only** biofeedback system: speak into a microphone → see your own vocal tract (midsagittal animated tongue/lips/jaw/velum) move live, with a per-articulator **deviation-from-healthy** overlay and a live **faithfulness meter**. Where faithfulness is low, the mirror **greys out / says "uncertain"** rather than rendering a confident (wrong, normalized) tongue.

### 2.2 Honest novelty (what is and isn't new)
- **Not new:** audio→animated-tongue in real time (Berkeley Streaming Speech-to-Avatar, 130 ms; RT-VC causal SPARC, 61 ms CPU); the inversion backbone (SPARC, open source).
- **New (the contribution):**
  1. **Faithfulness-aware pathological inversion** — the GT-free cycle-consistency metric that detects when inversion normalizes the disorder (§1). No prior articulatory system measures this.
  2. **Uncertainty-surfacing biofeedback** — the first articulator visualizer that refuses to show a confident tongue it can't justify.
  3. **Interpretable deviation in tract-variable space** — which articulator is off, when, by how much, vs healthy norms with leakage-clean (LOCO) references (carrying the one genuine methodological win from the prior ATD work into meaningful clinical coordinates).

### 2.3 Architecture (isolated units — purpose / interface / deps)

```
[mic/file] → audio_capture → ssl_streamer → inverter ─┬─→ faithfulness_estimator ─┐
                                                       │                           ├─→ fusion → renderer (browser)
                                              deviation_engine ←── reference_bank ─┘
             server (WebSocket) orchestrates; session holds per-connection state
```

- **`audio_capture`** — mic/file → 16 kHz mono frames. *Interface:* `frames()` generator. *Deps:* browser MediaRecorder (web) / sounddevice (local). Reuses existing `src/streaming/ring_buffer.py`.
- **`ssl_streamer`** — streaming WavLM-Large frame features (causal, 250 ms context, edge-frame handling). *Interface:* `push(pcm) → features[T,1024]`. *Deps:* WavLM (HF, open), existing `src/streaming/ssl_streamer` + `session.py`. Reuse the exact-parity streaming work already done.
- **`inverter`** — frozen WavLM features → tract variables (12–14 TV) + invertible synth. *Interface:* `invert(features) → A[T,TV]`, `synth(A) → audio`. *Deps:* **SPARC** (Berkeley, open weights + pip). Adapt to causal/streaming per RT-VC.
- **`faithfulness_estimator`** — the novel core (§1): cycle-consistency + OOD + ensemble variance → calibrated `F[T]`. *Interface:* `faithfulness(x, A) → F[T]`. *Deps:* inverter.synth, WavLM, a fitted OOD density on healthy features.
- **`reference_bank`** — healthy per-phone / per-articulator norm distributions (mean, cov) for deviation scoring, built LOCO-clean from healthy EMA corpora. *Interface:* `norm(phone, TV) → (μ,Σ)`. *Deps:* HPRC/USC data (offline build).
- **`deviation_engine`** — per-articulator z-deviation of `A` vs `reference_bank`, **faithfulness-gated** (suppress/flag deviations where `F` low). Optional MFA phone alignment for per-phone attribution. *Interface:* `deviation(A, F, phones?) → D[T,TV] + flags`. *Deps:* reference_bank, (optional) aligner.
- **`renderer`** — live midsagittal vocal-tract animation (Pink-Trombone-style canvas) + deviation heatmap + faithfulness meter + "uncertain" masking. *Interface:* consumes `{A, D, F}` stream → draws. *Deps:* browser canvas/WebGL.
- **`server`** — FastAPI + WebSocket: audio in → `{A, D, F}` out. *Deps:* existing `src/streaming/server.py` scaffold.
- **`session`** — per-connection buffers/state/calibration. Reuse `src/streaming/session.py`.
- **`validation`** (offline) — the science harness (§2.6).

### 2.4 Data flow (real-time loop, target < 200 ms GPU)
`50 ms PCM chunk → ring buffer → WavLM causal forward (features) → SPARC invert (A) → [in parallel] SPARC synth→WavLM→F, and deviation vs norms → fuse → WebSocket frame {A,D,F} → canvas render`. Budget is comfortable: RT-VC proves the SSL→articulation→render loop runs causally at ~61 ms on CPU. Faithfulness adds one extra synth+WavLM pass (batchable / can run at reduced rate).

### 2.5 Data (all real; access reality stated)
- **Inversion train / healthy norms (open):** HPRC EMA (8 spk, downloadable), USC 75-speaker real-time MRI (open, ~966 GB), MNGU0 (registration), Wisconsin XRMB (community re-host). SPARC is *already trained* — we build on its weights and only fit heads/norms.
- **Pathological validation (real, tiny — stated honestly):** TORGO EMA (~2 dysarthric patients with parallel acoustic-articulatory — the only open dysarthric EMA), AUSpeech ultrasound (11 dysarthric, 0.74 h, Mandarin, CC BY-NC-ND), UltraSuite UXSSD/UPX (disordered children, ultrasound). Pathological acoustic input for the deviation layer (no articulator GT): TORGO/UASpeech audio.
- **External validity:** clinician perceptual ratings in the Benway/Espy-Wilson mold (118-child precedent) — pursue a real SLP rater; do not fabricate.

### 2.6 Validation plan (the science — pre-registered, honest)
- **H1 (faithfulness validity):** `F` correlates with true inversion error on held-out healthy EMA (HPRC). *Pass:* significant negative correlation, calibrated.
- **H2 (normalization detection):** `F` drops with dysarthria severity on TORGO/UASpeech acoustic input. *Pass:* monotone severity trend.
- **H3 (anti-normalization):** on TORGO's real dysarthric EMA, faithfulness-gated inversion preserves true articulatory deviation better than naive SPARC (which regresses to normal). *Pass:* smaller normalization bias on the real patients (+ augmented), reported with honest n.
- **H4 (clinical alignment):** per-articulator deviation scores track clinician perceptual severity. *Pass:* significant rank alignment.
- **Reuse rigor from prior work:** LOCO references (no speaker-identity leakage), seed=42, fail-loud repro, frozen embeddings, every number → a results file.

### 2.7 Success criteria ("done + phenomenal" for Phase 1)
1. A **live, self-contained demo** (browser + HF Space) where any user speaks and sees a faithful, uncertainty-aware vocal tract in < 200 ms.
2. `F` validated (H1) and shown to detect normalization (H2, H3) on **real** data, honestly scoped.
3. Deviation shown to align with clinical judgment (H4) on real ratings, even if small-n.
4. Open weights + one-command reproduction; nothing stubbed, nothing synthetic presented as real.
5. A paper/preprint whose central claim is the faithfulness metric + honest biofeedback (not "accurate medical avatar").

### 2.8 Risks + mitigations
- **Pathological GT scarce → thin validation.** Mitigation: the faithfulness metric is *GT-free* by design (its main strength); frame the avatar as an illustrative training aid, not a metric medical device; report pathological validation at its true (small) n.
- **Over-normalization so severe the true tongue is unrecoverable.** Then we *honestly flag* "cannot represent faithfully" — that is the product working, not failing.
- **Real-time faithfulness cost.** Run synth+WavLM faithfulness pass at reduced cadence or on GPU; degrade gracefully.

### 2.9 Non-goals (YAGNI)
- Not a diagnostic medical device / no accuracy claim as a measurement instrument (v1).
- No new articulatory data collection in Phase 1 (that's an adjacent opportunity, not a dependency).
- No multi-language in v1 (English first).

---

## 3. Phase 2 — "Clarifier That Knows When To Shut Up" (OUTLINE)

**Goal:** real-time, voice-preserving dysarthric→clear resynthesis whose **star feature is the anti-hallucination safety gate**: when faithfulness `F` (§1) is too low to recover intended words, it does **not** fabricate speech — it degrades to on-screen word/phrase prediction with one-tap user confirmation ("confirm before I speak for you").

**Why after Phase 1:** the core clarifier is contested (CLARIS, CHI 2026; Whispp owns whisper). The defensible, novel contribution is the safety layer, which is exactly Phase 1's faithfulness metric applied to generation. Prove the metric in P1, weaponize it in P2.

**Real data:** SAP (415 h, DUA — needs an institution; pursue), TORGO/UASpeech + a per-user healthy voice bank for identity (parallel dysarthric↔clear data does not exist at scale — this is honest and shapes the design). Stack is open (HuBERT units, EnCodec, zero-shot VC).

**Hardest risk / ethics:** putting words in a disabled person's mouth. The safety gate is the mitigation and the contribution. Requires consent/IRB for any user study.

Full spec when Phase 1 lands.

---

## 4. Phase 3 — "Faithfulness Benchmark + Method" (OUTLINE)

**Goal:** the field-defining, open benchmark + toolkit that generalizes VERITAS's faithfulness metric across tasks (inversion, clarification, ASR) and corpora — quantifying, for any pathological-speech model, *how much it normalizes/fabricates the disorder*. Bundles the prior work's genuine methodological wins (speaker-identity leakage → LOCO fix; silence/articulation decomposition) as diagnostics.

**Why last:** it is the synthesis; it is credible only once the metric is proven in two real systems (P1, P2). This is the "field adopts it" outcome.

**Deliverable:** open benchmark, leaderboard-style harness, reference implementations, and a methods paper. Full spec when Phases 1–2 land.

---

## 5. Data acquisition plan (parallel track — what I need from you)
- **Now, free/open:** HPRC EMA, USC rtMRI, TORGO (have), UASpeech (have), MNGU0 (register), AUSpeech (CC BY-NC-ND), UltraSuite. I download/prepare these.
- **Pursue (gated):** Speech Accessibility Project (needs an institutional DUA — decisive for Phase 2; tell me your affiliation status), PC-GITA, EasyCall (author email), NeuroVoz (open), SAND (ICASSP-2026 challenge).
- **Human:** a licensed SLP willing to give perceptual ratings (Phase 1 H4 / Phase 2 validation). No IRB fabrication — real consent only.
- **Keys/compute:** stack is open-weight (WavLM/HuBERT/SPARC/EnCodec) — no paid API required for core. Training wants an A100-class GPU; inference targets GPU (<200 ms) / CPU (RT-VC-class ~60 ms). Confirm available compute.

## 6. Reuse from the existing repo (nothing rebuilt from zero)
Frozen WavLM extraction, the **exact-parity streaming DTW/engine** (already fixed to r=1.0), the WebSocket server scaffold, LOCO reference machinery, seed/repro/fail-loud infra, MFA phone alignment, 5 processed corpora. VERITAS is a new project layered on these real assets.

## 7. Sequencing + phase gates
P1 (design→plan→build→validate→ship) **must fully land** before P2 begins. P3 needs P1+P2 results. Each phase is independently shippable and independently "phenomenal." No half-built phases.

## 8. Red lines (non-negotiable, per user's standing mandate)
No synthetic/simulated/mock data presented as real. No stubs or dead code in shipped artifacts. No hidden fabrication when computation fails (fail loud). No overclaiming beyond what the validation supports. Data-augmentation, where used (e.g., Phase 2), is labeled as augmentation, never as real patient data. Consent/IRB for any human study is real, not assumed.
