# ATD — World-Class HONEST Finish Plan

**Purpose:** take the ATD dysarthria-severity project to its maximum *defensible, honest* quality ceiling and wrap it up. No hype, no fake claims. Every number below traces to a file in `results/`.

**Generated:** 2026-07-07, grounded in a read-only re-audit of the repo after the 8-agent audit + rebuild session.

**Verified anchor numbers (do not drift from these):**
- Headline: TORGO n=15, speaker-level Spearman **ρ = 0.756**, WavLM-Large **Layer 9**, p = 0.0011, bootstrap CI [0.40, 0.92]. Leakage-robust: LOCO de-leak 0.756 → 0.756; Layer 1 inflates 0.775 → 0.687 (`results/deleak_report_fast.json`).
- Fair baseline (exp42, `results/finetuned_ssl_baseline.json`): WavLM-L9 mean-pool **+ Ridge LOSO ρ = 0.760**, **+ LinearSVR LOSO ρ = 0.775**, MFCC RBF-SVR LOSO ρ = 0.609. **A fair pooled linear probe TIES/BEATS ATD.**
- Streaming: online == offline DTW, **r = 1.0**, max abs err 1.8e-12 vs full-matrix DP, regression-guarded (`results/streaming_dtw_correctness.json`); 132 µs/frame p50 **(DTW step only, on cached features — excludes the ~30 s CPU WavLM forward; r=1.0 is a correctness *identity*, not an empirical finding — see F10)**.
- Per-phoneme: exact `Σ D(p) + residual = total`, but residual (silence) ≈ **0.70–0.76 of path cost** on single-word TORGO. Per-phone ρ: vowels 0.629, consonants 0.625, voiceless stops 0.594 (all p < 0.05).

**Verified prior-art landscape (web-research, 2026-07):**
- **Concurrent frozen-SSL work** (`Muller2026TrainingFree` arXiv 2604.10123, 890 spk; `Subspace2026Collapse` arXiv 2604.21706, 3374 spk) uses a **different method** — *d′ separability along phonological-contrast directions*, NOT DTW trajectory geometry. So ATD's **method is genuinely distinct**. But they are *already* training-free AND open-source, so **"label-free" and "code release" are NOT differentiators** — do not pitch them as novelty. The honest white space vs. them is exactly three things: **(a) LOCO speaker-leakage control, (b) per-phoneme localisation, (c) real-time streaming.**
- **The speaker-identity-leakage *phenomenon* is already published** (WavRx, arXiv 2406.18731 — early SSL layers encode speaker identity). ATD's novel part is **only the specific LOCO reference-construction *fix*** for reference-based severity + the Layer-1 0.775→0.687 inflation demo. **Must cite WavRx and frame as "leakage-hygiene applied," not a discovery.** This is the single biggest reviewer risk.
- **The per-phoneme "tool" space is partially occupied:** GOP-based per-phoneme dysarthria (github `juice500ml/dysarthria-gop`), Park et al. (INTERSPEECH 2025, phoneme explainability), SpeechAce (L2 real-time). **No one combines label-free SSL-geometry + real-time streaming + per-phoneme attribution** — position as a novel **integration**, not "first per-phoneme tool."

---

## 1. THESIS — the single strongest honest positioning

> **ATD is a frozen-SSL articulatory *instrument* for dysarthric speech that reaches parity with trained baselines at zero training cost, and integrates three capabilities no prior system combines.** It scores how far a speaker's WavLM-Large trajectory *geometry* deviates (via DTW) from within-corpus healthy references — a **method distinct** from the concurrent *d′-subspace* frozen-SSL work — and it uniquely brings together **(a) leakage-hygienic reference construction** (leave-one-control-out, which keeps the Layer-9 signal at ρ = 0.756 while the naive early-layer alternative inflates 0.775→0.687), **(b) exact per-phoneme DTW-path localisation** for per-token feedback, and **(c) a real-time streaming variant**. On TORGO (n = 15), ρ = 0.756 at Layer 9 — *mid-pack, not state-of-the-art*, and matched by a fairly-tuned pooled WavLM linear probe (0.760/0.775).
>
> **We claim:** (1) a within-corpus-calibrated severity signal at **parity with trained probes for zero training cost**; (2) a **methodological hygiene contribution** — applying a LOCO reference-construction fix to reference-based severity metrics and demonstrating the early-layer inflation it removes (building on the known speaker-leakage phenomenon, WavRx arXiv 2406.18731); (3) the **novel integration** of label-free SSL-geometry + real-time streaming + per-phoneme attribution into one instrument.
>
> **We explicitly do NOT claim:** state-of-the-art or "beats-SOTA" accuracy (a fair pooled WavLM probe ties us; 0.756 is mid-pack); that "label-free" or "open-source" is a differentiator (the concurrent 890/3374-speaker work is already both); *discovery* of speaker-identity leakage (WavRx published it — we apply the fix); "first per-phoneme dysarthria tool" (GOP/Park et al. exist — we claim the *integration*); zero-shot cross-corpus transfer (it *fails*: TORGO→IPVS ρ = 0.18 NS, TORGO→UASpeech ρ = −0.22); or articulatory/formant-level specificity (our pre-registered perturbation test *falsified* it).

The paper's centre of gravity should be the **integrated instrument + the leakage-hygiene method**, with accuracy framed as *parity at zero training cost*, e.g. *"A frozen-SSL articulatory trajectory instrument for dysarthric speech: leakage-hygienic references, real-time per-phoneme deviation, at parity with trained probes."*

---

## 2. SOLIDIFY THE FOUNDATION — bulletproof-before-wrap checklist

Ordered by how load-bearing each item is. Items marked **BLOCKER** gate the honesty of the headline itself.

| # | Item | Why it's not yet bulletproof | Definition of done |
|---|------|------------------------------|--------------------|
| F1 | **Resolve the silence-confound** **(BLOCKER)** | `experiments/43_silence_confound.py` is written but **has never been run** — `results/silence_confound.json` does not exist. ~73% of DTW cost is silence residual, so we cannot yet say the 0.756 headline is *articulation* rather than *pausing/silence*. | Run it. Decision rule: if **speech-only ρ ≥ ~0.70** with overlapping CI → headline is articulatory, keep 0.756 and report speech-only as a robustness row. If it drops materially (toward the per-phone ~0.63 level) or its CI excludes 0.756 → **make silence-trimming the primary pipeline** and re-headline on trimmed audio. Either way report *both* numbers. |
| F2 | **Make per-phoneme attribution real** | With 70–76% of path cost unattributed silence, per-phone z-scores (and the clinical reports built on them) are weak. | Add MFA-span / VAD **silence trimming before scoring**; re-run the phoneme decomposition so residual becomes a *minority* of cost; re-state per-phone ρ on trimmed audio. Only then is per-token localisation defensible. |
| F3 | **Finalise the honest baseline framing** **(BLOCKER, honesty)** | `exp42` now exists and shows a fair pool+Ridge = 0.760 / LinearSVR = 0.775 — but paper **Table 1** (`paper/tables/table1_main_results.tex`) *still* shows the retracted `WavLM Pool + Ridge = −0.795` with a `TODO`, and **README** still prints −0.795 **and defends it** ("a real result, not a sign error"). This is now factually wrong. | Integrate exp42 into Table 1, abstract, results, intro, conclusion, README. Delete the README "Note on the WavLM Pool baseline". New framing: **ATD ties a fair frozen-SSL linear probe on accuracy, beats tuned MFCC (0.61); its value is real-time + localisation + leakage-robustness.** Remove all `TODO exp42` markers. |
| F4 | **Equivalence-test the accent-specificity** | Specificity currently leans on *non-significant* nulls (L2 PER ρ = 0.041 p = 0.85; L2 Kruskal p = 0.031, NS after Bonferroni; VCTK Kruskal p = 0.075). "p > 0.05" is *absence of evidence*, not evidence of specificity — a reviewer red flag. Also partly rests on a **circular within-corpus z-normalisation**. | Replace "not significant" with **TOST/equivalence bounds** + Bayes factors: show the accent effect is bounded *below* a pre-set fraction of the control–severe gap. Keep the positive control (dysarthric-vs-accented d = 0.9–1.7). Explicitly scope the claim to *within-corpus relative position* (already half-done in limitations). |
| F5 | **Fix + disclaim the LLM clinical reports** | 15 real Gemini/Groq reports exist (`results/clinical_reports/`) from real ATD numbers, but: **(i) every Gemini report is truncated mid-sentence** — the stored API response itself is cut off, missing the required SLP-RECOMMENDATIONS + CAVEAT sections (~7 of 15 incomplete); **(ii) zero clinical validation** (no SLP review, no ground-truth); (iii) they inherit the F2 silence caveat via per-phone z-scores. | **Fix the truncation** (raise `max_output_tokens`, regenerate) *or* drop the broken Gemini half and keep the complete Groq set. Then, default (no SLP access): **demote to "illustrative research-grade narrative, not a clinical instrument; not validated by an SLP"** with an explicit disclaimer; strip any language implying clinical utility. Upgrade path: ≥1 SLP face-validity rating + rubric. |
| F6 | **Wire everything into `reproduce_all` + one-command repro** | `41`/`42` are wired, but the **de-leak (`43b_deleak_fast.py`, which produces the headline leakage asset)** and `43_silence_confound.py` are **not** in the orchestrator. **The CPU one-command path does not actually work today** — Tier-2 depends on embeddings that are not released (F7). | Add 43b + 43 (post-silence-decision) to `reproduce_all.py`; **after F7**, confirm `python reproduce_all.py --skip-extract` regenerates every headline JSON + `paper_numbers.json` on CPU from the released embeddings; add a CI smoke job. Also add a **pytest config** (none exists) and fix the stale "46 tests" claim (actual: ~57). |
| F7 | **Freeze embeddings for reproducibility** **(prerequisite for F6's CPU path)** | REPRODUCE.md Tier-2 points at `zenodo.org/records/<ID>` — a placeholder; embeddings (~66 GB WavLM + ~10 GB XLS-R) are gitignored and unreleased, so **only Tier-1 (read committed JSONs) and Tier-3 (full GPU re-extraction from gated data) work today**. No freeze/checksum script exists. | Deposit the pre-extracted embeddings (Zenodo or HF Datasets) + a **sha256 checksum manifest + a freeze script**; replace `<ID>` with a real DOI. This is the only thing that makes the CPU-friendly repro tier *executable*. |
| F8 | **Reconcile every number to a single source of truth** | Multiple TODOs + a stale README table mean the paper and README currently disagree with `exp42`. | After F1–F5, re-run `21_consolidate_paper_numbers.py`; grep paper + README for any number not in `results/paper_numbers.json`. |
| F9 | **De-oversell the streaming latency** | "132 µs / r = 1.0 real-time" is misleading: 132 µs is the **DTW step only, on cached features** (the real bottleneck is the WavLM forward — ~30 s/utt on CPU per `space/app.py`), and r = 1.0 is a **correctness identity** (exact online DP == offline optimum), not an empirical result. The live mic path (`session.py`) uses a heuristic frame-emission and is **unbenchmarked / not parity-guaranteed**. | Report streaming as: *exact-parity online DTW (correctness proof)* + *per-frame DTW cost 132 µs* + *end-to-end latency dominated by the WavLM forward (X ms GPU / ~30 s CPU)*. Benchmark the true mic-to-score latency, or scope the real-time claim to GPU. Never present 132 µs as end-to-end. |
| F10 | **Commit + tag the branch** | 64 modified + 24 untracked files (incl. exp41/42/43) uncommitted on `audit-cleanup-2026-05-08`. | One clean, reviewed commit; tag a `v1.0-submission` once F1–F9 land. |

---

## 3. FINISHING TASKS — two tiers

### TIER A — NO NEW DATA (reach a strong, submittable, honest artifact)

Target: **~2–3 focused weeks.** This produces a coherent, reproducible, honestly-scoped artifact from what is already on disk.

| ID | Task | Concrete deliverable | Effort |
|----|------|----------------------|--------|
| A1 | Run silence-confound (F1) | `results/silence_confound.json` + decision recorded in method/results | 0.5 d (mostly compute) |
| A2 | Silence-trimmed scoring + re-run phoneme decomposition (F2) | Trimmed pipeline in `src/`, updated `phoneme_summary.json`, updated per-phone ρ | 1–2 d |
| A3 | Integrate exp42 fair baseline everywhere; kill −0.795 (F3) | New Table 1 + abstract/results/intro/conclusion/README; README note deleted | 0.5 d |
| A4 | Equivalence tests + z-norm scoping for specificity (F4) | New TOST/Bayes-factor rows + tightened specificity paragraphs | 1 d |
| A5 | Clinical reports: fix Gemini truncation + disclaim/demote (F5) | Complete reports (or Groq-only); disclaimer in report headers + paper; optional SLP face-validity note | 0.5 d |
| A7 | **Freeze embeddings → Zenodo/HF; real DOI (F7) — do this first, it unblocks A6** | DOI in REPRODUCE.md; sha256 manifest + freeze script | 0.5 d + upload |
| A6 | Wire 43b/43 into `reproduce_all`; add pytest config; one-command CPU repro (needs A7); CI smoke (F6) | Green `reproduce_all.py --skip-extract` from released embeddings + CI badge | 1 d |
| A8 | **Frame the leakage finding as hygiene-applied (NOT discovery)** | Dedicated section/appendix "Leakage-hygienic references for SSL severity metrics" — **cite WavRx (arXiv 2406.18731) as the source of the phenomenon**, then contribute the *LOCO reference-construction fix* + the 9-layer leaked-vs-LOCO sweep + the Layer-1 0.775→0.687 inflation demo + a reusable recommendation (always LOCO; prefer mid-network layers). This is the biggest reviewer-risk item — get the attribution exactly right. | 1 d |
| A9 | Streaming + demo: de-oversell + scope (F10) | Correctness-proof + honest latency figure; **scope the demo deliverable to the real, self-contained HF Space** (does real WavLM+ATD on user audio); label `realtime_demo/` (websocket) as a *local dev scaffold* (hardcoded conda path, needs gitignored data) — either package it or disclose it is not reviewer-runnable | 0.5 d |
| A10 | Paper → venue template + anonymise + page-fit | `main.tex` moved off generic `article` to the target style; figures final; exiftool scrub | 2–3 d |
| A11 | Honest model card + "what we do NOT claim" box | `docs/MODEL_CARD.md` (intended use = research + within-corpus calibration; the four non-claims from the Thesis) | 0.5 d |
| A12 | Optional no-new-data strengthening | Partial-Spearman intelligibility anchor (already have `uaspeech_intelligibility_anchor.json`) framed carefully; per-speaker LOO influence table from exp41 jackknife | 0.5 d |

**Tier-A exit state:** a strong, honest, fully-reproducible artifact whose contribution is *the tool + the leakage method*, with every negative disclosed. Submittable to a workshop / as a main-conference short.

### TIER B — NEW DATA (from "strong" to "genuinely competitive")

Premise: **n = 15 is the fatal ceiling** — the single biggest lever is more speakers with severity labels. **Reality check (web-research, 2026-07): SAP (Speech Accessibility Project, ~1000+ spk) is very likely BLOCKED for an unaffiliated individual — its DUA must be signed by an *institution*.** So the plan must NOT depend on SAP. The realistic near-term scale path is **NeuroVoz + SAND**, both obtainable without an institutional DUA:

| Dataset | Status / access | What it gives |
|---------|-----------------|---------------|
| **NeuroVoz** (Spanish PD, GRBAS-rated) | **OPEN NOW** — Nature Scientific Data 2024 | Immediate real scale-up with clinician severity ratings; cross-language PD |
| **SAND / Vox4Health** (Italian ALS, 339 spk, 5-class severity) | Obtainable via **ICASSP 2026 challenge registration** | 339 speakers, ordinal 5-class severity — a genuine power boost + a challenge leaderboard |
| **PC-GITA** (Spanish PD), **EasyCall** (Italian dysarthric) | By **author email** | Additional cross-language/cross-disorder corpora |
| **SAP** | Likely blocked (institution-signed DUA) | *Only if an institutional partner materialises — do not bank on it* |

| ID | Task | Gate / dependency | What it buys |
|----|------|-------------------|--------------|
| B1 | **Get NeuroVoz (now) + register for SAND.** Email PC-GITA/EasyCall authors in parallel. | NeuroVoz: none (open). SAND: challenge reg. | Escape from n = 15 without SAP |
| B2 | Run ATD + fair pooled-probe + MFCC on NeuroVoz & SAND with speaker-disjoint CV | after B1 | The *real* accuracy verdict at n = 100s with tight CIs (parity or not) |
| B3 | **The decisive experiment:** replicate the LOCO leakage-hygiene effect across NeuroVoz + SAND + TORGO + IPVS + UASpeech and ≥2 SSL backbones | after B2 | Turns the leakage-hygiene note into a **broad, multi-corpus methods contribution** — the strongest path to a main-track paper |
| B4 | SAND 5-class ordinal severity: does ATD's *ranking* hold on a real ordinal scale (not 3-level)? Submit to the ICASSP 2026 challenge track. | after B2 | A benchmarked, externally-comparable number |
| B5 | Clinical anchoring: NeuroVoz GRBAS + SAND severity as external validators of per-phoneme localisation; SLP look at the real-time UI if reachable | parallel to B3/B4 | Makes the "tool" claim clinically grounded |
| B6 | Write up: if the scale corpora confirm ATD *ties* on accuracy but the **integration (leakage-hygiene + streaming + per-phoneme) replicates across corpora**, that is a legitimate main-conference contribution | after B3–B5 | Tier upgrade |

**Sequence:** B1 (NeuroVoz immediate; SAND via challenge) → B2/B3 (2–4 weeks compute+analysis) → B4/B5 (parallel) → B6. **Do not gate any milestone on SAP.**

---

## 4. TARGET VENUES + blunt ceiling assessment

**Current artifact (Tier A, n = 15) — honest ceiling: a strong WORKSHOP paper, or a *borderline* main-conference SHORT paper.** It is **not** a top-tier main-track accept on merit: n = 15, accuracy only *at parity* with a fair linear probe (pitch it as **parity at zero training cost**, never "beats SOTA"), a crowded idea-space (concurrent frozen-SSL work — though ATD's DTW method is distinct from their d′-subspace approach), and a failed zero-shot claim are each, on their own, reasons a strong reviewer rejects a full paper that leads with the metric. **Biggest single reviewer risk:** any hint that ATD *discovers* speaker-identity leakage — WavRx (arXiv 2406.18731) published the phenomenon; we must cite it and claim only the LOCO *fix*.

Best homes, in order of fit:
1. **Clinical/assistive-speech workshops** — ISCA SPSC / speech-&-language-disorders satellites, ACL ClinicalNLP, IEEE SLT special sessions. These *value* honest negatives, tools, and methods notes; highest acceptance probability and the most honest scope for the current artifact. **Recommended primary target.**
2. **Interspeech 2026 (short/regular)** — viable *only if the paper leads with the leakage finding* (the most novel, defensible core) rather than the metric. Expect pushback on n = 15 + no accuracy win.
3. **ICASSP 2027** — the streaming/exact-parity engineering + leakage method fit ICASSP's taste; same n = 15 weakness.

**What moves it up a tier (in order of leverage):**
1. **Scale validation on NeuroVoz (open now) + SAND (challenge)** — the single biggest lever, and *reachable without SAP*. Do NOT bank on SAP (institution-gated).
2. **Making the leakage-hygiene fix a broad multi-corpus / multi-backbone methods result** (Tier B3) — can lift even the n = 15 story toward a main-track short *now*.
3. **A real clinical anchoring of the tool** (NeuroVoz GRBAS / SAND severity as external validators; SLP-in-the-loop if reachable).

Any one helps; all three = a genuine main-conference contribution.

---

## 5. THE DELIVERABLE SET for a "phenomenal" finish

1. **Paper** — honest, venue-formatted, exp42-integrated, silence-resolved, equivalence-tested, with the **leakage finding as headline or co-headline**. Plus an arXiv preprint.
2. **Reproducible code** — one command (`reproduce_all.py --skip-extract`) regenerates every headline number from **frozen embeddings (real Zenodo/HF DOI)**; CI smoke; seed 42; every number → `paper_numbers.json`.
3. **The demo (the tool)** — two honest pieces: (i) the **HF Space** (`space/app.py`), which is *real, self-contained, batch* inference — real WavLM + ATD on uploaded/recorded audio, ~30 s/forward on CPU, references via Git LFS, already self-labelled "research demo only" — **ship and link this as the flagship**; (ii) the **live per-frame mic-streaming** demo (`realtime_demo/` + `src/streaming/server.py`), which is real code but a **local-only dev scaffold** — *either* package it into the Space (add live streaming) *or* clearly disclose it is not reviewer-runnable. Do not conflate the two: the batch Space is the deployable tool today; live streaming is a prototype.
4. **Leakage-hygiene methods note** — a reusable contribution framed correctly: *"Leakage-hygienic reference construction for SSL severity metrics"* — **cite WavRx (arXiv 2406.18731) for the phenomenon**, contribute the LOCO fix + the leaked-vs-LOCO 9-layer sweep + the Layer-1 inflation demo as the recommendation.
5. **Honest model card + limitations** — n = 15; ties a linear probe; no zero-shot; specificity is pathology-vs-accent (not formant-level); clinical reports are illustrative/unvalidated; intended use = research + within-corpus calibration; the four explicit non-claims.

---

## 6. RED LINES — things that would be fake/overclaim; must NOT be done

1. **No SOTA / "beats-SOTA" / accuracy-superiority claim.** A fair pooled WavLM probe ties ATD (0.760/0.775 vs 0.756); 0.756 is mid-pack. Pitch is *parity at zero training cost*. Keep that row visible in Table 1.
2. **No novelty-of-idea claim** for frozen-SSL healthy-reference distance, and **do NOT present "label-free" or "open-source" as a differentiator** — the concurrent 890/3374-speaker work is already both. The honest delta is the *method* (DTW trajectory geometry ≠ their d′ subspace) + the integration of the three capabilities. Cite that work.
3. **Do NOT claim to *discover* speaker-identity leakage.** WavRx (arXiv 2406.18731) published the phenomenon; cite it and claim only the LOCO reference-construction *fix* + the inflation demo.
4. **Do NOT claim "first per-phoneme dysarthria tool."** GOP-based per-phoneme work, Park et al. (INTERSPEECH 2025), and SpeechAce exist; claim the *integration* (label-free SSL-geometry + streaming + per-phoneme), not primacy.
5. **No zero-shot cross-corpus claim.** It fails (0.18 NS / −0.22). Only *within-corpus calibration*.
6. **Never resurrect the "−0.795 pooled" story** or any "ATD beats WavLM pooling" claim. It was an under-regularisation bug; it is retracted. Fix Table 1 + README.
7. **No articulatory/formant-level specificity claim** from the perturbation study. The pre-registered hypothesis was *falsified* (pink noise > formant shift at both layers). Keep it reported as falsified.
8. **Do not sell the accent-specificity nulls as positive proof** via "p > 0.05". Use equivalence tests; disclose the within-corpus z-norm scoping.
9. **Do not let the headline ρ ride on silence.** Run F1 first; if silence-driven, re-headline on trimmed audio. Never present per-phoneme localisation while ~73% of cost is unattributed silence without disclosure.
10. **Do not sell "132 µs real-time" as end-to-end latency,** nor claim the live mic-streaming path has exact parity. 132 µs is DTW-only on cached features; r = 1.0 is a correctness identity on whole-utterance replay; the mic path is approximate and unbenchmarked (F10).
11. **Do not present the LLM clinical reports as clinically validated or diagnostic,** and **do not ship the truncated Gemini reports** as-is. Unvalidated generated text; illustrative only.
12. **Do not report the dysarthric-subgroup ρ = 0.394 (n = 8, p = 0.33) as a CV/generalisation result.** Directional indicator only.
13. **Do not hand-tune per-fold or cherry-pick.** Keep nested CV. Do not make the F04-dropped ρ = 0.855 the headline — report the full leave-one-speaker-out distribution [0.71, 0.86].
14. **Do not ship "Zenodo forthcoming" as if done.** Either deposit real embeddings or clearly mark as post-acceptance.
15. **Do not gate the scale-up story on SAP,** and do not imply SAP access is in hand — it is institution-gated. Plan on NeuroVoz + SAND.
16. **No synthetic data, no fine-tuning (stay frozen), no fabricated numbers.** Every number traces to `results/`.

---

### Bottom line

The project's *accuracy/novelty* story is capped and honestly should be. Its defensible, phenomenal finish is the **integrated instrument + the leakage-hygiene method**: run the silence test, integrate the fair baseline (parity, not superiority), frame the leakage fix against WavLM/WavRx correctly, ship the real HF-Space demo and a *working* frozen-embedding repro, disclaim what isn't validated — then scale on **NeuroVoz (open now) + SAND (challenge)**, not the institution-gated SAP, which is the realistic path from "strong and honest" to "genuinely competitive."
