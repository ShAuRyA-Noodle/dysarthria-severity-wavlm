# Pre-registration v2 — ATD primary endpoints (locked 2026-05-11)

> Reviewer-defense locked document. Any deviation from this protocol after
> this commit must be reported in the paper as a *deviation from
> pre-registration* with rationale, not as a "found" result.

## Author

Shaurya (ShAuRyA-Noodle, solo first author), independent.

## Lock timestamp

This document is locked at the commit that introduces it (`git log -1`).
Subsequent analysis decisions deviating from below must reference the
deviation explicitly in the paper.

---

## Primary corpus

**TORGO** (Rudzicz et al. 2012). n=15 speakers (7 control, 8 dysarthric).
Headset-microphone subset, 16 kHz resampled, −20 dBFS normalised.

## Primary feature

**WavLM-Large hidden states at Layer 9.** Justification: pre-audit selection
based on existing literature on mid-network speech features
(Chen et al. 2022 WavLM, layer-9 articulatory peak). Reported in
`results/best_layer.json` from the prior audit.

Layer-1 results are reported as **secondary/exploratory**, NOT swapped
into primary. The historical commits `27c9e78`, `ec795e2`, `292fd06`,
`3b257a8` that moved Layer 1 into the abstract were a methodological
error and must be reversed at the next paper revision.

## Primary metric

ATD as defined in `src/atd_metric.py`:

  ATD(u, w, ℓ) = DTW(PCA_64(E_u^(ℓ)), R_w^(ℓ)) / max(T_u, T_R)

Speaker-level mean across all utterances of that speaker.

## Primary endpoint

**Speaker-level Spearman ρ** between speaker mean ATD (Layer 9) and TORGO
severity_score ∈ {0, 1, 2}, computed over n=15 speakers.

Hypothesis: ρ > 0 with two-sided 95% bootstrap CI excluding zero.

Pre-registered value of interest: TORGO speaker-level ρ at Layer 9 with
10k bootstrap CI. **No layer selection at any later stage.**

## Secondary endpoints (must be labelled exploratory in paper)

1. Layer-1 Spearman (`results.tex` Layer-1 specificity section)
2. LOSO ρ on dysarthric-only (n=8); non-significance expected per
   Fisher-power analysis at this sample size
3. Cross-corpus IPVS Layer-1 sustained-vowel discrimination
4. Track C perturbation slopes at L1 + L9
5. Phoneme-level decomposition correlations
6. Streaming offline-online parity
7. SSL backbone bake-off

All secondaries reported in dedicated sections labelled "Secondary /
exploratory" with explicit warning that they were not pre-locked.

## Multiple-comparison correction

For any family of tests reported in the paper:
- Bonferroni at α=0.05 across the family
- Benjamini–Hochberg FDR at q=0.05

Both reported side-by-side in tables.

## Reporting deviations from pre-registration

The following are deviations recorded for honesty:
1. **Layer-1 moved to primary abstract earlier in development** (commits
   listed above). Reverted in next revision.
2. **Track C predictions partially falsified at Layer 9** (pink-noise,
   reverberation, time-stretch all responded more than predicted).
   Reported in `results.tex` as falsified, not rescued.
3. **IPVS within-corpus required corpus-internal PCA refit**. This is
   NOT zero-shot. Will be re-labelled "within-corpus" in next revision.
4. **Zero-shot TORGO→IPVS ρ=0.18 NS** (`zero_shot_torgo2ipvs.json`) is
   the only genuine zero-shot test and it FAILED. Cannot be relabelled.
5. **Differential-dx classifier both raw (F1=0.573) and z-rank (F1=0.163)
   collapse**. Cannot be presented as a deployment result.

## What this pre-registration does NOT rescue

- n=15 sample size for the primary endpoint
- Lack of clinical co-author
- Absence of a fine-tuned-SSL baseline at lock time. **Update (post-lock):**
  a fairly-tuned frozen-WavLM probe (mean-pool L9 + Ridge/LinearSVR, head
  and hyperparameters refit per LOSO fold) plus a tuned MFCC-RBF baseline
  have since been added (`results/finetuned_ssl_baseline.json`). They show
  the trained probe (rho = 0.760 / 0.775) matches ATD (0.756) — i.e. ATD's
  contribution is label-free zero-shot operation, not superior accuracy.
  Full backbone fine-tuning remains degenerate at n=15.
- F04 leverage on the headline

## Signed

`audit-cleanup-2026-05-08` branch, commit hash will be stamped at
the merge to main.
