# Track C — Pre-Registered Predictions (LOCKED)

**Frozen:** 2026-05-08. Any deviation from this document during analysis must
be reported as a deviation, not a finding.

## Hypothesis

ATD measures *articulatory execution geometry*, not generic acoustic
distance. Length-normalized DTW + WavLM denoising-pretraining make ATD
insensitive to non-articulatory distortions (rate, additive noise, channel)
and sensitive to articulatory distortions (formant shift, frame dropout).

If the predicted ordering below is observed (with statistical confidence
matching the pre-registered tests), ATD's mechanistic interpretation is
supported. If MFCC-trajectory DTW also responds invariantly to time-stretch
or shows the same ordering, ATD's specificity advantage is unsupported.

## Predicted ordering (locked)

```
β(pink_noise)  <  β(time_stretch) ≈ β(reverberate) ≈ β(pitch_shift)
                                  ≪
               β(formant_shift)   ≈   β(partial_mute)
```

`β` = slope of ΔATD with respect to z-scored intensity. ΔATD = ATD(perturbed)
− ATD(clean) per (speaker, word) pair, computed under the LOCO reference
protocol (held-out control's references rebuilt from the other 6 controls).

## Per-family predictions

| Family | Intensity grid (physical units) | Predicted β (relative to control→severe gap) | Predicted rank |
|---|---|---|---|
| time_stretch (PSOLA) | rate ∈ {0.6, 0.8, 1.0, 1.25, 1.5, 1.75} | ≤ 5% | LOW |
| pink_noise | SNR ∈ {30, 20, 10, 5, 0, −5} dB | ≤ 10% | LOW-MID |
| formant_shift (LPC) | F1/F2 scale ∈ {0.85, 0.90, 0.95, 1.05, 1.10, 1.15} | ≥ 50% | HIGH |
| partial_mute | drop% ∈ {2, 5, 10, 20, 30} | ≥ 50% | HIGH |
| reverberate | RT60 ∈ {0.1, 0.3, 0.6, 1.0, 1.5} s | 10–30% | LOW-MID |
| pitch_shift (PSOLA) | semitones ∈ {−6, −3, −1, +1, +3, +6} | 10–25% | LOW-MID |

The "control→severe gap" anchor is the absolute mean-ATD difference between
TORGO controls and TORGO severe speakers measured on the LOCO baseline,
extracted from `results/atd_validation_report.json`.

## Statistical model (locked)

Linear mixed-effects (lme4 / statsmodels MixedLM):

```
delta_atd ~ C(family, Treatment("time_stretch")) * intensity_z
            + (1 + intensity_z | speaker)
            + (1 | word)
```

REML estimation. 95% CIs from `res.conf_int(0.05)`. Family β is the
contrast against `time_stretch` baseline.

## Pre-registered contrasts (one-sided, Holm-corrected, α = 0.05)

1. β(formant_shift) > β(time_stretch)
2. β(partial_mute)  > β(time_stretch)
3. β(formant_shift) > β(pink_noise)
4. β(formant_shift) > β(reverberate)
5. β(partial_mute)  > β(reverberate)
6. β(time_stretch)  ≈ 0 (two-sided, |β| < 0.05·gap, with CI containing 0)

## MFCC specificity comparison (locked)

The same model is run on `mfcc_dtw` (MFCC-trajectory DTW distance) instead of
`atd_layer9`. ATD's specificity claim is supported iff:

- ATD: β(time_stretch) ≈ 0 (CI contains 0), AND
- MFCC: β(time_stretch) ≠ 0 (CI excludes 0)

## Sample protocol

- 7 controls × 794 words (or shared-words subset for sanity run)
- 6 perturbation families × ~6 intensity levels = 36 conditions
- LOCO reference cache: 7 packs in `data/processed/references_loco/`
- Random seed: 42 (in `src.constants.RANDOM_SEED`)
- All perturbations deterministic via `hashlib.md5(family|intensity|key)`

## Falsification criteria

Track C **fails** (and we say so honestly in the paper) if any of:

1. β(time_stretch) is significantly non-zero (p < 0.05 corrected) — then
   length-normalization does not absorb rate as claimed.
2. β(pink_noise) > β(formant_shift) — then ATD is more sensitive to
   acoustic noise than articulatory distortion.
3. MFCC's β(time_stretch) ≈ 0 too — then ATD has no specificity advantage.

Failures will be reported in the paper's Limitations section, not buried.

## Compute budget

- Empirical: ~0.05 s WavLM forward per ~1 s utterance on RTX 4080 (FP16).
- 200,088 perturbed utterances × 0.05 s ≈ 2.78 GPU-h.
- LOCO reference rebuild: ~2.3 h (mixed CPU/GPU).
- DSP: ~5 min wall (joblib 8 workers).
- **Total: ≤ 6 GPU-h.**

## Outputs

- `results/perturbation_atd.csv` — one row per (speaker, word, family, intensity)
  with `atd_clean`, `atd_perturbed`, `mfcc_clean`, `mfcc_perturbed`,
  `delta_atd`, `delta_mfcc`.
- `results/perturbation_lme.json` — fitted model coefficients + CIs +
  contrast results.
- `results/figures/fig9_perturbation_validation.{png,pdf}` — main result figure.
