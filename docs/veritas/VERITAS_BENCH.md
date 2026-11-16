# VERITAS-Bench — a validity test-suite for frozen-SSL clinical-speech metrics

**Phase 3 of the VERITAS project.** Anyone can build a training-free metric on a frozen
SSL model (WavLM/HuBERT) and report a single-corpus correlation with severity. VERITAS-Bench
answers the harder question: *does that number measure the disorder, or a confound?* Feed it
**any per-speaker metric** and a corpus and it runs six validity tests, emitting a machine-readable
**PASS / WARN / FAIL report card**. The thresholds encode this project's own findings (speaker-identity
leakage, silence/pausing, energy, cross-corpus overfitting) as reusable acceptance criteria.

It is not a leaderboard. A metric "wins" only by surviving the confound controls.

## The six tests

| Test | Statistic | PASS | WARN | FAIL | N/A |
|---|---|---|---|---|---|
| **predictive_validity** | within-corpus LOSO Spearman ρ (metric vs severity) | ρ≥0.50 & p<0.05 | ρ≥0.30 | ρ<0.30 | ρ undefined |
| **leakage_loco** | ρ(naive ref) − ρ(leave-one-control-out ref) | Δ≤0.05 | Δ≤0.15 | Δ>0.15 | metric not reference-based |
| **silence_confound** | retention = ρ(speech-only)/ρ(full); + silence frame fraction | retain≥0.70 & silence<0.50 | retain≥0.40, or majority-silence | retain<0.40 | not applicable |
| **energy_confound** | partial Spearman(metric,severity \| mean speech energy) vs raw | retain≥0.70 | retain≥0.40 | retain<0.40 | not applicable |
| **cross_corpus_transfer** | train probe on other corpora → test held-out; speaker ρ | ρ≥0.50 & p<0.05 | ρ≥0.30 & p<0.05 | ρ<0.30 or n.s. | <2 corpora / no representation |
| **leakage_by_layer** | speaker-classification AUC of the metric's representation | AUC≤0.70 | AUC≤0.85 | AUC>0.85 | no representation exposed |

α = 0.05 throughout. The overall verdict is the **worst applicable** test (N/A ignored unless all N/A).
Every threshold is a documented policy choice in [`src/veritas/bench/thresholds.py`](../../src/veritas/bench/thresholds.py)
(`THRESHOLD_DOC`), so a card is auditable — change the policy in one place.

## Plug in your own metric

Subclass `VeritasMetric` and implement `predictive_scores` (higher = more severe). That alone runs
predictive-validity, silence, energy, cross-corpus, and (if you expose a representation) leakage-by-layer.
Reference-based metrics additionally implement `loco_scores` to enable the leakage test.

```python
import src.veritas                      # offline HF-cache bootstrap (import FIRST)
import numpy as np
from src.veritas.bench import build_corpus_bundle, run_bench, render_table
from src.veritas.bench.metric import VeritasMetric
from src.veritas.corpora import torgo

class MyMetric(VeritasMetric):
    name = "my_metric"
    kind = "intrinsic"          # or "reference"
    has_representation = True    # expose L9 features -> enables leakage-by-layer
    representation_desc = "WavLM-L9 mean-pool"

    def predictive_scores(self, views, *, feature_view="full"):
        # views: per-speaker pooled features; return {speaker_id: float}
        out = {}
        for v in views:
            feats = v.pooled_full if feature_view == "full" else v.pooled_speech
            out[v.speaker_id] = float(np.linalg.norm(feats.mean(axis=0)))
        return out

recs   = torgo.load()
bundle = build_corpus_bundle(recs, device="cuda:0", with_silence=True, need_speech_pool=True)
card   = run_bench(MyMetric(), bundle, other_bundles=[], seed=42)
print(render_table([card]))
```

For a reference-based metric (distance to a healthy-control reference), also override
`loco_scores(self, views) -> dict | None` to return leave-one-control-out scores; returning `None`
marks the leakage test N/A. See the shipped adapters
[`src/veritas/bench/metrics/atd.py`](../../src/veritas/bench/metrics/atd.py) (reference-based) and
[`pooled_probe.py`](../../src/veritas/bench/metrics/pooled_probe.py) (intrinsic probe) for full examples,
and [`tests/test_veritas_bench.py`](../../tests/test_veritas_bench.py) for the plug-in contract.

## Run the self-demonstration

```
C:/Users/Dell/.conda/envs/dysarthria/python.exe experiments/veritas/B3_run_bench.py
```

This applies the full battery to the repo's own metrics — ATD reference-distance (WavLM-L9 and the
early-layer L1 leakage demonstrator) and a fairly-tuned pooled-WavLM Ridge probe — on real TORGO
(with IPVS + EasyCall as cross-corpus references) and writes
[`results/veritas/veritas_bench_report.json`](../../results/veritas/veritas_bench_report.json).
Real data only, seed 42; no synthetic inputs.

## Reference report card (real TORGO, seed 42)

`B3_run_bench.py` produces this. The suite reproduces the paper's committed findings **exactly** —
e.g. ATD speech-only ρ=0.181 and early-layer leakage Δ=0.089 match the standalone
`silence_confound.json` / `deleak_report_fast.json`.

| Test | ATD WavLM-L9 (ref) | ATD WavLM-L1 (ref) | Pooled-WavLM probe |
|---|---|---|---|
| predictive_validity | PASS ρ=0.772 | PASS ρ=0.687 | PASS ρ=0.745 |
| leakage_loco | PASS Δ=0.000 | **WARN** Δ=0.089 | N/A (intrinsic) |
| silence_confound | **FAIL** 0.772→0.181 (24% kept; 63% silence) | — | WARN 0.745→0.745 (100% kept; 63% silence) |
| energy_confound | PASS (96% kept) | — | PASS (80% kept) |
| cross_corpus_transfer | **FAIL** ρ=0.432 (p=0.11, n.s.) | — | **FAIL** ρ=0.432 (p=0.11, n.s.) |
| leakage_by_layer | **FAIL** AUC=0.996 | — | **FAIL** AUC=0.996 |
| **OVERALL** | **FAIL** | **WARN** | **FAIL** |

**Reading — no metric PASSES overall, the paper's thesis delivered as a reusable verdict:**
- The **ATD reference-distance** metric FAILS: its severity signal is 76% carried by silence
  (speech-only keeps only 24%), it does not transfer cross-corpus, and its representation encodes
  speaker identity (AUC 0.996).
- The **early layer (L1)** additionally leaks identity through the reference (Δ=0.089) — the reason
  to prefer mid-network layers.
- The **pooled probe** is *not* silence-dependent (keeps 100% — a real discriminative finding the
  battery surfaces), yet still fails cross-corpus transfer and carries the same identity-leakage risk.

The value is exactly this: a single headline ρ (0.74–0.77 — all "PASS" on predictive validity) hides
that every metric fails at least one confound control. VERITAS-Bench makes that visible in one card.

## Auditing the strongest published metric family

To show the battery **discriminates** rather than simply failing everything, we ran it on an
independent reimplementation of the strongest current training-free metric family — the
**phonological-subspace d′** metric of Muller et al. (2026): MFA phone-level frozen-WavLM embeddings,
binary phonological-contrast directions (nasality/voicing/stridency/sonorance/continuancy) estimated
from healthy controls only, scored per speaker by d′ along those directions.
(`experiments/veritas/B4_audit_phonosubspace.py` → `results/veritas/veritas_bench_phonosubspace.json`.
**This is our reimplementation from the published description, not the authors' code.**)

| Control | ATD-L9 | Pooled probe | **Phono-subspace d′** |
|---|---|---|---|
| predictive validity | PASS ρ=0.772 | PASS ρ=0.745 | **PASS ρ=0.741** |
| leakage (LOCO Δ) | PASS 0.000 | N/A | **PASS 0.000** |
| silence (kept) | **FAIL** 24% | WARN 100% | **WARN** 100% |
| energy (kept) | PASS 96% | PASS 80% | **PASS 88%** |
| cross-corpus | **FAIL** n.s. | **FAIL** n.s. | **FAIL** ρ=0.467 (p=0.079) |
| identity AUC | **FAIL** 0.996 | **FAIL** 0.996 | **FAIL** 0.999 |
| **overall** | **FAIL** | **FAIL** | **FAIL** |

**Reading — the battery works as intended:**
- It **validates** the phonological-subspace design where it deserves it: leakage-robust, energy-robust,
  and *silence-robust by construction* (100% retained, versus 24% for the trajectory-distance metric).
  A benchmark that only ever returns FAIL is a hit-piece; this one separates good design from bad.
- It **surfaces one confound the original work does not report checking**: the representation is almost
  perfectly speaker-identifiable (**AUC 0.999**). Crucially this holds for *every* metric here
  (0.996–0.999), so it is a **systemic property of the frozen SSL representation** — a leakage risk
  inherited by any reference-based clinical metric built on it — not a defect unique to that method.
- The cross-corpus FAIL is on our **n=15 graded** TORGO and is **underpowered** (critical ρ≈0.52).
  The original work demonstrates transfer at n=3,374; we do not claim otherwise. The battery reports
  what it can measure on the data at hand, and says so.
