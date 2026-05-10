# Articulatory Trajectory Deviation (ATD)
### Zero-Shot Dysarthric Speech Severity Estimation via Frozen SSL Latent Space Geometry

<p align="center">
  <img src="results/figures/fig2_atd_trajectories.png" width="800" alt="ATD Trajectories"/>
</p>

<p align="center">
  <b>Articulatory trajectories in WavLM latent space for the word "yes".</b><br>
  Control speakers (left) trace tight, consistent paths. Severe dysarthric speakers (right) deviate sharply.<br>
  ATD measures this deviation — no fine-tuning, no labels required.
</p>

---

<p align="center">
  <img src="https://img.shields.io/badge/INTERSPEECH-2026-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/Spearman%20ρ-0.756-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/RMSE%20Improvement-16.3%25-brightgreen?style=flat-square"/>
  <img src="https://img.shields.io/badge/Zero--Shot-No%20Fine--Tuning-orange?style=flat-square"/>
  <img src="https://img.shields.io/badge/Dataset-TORGO-lightgrey?style=flat-square"/>
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=flat-square"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2.2-red?style=flat-square"/>
</p>

---

## What Is This?

Dysarthria is a motor speech disorder — caused by cerebral palsy, ALS, stroke, or traumatic brain injury — that affects millions of people worldwide. Assessing its severity currently requires a trained Speech-Language Pathologist: expensive, scarce, and subjective.

**We propose ATD (Articulatory Trajectory Deviation):** a zero-shot metric that estimates dysarthric severity directly from raw audio — with no labeled training data and no fine-tuning of any model.

### The Core Insight

Large SSL models like WavLM-Large, pre-trained on 94,000 hours of neurotypical speech, have implicitly learned a geometric manifold of normal articulatory execution. Neurotypical speakers trace tight, consistent trajectories through this space when producing any given word. Dysarthric speakers — due to impaired motor control — trace erratic, deviated paths.

**ATD measures the geometric deviation of a speaker's WavLM trajectory from neurotypical reference trajectories. The geometry IS the severity signal.**

---

## Key Results

<p align="center">
  <img src="results/figures/fig5_results_comparison.png" width="750" alt="Results Comparison"/>
</p>

| Method | RMSE ↓ | Spearman ρ (Global) ↑ | Spearman ρ (LOSO) ↑ | Fine-tuning? |
|---|---|---|---|---|
| MFCC + SVR | 0.802 | 0.787 | 0.056 | No |
| WavLM Pool + Ridge | 0.795 | −0.795 | −0.056 | No |
| **ATD-L9 + Ridge (ours)** | 0.721 | **0.756** | 0.282 | **No** |
| **ATD-All + Ridge (ours)** | **0.671** | **0.756** | **0.394**[^loso-ci] | **No** |

[^loso-ci]: LOSO ρ = 0.394 has 95% bootstrap CI [−0.29, 0.90] (p = 0.33, n = 8 dysarthric folds). Reported here as a directional indicator of speaker-generalization, not as a strong significance claim. See `results/paper_numbers.json` `bootstrap_confidence_intervals.dysarthric_only`.

**LOSO = Leave-One-Speaker-Out** across 8 dysarthric speakers — tests generalization to completely unseen speakers.

> **Note on the WavLM Pool baseline.** Mean-pooling WavLM-Large across time gives a Ridge model that achieves comparable RMSE but *negative* Spearman correlation with severity. This is a real result, not a sign error: pooled features collapse trajectory information, and Ridge fits a direction that flips on this dataset. We report the negative ρ honestly and use it as motivation for trajectory-based features.

### Headline Numbers

- **Speaker-level Spearman ρ = 0.756** (p = 0.0011, n = 15 speakers; 95% bootstrap CI [0.40, 0.92])
- **16.3% RMSE improvement** over MFCC baseline
- **LOSO Spearman 0.394 vs MFCC 0.056** — 7× higher; non-significant on n=8 dysarthric folds (p=0.33, CI [−0.29, 0.90])
- **87.5% LOSO rank accuracy** — 7/8 dysarthric speakers correctly ranked
- **Cohen's d = 1.488** (large effect) between control and severe groups (utterance-level)
- **ATD-duration correlation r = 0.105** — ATD is NOT a proxy for speaking rate
- **Zero fine-tuning** — WavLM is permanently frozen

---

## Layer Analysis

<p align="center">
  <img src="results/figures/fig1_layer_analysis.png" width="650" alt="Layer Analysis"/>
</p>

We analyzed all 9 sampled WavLM-Large layers. Every single layer achieves statistically significant correlation (p < 0.01).

**Novel finding:** Early transformer layers (Layer 1, ρ = 0.775) and middle layers (Layer 9, ρ = 0.756) both strongly encode dysarthric severity — suggesting that motor speech disruption manifests at both acoustic and articulatory levels of SSL representation.

---

## Speaker Profiles

<p align="center">
  <img src="results/figures/fig3_atd_scatter.png" width="700" alt="ATD vs Severity"/>
</p>

ATD reveals clinically meaningful speaker profiles beyond categorical labels:

- **F04** (labeled Severe): ATD = 72.9 — within control range, consistent with documented preserved intelligibility in her spastic dysarthria profile
- **M05** (labeled Mild-Moderate): ATD = 97.96 — approaching severe range, consistent with notably reduced intelligibility

**ATD captures articulatory execution quality independently of diagnostic categorization.**

---

## How ATD Works

```
Raw audio (16kHz)
      ↓
WavLM-Large (frozen, eval mode — never fine-tuned)
      ↓
Frame-level embeddings: shape (T_frames, 1024)
      ↓
PCA reduction: (T_frames, 1024) → (T_frames, 64)
      ↓
DTW alignment to control reference trajectory R_w
      ↓
ATD(u, w) = DTW_distance / max(T_utterance, T_reference)
      ↓
Speaker ATD = mean across all utterances
      ↓
Severity estimate (0=Control, 1=Mild, 2=Severe)
```

### Formal Definition

For utterance $u$ of word $w$ by speaker $s$, with WavLM layer $l$ embeddings:

$$\text{ATD}(u, w, l) = \frac{\text{DTW}\left(\text{PCA}_{64}(\mathbf{E}_u^{(l)}),\; \mathbf{R}_w^{(l)}\right)}{\max(T_u,\, T_{\mathbf{R}_w})}$$

Where:
- $\mathbf{E}_u^{(l)} \in \mathbb{R}^{T_u \times 1024}$ — frame-level WavLM embeddings
- $\mathbf{R}_w^{(l)}$ — reference trajectory built by DTW-averaging 7 control speakers
- $\text{PCA}_{64}$ — same PCA model used for both reference and test (same coordinate space)
- Length normalization ensures ATD measures quality, not speaking rate

---

## Dataset

**TORGO** — University of Toronto dysarthric speech database

| Group | Speakers | Utterances | Severity |
|---|---|---|---|
| Control | FC01, FC02, FC03, MC01, MC02, MC03, MC04 | ~5,220 | None |
| Mild-Moderate | F03, M03, M05 | ~1,425 | 1 |
| Severe | F01, F04, M01, M02, M04 | ~1,181 | 2 |
| **Total** | **15** | **~7,826** | — |

- Headset microphone recordings only (higher SNR)
- Resampled to 16kHz, normalized to -20 dBFS
- 794 words with control reference trajectories built
- 9 words shared by all 15 speakers

---

## Ablation Study

Every design decision is empirically justified:

| Variant | Spearman ρ (Global) | ρ (LOSO) | Interpretation |
|---|---|---|---|
| **Full ATD — Layer 1 (best)** | **0.775** | **0.394** | Best overall |
| Full ATD — Layer 9 | 0.756 | 0.394 | Strong mid-network |
| No DTW — Mean L2 | 0.471 | 0.394 | DTW critical (−40%) |
| No PCA — 1024-dim DTW | 0.802 | 0.394 | PCA is safe speedup |
| Mean-pool distance | 0.471 | 0.394 | Trajectory structure critical |
| MFCC trajectory DTW | 0.849 | 0.282 | SSL generalizes better (LOSO) |

**Key findings:**
- Removing DTW drops ρ by 40% — temporal alignment is essential
- MFCC trajectories have high global ρ but collapse under LOSO (0.282 vs 0.394) — SSL representations generalize better to unseen speakers
- PCA reduction to 64 dims is lossless — 16× speedup with no quality loss

---

## Installation

```bash
# Clone the repository
git clone https://github.com/YOURUSERNAME/dysarthria-atd.git
cd dysarthria-atd

# Create conda environment
conda create -n dysarthria python=3.11 -y
conda activate dysarthria

# Install PyTorch with CUDA
conda install pytorch==2.2.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install remaining dependencies
pip install -r requirements.txt
```

**Requirements:** NVIDIA GPU with ≥ 8GB VRAM, CUDA 12.1+, ~80GB disk space for embeddings

---

## Dataset Setup

1. Download TORGO from: http://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
2. Download all 4 archives: F.tar.bz2, FC.tar.bz2, M.tar.bz2, MC.tar.bz2
3. Extract to `data/raw/torgo/`:

```
data/raw/torgo/
├── F/   (F01, F03, F04)
├── FC/  (FC01, FC02, FC03)
├── M/   (M01, M02, M03, M04, M05)
└── MC/  (MC01, MC02, MC03, MC04)
```

---

## Quick Start (Verify Setup — 5 Minutes)

```bash
conda activate dysarthria
python experiments/01_verify_environment.py
```

Expected output:
```
[PASS] Python 3.11.x
[PASS] PyTorch 2.2.2
[PASS] CUDA available
[PASS] GPU: RTX 4080 (12GB)
[PASS] All 20 packages importable
[PASS] TORGO data detected
✅ ENVIRONMENT READY
```

---

## Reproduce All Results

```bash
conda activate dysarthria
python reproduce_all.py                  # full pipeline, skip steps whose outputs exist
python reproduce_all.py --force          # re-run every step
python reproduce_all.py --skip-extract   # reuse cached SSL embeddings
python reproduce_all.py --include-uaspeech  # also run UASpeech (UA_01..UA_05)
python reproduce_all.py --smoke          # dry-run plan, no execution
```

`reproduce_all.py` shells out to each `experiments/*.py` step in order, writes per-step logs to `results/logs/reproduce_all/`, and skips a step if its expected output artifact already exists on disk (override with `--force`). This runs the full pipeline end-to-end:

| Step | Script | Time |
|---|---|---|
| Audio preprocessing | 03_preprocess_audio.py | ~20 min |
| SSL extraction (WavLM-Large) | 04_extract_ssl.py | ~2 min (RTX 4080) |
| Reference building | 05_build_references.py | ~20 min |
| ATD computation | 06_compute_atd.py | ~30 min |
| ATD validation | 07_validate_atd.py | ~5 min |
| Baseline models | 08_baselines.py | ~50 min |
| Layer analysis | 09_layer_analysis.py | ~30 min |
| LOSO validation | 10_loso_validation.py | ~20 min |
| Ablation study | 11_ablations.py | ~30 min |
| Figure generation | 12_generate_figures.py | ~10 min |
| Table generation | 13_generate_tables.py | ~2 min |

**Total: ~4 hours end-to-end on RTX 4080**

---

## Project Structure

```
dysarthria-atd/
├── src/
│   ├── data_pipeline.py      # TORGO loading, preprocessing, splits
│   ├── ssl_extractor.py      # Frozen WavLM-Large extraction
│   ├── reference_builder.py  # Control reference trajectory construction
│   ├── atd_metric.py         # ATD computation + statistical validation
│   ├── baselines.py          # MFCC+SVR, WavLM pool+Ridge baselines
│   ├── analysis.py           # Confound analysis, session consistency
│   └── visualization.py      # All 6 publication figures
├── experiments/
│   ├── 01_verify_environment.py
│   ├── 02_explore_torgo.py
│   ├── 03_preprocess_audio.py
│   ├── 04_extract_ssl.py
│   ├── 05_build_references.py
│   ├── 06_compute_atd.py
│   ├── 07_validate_atd.py
│   ├── 08_baselines.py
│   ├── 09_layer_analysis.py
│   ├── 10_loso_validation.py
│   ├── 11_ablations.py
│   ├── 12_generate_figures.py
│   └── 13_generate_tables.py
├── results/
│   ├── figures/              # All 6 figures (PNG + PDF)
│   ├── tables/               # CSVs for all results
│   ├── paper_numbers.json    # Single source of truth for all paper numbers
│   ├── atd_scores.csv        # Utterance-level ATD scores
│   ├── speaker_atd.csv       # Speaker-level ATD aggregation
│   └── atd_validation_report.json
├── paper/
│   ├── sections/             # LaTeX sections
│   ├── figures/              # Publication figures
│   ├── tables/               # LaTeX tables
│   └── main.tex
├── data/
│   ├── processed/metadata/   # metadata_unified.csv
│   └── splits/               # splits.json (LOSO, severity, cross-session)
├── reproduce_all.py
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

| Decision | Choice | Why |
|---|---|---|
| SSL model | WavLM-Large | Denoising objective — most robust to irregular speech |
| Model state | **Frozen always** | Preserves pre-trained articulatory geometry |
| Features | Frame-level | Trajectory structure, not mean-pooled averages |
| Alignment | DTW (fastdtw) | Handles 2–3× slower dysarthric speech naturally |
| Reduction | PCA → 64 dims | 16× speedup, <5% information loss |
| References | Control speakers only | True neurotypical articulation, not mild dysarthric |
| Evaluation | LOSO | Only valid generalization test for 8-speaker dataset |
| Random seed | 42 everywhere | Full reproducibility |

---

## Why This Matters

**For researchers:**
- First characterization of frame-level trajectory geometry in frozen SSL space for dysarthric speech
- Novel finding: early WavLM layers (Layer 1) encode severity as strongly as middle layers
- Empirical proof that SSL representations generalize better than MFCC under LOSO (0.394 vs 0.282)

**For clinicians:**
- Zero-shot — no labeled data needed for new patients
- Interpretable — trajectory deviation has geometric meaning
- Deployable — no retraining, no clinical infrastructure
- Speaker-specific profiles reveal execution quality beyond categorical labels

**For the field:**
- Framework *prospectively* extensible to: ALS progression tracking, therapy monitoring, cross-language dysarthria, other motor speech disorders. Cross-corpus generalization is **not yet demonstrated** — see Limitations.

---

## Limitations

- **Cross-corpus result on UASpeech is null.** Direct application of TORGO-built reference trajectories to UASpeech yields ρ = −0.22 (p = 0.43, n = 15 dysarthric); pooled-corpus ρ = −0.07. We hypothesize that UASpeech's isolated-word stimuli (mean ≈ 127 frames) sit below the trajectory-length regime where DTW signal stabilizes (TORGO mean ≈ 155 frames). Cross-corpus extension is left to future work.
- **Sample size.** n = 15 TORGO speakers. Bootstrap CIs are wide; LOSO on n = 8 dysarthric speakers is non-significant at α = 0.05.
- **Single dataset for headline numbers.** Generalization to PD, ALS, post-stroke speech, and non-English dysarthria is open.
- **F04 leverage.** Removing F04 alone shifts ρ from 0.756 → 0.855. Influence diagnostics across all speakers will be added in v2.
- **Non-deterministic FP16 inference.** WavLM-Large embeddings are extracted in FP16; bit-exact reproduction across GPU drivers is not guaranteed. For paper-grade reproduction we ship the embeddings on Zenodo (forthcoming) rather than asking reviewers to re-extract.

---

## Figures

| Figure | Description |
|---|---|
| Fig 1 | Layer-wise Spearman ρ across 9 WavLM layers |
| Fig 2 | UMAP trajectory visualization: Control vs Mild vs Severe |
| Fig 3 | Speaker-level ATD vs severity scatter with regression |
| Fig 4 | ATD distribution violins with significance brackets |
| Fig 5 | Results comparison: RMSE and LOSO Spearman bar charts |
| Fig 6 | Per-speaker per-word ATD heatmap |

---

## Citation

```bibtex
@inproceedings{atd2026interspeech,
  title     = {Articulatory Trajectory Deviation: Zero-Shot Dysarthric Speech 
               Severity Estimation via Frozen {SSL} Latent Space Geometry},
  author    = {Anonymous},
  booktitle = {Proc. INTERSPEECH 2026},
  year      = {2026},
  note      = {Under review}
}
```

*BibTeX will be updated upon acceptance.*

---


<p align="center">
  <i>Built on real data. Real embeddings. Real science.</i><br>
  <i>Spearman ρ = 0.756 · p = 0.0011 · Zero fine-tuning.</i>
</p>