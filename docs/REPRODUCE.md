# Reproducing the ATD results (Track 9)

Three reproducibility tiers depending on what you have:

## Tier 1 — paper numbers only (5 minutes, no GPU)

Just want to read the actual JSONs the paper reports?

```bash
git clone https://github.com/ShAuRyA-Noodle/ShAuRyA_Dysar.git
cd ShAuRyA_Dysar
cat results/paper_numbers.json | jq .
```

All 37 sections of `paper_numbers.json` are regenerated from the
on-disk JSON + CSV files by `experiments/21_consolidate_paper_numbers.py`.

## Tier 2 — re-run analysis from pre-extracted embeddings (30–60 minutes, CPU OK)

Pre-extracted WavLM-Large + XLS-R-300m embeddings (~66 GB + 10 GB) will be
published on a Zenodo deposit after acceptance. Once available:

```bash
# 1. Clone repo.
git clone https://github.com/ShAuRyA-Noodle/ShAuRyA_Dysar.git
cd ShAuRyA_Dysar

# 2. Build the Docker image (Python 3.11 + pinned requirements).
docker build -t atd:repro .

# 3. Fetch the published embeddings tarball to ./data/.
#    (replace with the actual Zenodo URL once released)
wget -O data.tar.gz https://zenodo.org/records/<ID>/files/atd-embeddings.tar.gz
tar -xzf data.tar.gz -C ./data/

# 4. Run the analysis-only pipeline (skips SSL extraction; recomputes ATD,
#    ablations, layer analysis, perturbations, IPVS within-corpus,
#    phoneme decomposition, all bootstrap CIs).
docker run --rm -v $PWD/data:/workspace/data -v $PWD/results:/workspace/results \
    atd:repro python reproduce_all.py --skip-extract --include-cross-corpus
```

## Tier 3 — full pipeline from raw audio (8–24 hours, GPU required)

You need:
- TORGO (gated from U. Toronto)
- UASpeech (gated from UIUC)
- IPVS (open on Hugging Face, see `docs/DATASETS.md`)
- NVIDIA GPU with ≥ 8 GB VRAM
- 200 GB disk free

```bash
# Conda environment.
conda create -n dysarthria python=3.11 -y
conda activate dysarthria
conda install pytorch==2.2.2 torchaudio==2.2.2 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -y
pip install -r requirements.txt praatio pydantic

# IPVS via HuggingFace (open, ~1.4 GB).
python -m huggingface_hub download \
    --repo-type dataset \
    --local-dir data/raw/ipvs \
    birgermoell/Italian_Parkinsons_Voice_and_Speech

# TORGO + UASpeech — request from upstream + place under data/raw/.

# Full reproduction.
python reproduce_all.py --force --include-uaspeech --include-cross-corpus
```

## Manual artifact verification (smoke)

```bash
python -m pytest tests/ -q
```

Expected: 46 tests collected (`pytest --collect-only -q`); pass/skip depends on optional extras — install requirements-core + requirements-llm + requirements-demo for the full suite.

## What is reproducible vs. not

| Artifact | Reproducible? | Note |
|---|---|---|
| TORGO speaker ρ (Layer 9 = 0.756, primary) | ✓ deterministic post-extraction | FP16 cudnn benchmark off |
| Fair-tuned baselines (LOSO) | ✓ seeded | `results/finetuned_ssl_baseline.json`: WavLM-pool Ridge ρ=0.760/RMSE=0.581, LinearSVR ρ=0.775/0.623, MFCC-RBF ρ=0.609/0.743 vs ATD ρ=0.756 (zero-shot). Trained probe matches ATD; the old ρ=−0.795 pooled-WavLM figure was an α=1 artifact and is retracted |
| Leakage control (LOCO de-leak) | ✓ | `results/deleak_report_fast.json`: L9 0.756→0.756, L1 0.775→0.687, seed 42 |
| TORGO ablation table | ✓ | fixes the 0.3944-everywhere bug; No-PCA subgroup ρ = −0.0563, see `results/tables/table2_ablations.csv` |
| TORGO bootstrap CI | ✓ | 10k resamples seeded |
| Phoneme decomposition (+ silence residual) | ✓ | requires MFA alignment first; per-utterance residual ≈ 0.70–0.76 of path cost on TORGO single-word audio |
| Track C perturbation slopes | ✓ | LOCO refs deterministic; **pre-registered specificity hypothesis falsified** (pink noise > formant shift at both layers) |
| IPVS Layer-1 vowel BH-FDR | ✓ | XLS-R-300m frozen, seeded; corpus-specific within-corpus finding, not a general layer recommendation |
| Streaming exact parity (r = 1.0) | ✓ | `results/streaming_dtw_correctness.json`; online == offline DTW, max rel err < 1e-15 |
| Streaming latency p50 (132 µs) | ⚠ machine-dependent | varies with CPU |
| SSL feature embeddings | ⚠ FP16 minor drift | pre-extracted files preferred |
