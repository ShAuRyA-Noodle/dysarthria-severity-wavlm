# Articulatory Trajectory Deviation (ATD)

A novel zero-shot metric for dysarthric speech severity estimation using frozen SSL speech model latent spaces.

## Quick Start

```batch
REM 1. Set up conda environment
setup_environment.bat

REM 2. Activate environment
conda activate dysarthria

REM 3. Verify everything works
python experiments/01_verify_environment.py

REM 4. Proceed with experiments in order
python experiments/02_explore_torgo.py
```

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `src/` | Core library modules |
| `experiments/` | Numbered experiment scripts (run in order) |
| `data/` | Raw, processed, embeddings, splits |
| `results/` | Figures, tables, logs |
| `paper/` | Manuscript sections and assets |

## Key References

- **Dataset:** TORGO (15 speakers: 8 dysarthric + 7 controls)
- **Model:** WavLM-Large (frozen, zero-shot)
- **Target:** INTERSPEECH 2026 / ICASSP 2027
