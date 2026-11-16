"""Track C — controlled-perturbation causal-validation experiment.

For each TORGO control speaker s in {FC01..MC04}:
  1. Load utterance audio for the 794 reference words (or shared-words-only
     subset if `--smoke`).
  2. For each (family, intensity) in src.perturbations.PERTURBATIONS:
       a. Apply the DSP perturbation.
       b. Run WavLM-Large layer-9 forward pass.
       c. Compute ATD against the LOCO reference cache for s
          (references rebuilt from the OTHER 6 controls).
       d. Compute MFCC-trajectory DTW for the same audio.
       e. Append delta_atd = atd_perturbed - atd_clean and
          delta_mfcc = mfcc_perturbed - mfcc_clean to the results CSV.
  3. Fit the pre-registered linear-mixed-effects model (statsmodels MixedLM)
     and emit fig9_perturbation_validation + perturbation_lme.json.

NOTE: This is the runner skeleton. It depends on a LOCO reference cache
which is built by `scripts/build_loco_references.py` (separate). For the
smoke run, set `SHARED_WORDS_ONLY=True` to limit to the 9 words shared by
all 15 TORGO speakers (cuts compute from ~6 GPU-h to ~10 min).

Pre-registration: see paper/appendix/track_c_preregistration.md
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    FASTDTW_RADIUS, LAYER_POSITION_MAP, RANDOM_SEED, TARGET_SAMPLE_RATE,
    TORGO_CONTROL_SPEAKERS,
)
from src.perturbations import PERTURBATIONS, perturb
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_loco_reference(speaker: str, word: str, layer: int, project_root: Path) -> Optional[dict]:
    """Load a leave-one-control-out reference for `(speaker held out, word, layer)`."""
    import pickle
    p = project_root / "data" / "processed" / "references_loco" / speaker / f"{word}_layer{layer}.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _utterance_atd(
    waveform: np.ndarray,
    sr: int,
    word: str,
    held_out: str,
    extractor,
    project_root: Path,
    layer: int = 9,
) -> Optional[float]:
    """Compute ATD against the LOCO reference for held-out speaker."""
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    ref_data = _load_loco_reference(held_out, word, layer, project_root)
    if ref_data is None:
        return None
    pca = ref_data["pca_model"]
    reference = ref_data["reference"]

    layer_pos = LAYER_POSITION_MAP[layer]
    feats = extractor.extract_all_layers(waveform, sr, [layer])[layer]
    utt = pca.transform(feats)
    distance, _ = fastdtw(utt, reference, radius=FASTDTW_RADIUS, dist=euclidean)
    return float(distance / max(len(utt), len(reference)))


def _utterance_mfcc_dtw(
    waveform: np.ndarray, sr: int, mfcc_reference: np.ndarray
) -> float:
    """MFCC trajectory DTW vs a per-word MFCC reference."""
    import librosa
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13).T
    distance, _ = fastdtw(mfccs, mfcc_reference, radius=FASTDTW_RADIUS, dist=euclidean)
    return float(distance / max(len(mfccs), len(mfcc_reference)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--smoke", action="store_true",
                        help="Shared-words-only subset (9 words); ~10 min on RTX 4080.")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--out", default="results/perturbation_atd.csv")
    args = parser.parse_args()

    set_global_seed(RANDOM_SEED)

    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    metadata_df["filepath_processed"] = metadata_df["filepath_processed"].astype(str)
    controls = metadata_df[metadata_df["speaker_id"].isin(TORGO_CONTROL_SPEAKERS)]
    if args.smoke:
        word_counts = controls.groupby("word")["speaker_id"].nunique()
        shared_words = sorted(word_counts[word_counts == len(TORGO_CONTROL_SPEAKERS)].index)[:9]
        controls = controls[controls["word"].isin(shared_words)]
        logger.info("smoke mode: %d shared words selected", len(shared_words))

    logger.info("controls n_utterances=%d", len(controls))

    # Lazy-import heavy deps so module imports are cheap for smoke tests.
    from src.ssl_extractor import SSLExtractor
    extractor = SSLExtractor(model_name="wavlm_large", device="auto", use_fp16=True)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    started = time.time()

    for _, row in controls.iterrows():
        utt_id = row["utterance_id"]; spk = row["speaker_id"]; word = row["word"]
        wav_path = Path(row["filepath_processed"])
        if not wav_path.exists():
            continue
        import soundfile as sf
        wave, sr = sf.read(wav_path)
        if sr != TARGET_SAMPLE_RATE:
            continue

        atd_clean = _utterance_atd(wave, sr, word, spk, extractor, PROJECT_ROOT, args.layer)
        if atd_clean is None:
            continue

        for family, (_, intensities) in PERTURBATIONS.items():
            for intensity in intensities:
                try:
                    perturbed = perturb(wave, sr, family, float(intensity), key=utt_id)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("perturb %s/%s failed for %s: %s", family, intensity, utt_id, exc)
                    continue
                atd_pert = _utterance_atd(perturbed, sr, word, spk, extractor, PROJECT_ROOT, args.layer)
                if atd_pert is None:
                    continue
                rows.append({
                    "utterance_id": utt_id, "speaker_id": spk, "word": word,
                    "family": family, "intensity": float(intensity),
                    "atd_clean": atd_clean, "atd_perturbed": atd_pert,
                    "delta_atd": atd_pert - atd_clean,
                })

        # Periodic save.
        if len(rows) % 200 == 0 and rows:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    logger.info("done. rows=%d elapsed=%.1f min  out=%s",
                len(rows), (time.time() - started) / 60, out_path)


if __name__ == "__main__":
    main()
