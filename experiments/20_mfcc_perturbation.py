"""Track C MFCC-trajectory specificity comparison.

For each (speaker, word, family, intensity) row already in
results/perturbation_atd.csv, recompute the same DTW distance using
MFCC features instead of WavLM. Builds MFCC trajectory references from
TORGO controls on the same nine shared words. Outputs a paired
delta_mfcc column merged with the existing perturbation results.

Closes the Track C specificity gap: relative slope ATD-vs-MFCC per
family. A higher MFCC slope than ATD on rate / noise + similar slope
on formant / mute would constitute the specificity evidence.

Run::

    python experiments/20_mfcc_perturbation.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    FASTDTW_RADIUS, RANDOM_SEED, TARGET_SAMPLE_RATE,
    TORGO_CONTROL_SPEAKERS,
)
from src.perturbations import PERTURBATIONS, perturb
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _build_mfcc_references(
    metadata_df: pd.DataFrame, words: list[str], project_root: Path,
) -> dict[str, np.ndarray]:
    """Build per-word MFCC reference (median-length control trajectory)."""
    refs: dict[str, list[np.ndarray]] = {}
    for _, row in metadata_df.iterrows():
        if row["speaker_id"] not in TORGO_CONTROL_SPEAKERS:
            continue
        word = row["word"]
        if word not in words:
            continue
        wav_path = Path(row["filepath_processed"])
        if not wav_path.exists():
            continue
        try:
            y, sr = librosa.load(str(wav_path), sr=TARGET_SAMPLE_RATE)
            if len(y) < 512:
                continue
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T
            refs.setdefault(word, []).append(mfccs)
        except Exception as exc:  # noqa: BLE001
            logger.debug("mfcc ref load fail %s: %s", wav_path, exc)
            continue
    out: dict[str, np.ndarray] = {}
    for word, trajs in refs.items():
        if len(trajs) < 2:
            continue
        lengths = [len(t) for t in trajs]
        out[word] = trajs[int(np.argsort(lengths)[len(lengths) // 2])]
    return out


def _utt_mfcc_dtw(wave: np.ndarray, sr: int, reference: np.ndarray) -> float:
    mfccs = librosa.feature.mfcc(y=wave.astype(np.float32), sr=sr, n_mfcc=13).T
    if len(mfccs) < 2:
        return float("nan")
    dist, _ = fastdtw(mfccs, reference, radius=FASTDTW_RADIUS, dist=euclidean)
    return float(dist / max(len(mfccs), len(reference)))


def main() -> None:
    set_global_seed(RANDOM_SEED)

    perturb_csv = PROJECT_ROOT / "results/perturbation_atd.csv"
    if not perturb_csv.exists():
        raise SystemExit("missing results/perturbation_atd.csv; run Track C smoke first.")
    pert_df = pd.read_csv(perturb_csv)
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    metadata_df["filepath_processed"] = metadata_df["filepath_processed"].astype(str)

    shared_words = sorted(pert_df["word"].unique())
    logger.info("building MFCC refs for %d shared words", len(shared_words))
    mfcc_refs = _build_mfcc_references(metadata_df, shared_words, PROJECT_ROOT)
    logger.info("got %d MFCC refs", len(mfcc_refs))

    started = time.time()
    rows: list[dict] = []
    # MFCC clean ATD per (speaker, word) cache: compute once.
    clean_cache: dict[tuple[str, str], float] = {}
    n_done = 0
    for _, r in pert_df.iterrows():
        spk = r["speaker_id"]; word = r["word"]; family = r["family"]; intensity = float(r["intensity"])
        utt_id = r["utterance_id"]
        ref = mfcc_refs.get(word)
        if ref is None:
            continue
        match = metadata_df[metadata_df["utterance_id"] == utt_id]
        if match.empty:
            continue
        wav_path = Path(match.iloc[0]["filepath_processed"])
        if not wav_path.exists():
            continue
        try:
            wave, sr = sf.read(wav_path)
            if wave.ndim == 2:
                wave = wave.mean(axis=1)
            wave = wave.astype(np.float32)
        except Exception:  # noqa: BLE001
            continue

        cache_key = (spk, word)
        if cache_key not in clean_cache:
            clean_cache[cache_key] = _utt_mfcc_dtw(wave, int(sr), ref)
        mfcc_clean = clean_cache[cache_key]
        if not np.isfinite(mfcc_clean):
            continue

        try:
            perturbed = perturb(wave, int(sr), family, intensity, key=utt_id)
        except Exception:  # noqa: BLE001
            continue
        mfcc_pert = _utt_mfcc_dtw(perturbed, int(sr), ref)
        if not np.isfinite(mfcc_pert):
            continue
        rows.append({
            "utterance_id": utt_id, "speaker_id": spk, "word": word,
            "family": family, "intensity": intensity,
            "mfcc_clean": mfcc_clean, "mfcc_perturbed": mfcc_pert,
            "delta_mfcc": mfcc_pert - mfcc_clean,
        })
        n_done += 1
        if n_done % 500 == 0:
            logger.info("done %d rows (%.1f min)", n_done, (time.time() - started) / 60)

    df_out = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "results/perturbation_mfcc.csv"
    df_out.to_csv(out_csv, index=False)
    logger.info("wrote %s rows=%d elapsed=%.1f min", out_csv, len(df_out),
                (time.time() - started) / 60)

    # Joint slopes ATD vs MFCC per family.
    merged = pert_df.merge(df_out[["utterance_id", "family", "intensity",
                                    "delta_mfcc", "mfcc_clean", "mfcc_perturbed"]],
                           on=["utterance_id", "family", "intensity"],
                           how="inner")
    from scipy.stats import spearmanr
    summary: dict = {"n_rows": int(len(merged)), "by_family": {}}
    for fam, sub in merged.groupby("family"):
        x = sub["intensity"].to_numpy()
        if fam in ("time_stretch", "formant_shift"):
            x = np.abs(x - 1.0)
        elif fam == "pitch_shift":
            x = np.abs(x)
        elif fam == "pink_noise":
            x = -x
        rho_atd, _ = spearmanr(x, sub["delta_atd"])
        rho_mfcc, _ = spearmanr(x, sub["delta_mfcc"])
        summary["by_family"][fam] = {
            "n": int(len(sub)),
            "rho_atd": float(rho_atd),
            "rho_mfcc": float(rho_mfcc),
            "atd_minus_mfcc": float(rho_atd - rho_mfcc),
        }
    out_json = PROJECT_ROOT / "results/perturbation_atd_vs_mfcc.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary -> %s", out_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
