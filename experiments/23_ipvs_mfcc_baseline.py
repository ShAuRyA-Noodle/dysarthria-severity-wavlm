"""MFCC + SVR baseline on IPVS within-corpus PD vs HC discrimination.

For comparison with the ATD-based CC_02 result. Computes MFCC features
from raw audio (with on-the-fly 44.1->16 kHz resampling), aggregates
per (speaker, task) to mean MFCC vector, fits an SVR on severity_score
(binary 0/1) under LOSO-style speaker hold-out, and reports the LOSO
Spearman rho vs binary HC/PD labels. Also reports a direct Spearman of
mean-MFCC-L2-norm vs label as a no-classifier reference.
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
from scipy.stats import spearmanr
from sklearn.svm import SVR

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TARGET_SAMPLE_RATE
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _mfcc_mean(wave: np.ndarray, sr: int) -> np.ndarray:
    if sr != TARGET_SAMPLE_RATE:
        wave = librosa.resample(wave.astype(np.float32), orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
        sr = TARGET_SAMPLE_RATE
    mfcc = librosa.feature.mfcc(y=wave.astype(np.float32), sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)  # (13,)


def main() -> None:
    set_global_seed(RANDOM_SEED)
    md = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_ipvs.csv")
    eligible_tasks = (
        "sustained_vowel_a", "sustained_vowel_e", "sustained_vowel_i",
        "sustained_vowel_o", "sustained_vowel_u",
        "read_passage", "read_phrase", "ddk_baba", "ddk_dada",
    )
    md = md[md["task"].isin(eligible_tasks)].copy()
    logger.info("ipvs eligible-task utterances: %d", len(md))

    feats: dict[str, list[np.ndarray]] = {}
    speaker_label: dict[str, int] = {}
    started = time.time()
    for _, r in md.iterrows():
        path = Path(r["filepath_processed"])
        if not path.exists():
            continue
        try:
            wave, sr = sf.read(path)
        except Exception:  # noqa: BLE001
            continue
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        try:
            v = _mfcc_mean(wave, int(sr))
        except Exception:  # noqa: BLE001
            continue
        spk = str(r["speaker_id"])
        feats.setdefault(spk, []).append(v)
        speaker_label[spk] = 0 if bool(r["is_control"]) else 1
    logger.info("speakers with features: %d  elapsed=%.1f min",
                len(feats), (time.time() - started) / 60)

    speakers = sorted(feats.keys())
    X = np.stack([np.mean(feats[s], axis=0) for s in speakers])
    y = np.array([speaker_label[s] for s in speakers])

    # Baseline 1: norm of mean-MFCC vs label.
    norms = np.linalg.norm(X, axis=1)
    rho_norm, p_norm = spearmanr(y, norms)

    # Baseline 2: LOSO SVR predictions.
    preds = np.empty(len(speakers))
    for i in range(len(speakers)):
        train_idx = np.array([j for j in range(len(speakers)) if j != i])
        clf = SVR(kernel="rbf", C=1.0, gamma="scale")
        clf.fit(X[train_idx], y[train_idx])
        preds[i] = clf.predict(X[i:i+1])[0]
    rho_svr, p_svr = spearmanr(y, preds)

    summary = {
        "corpus": "ipvs",
        "n_speakers": int(len(speakers)),
        "n_hc": int((y == 0).sum()),
        "n_pd": int((y == 1).sum()),
        "mfcc_norm_rho": float(rho_norm),
        "mfcc_norm_p": float(p_norm),
        "mfcc_svr_loso_rho": float(rho_svr),
        "mfcc_svr_loso_p": float(p_svr),
    }
    out_path = PROJECT_ROOT / "results/cross_corpus/ipvs_mfcc_baseline.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
