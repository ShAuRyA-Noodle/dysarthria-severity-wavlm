"""VCTK within-corpus LOCO ATD (linear-interp ref + dtaidistance DTW)."""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dtaidistance import dtw_ndim
from sklearn.decomposition import PCA
from scipy.interpolate import interp1d
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

FEAT_ROOT = PROJECT_ROOT / "data/processed/features/vctk"
PCA_DIM = 64
DTW_WINDOW = 20
MIN_REF_SPEAKERS = 2


def _interp_to(traj: np.ndarray, T: int) -> np.ndarray:
    T_src = traj.shape[0]
    if T_src == T:
        return traj.copy()
    x_src = np.linspace(0.0, 1.0, T_src)
    x_dst = np.linspace(0.0, 1.0, T)
    f = interp1d(x_src, traj, axis=0, kind="linear",
                 bounds_error=False, fill_value="extrapolate")
    return f(x_dst).astype(np.float32)


def _dtw_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(dtw_ndim.distance_fast(a.astype(np.float64), b.astype(np.float64),
                                        window=DTW_WINDOW))


def main() -> None:
    set_global_seed(RANDOM_SEED)
    md = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_vctk.csv")
    logger.info("VCTK utts: %d", len(md))

    feats: dict[tuple[str, str], np.ndarray] = {}
    pca_train: list[np.ndarray] = []
    for _, r in tqdm(md.iterrows(), total=len(md), desc="load"):
        sid = str(r["sentence_id"])
        p = FEAT_ROOT / r["speaker_id"] / f"{sid}.npy"
        if not p.exists():
            continue
        a = np.load(p).astype(np.float32)
        feats[(r["speaker_id"], sid)] = a
        pca_train.append(a[::4])
    logger.info("loaded %d trajectories", len(feats))

    train_mat = np.concatenate(pca_train, axis=0)
    logger.info("PCA fit: %d frames x %d", *train_mat.shape)
    pca = PCA(n_components=PCA_DIM, random_state=RANDOM_SEED)
    pca.fit(train_mat)
    logger.info("PCA explained var ratio sum: %.4f", pca.explained_variance_ratio_.sum())
    proj = {k: pca.transform(v).astype(np.float32) for k, v in feats.items()}
    del feats, train_mat, pca_train

    sent_to_spks: dict[str, list[str]] = {}
    for _, r in md.iterrows():
        sid = str(r["sentence_id"])
        if (r["speaker_id"], sid) in proj:
            sent_to_spks.setdefault(sid, []).append(r["speaker_id"])

    out_rows: list[dict] = []
    t0 = time.time()
    for sid, spks in tqdm(sent_to_spks.items(), desc="ATD"):
        if len(spks) < MIN_REF_SPEAKERS + 1:
            continue
        trajs = [proj[(s, sid)] for s in spks]
        lens = [t.shape[0] for t in trajs]
        T_ref = int(np.median(lens))
        interp_trajs = [_interp_to(t, T_ref) for t in trajs]
        ref_full = np.mean(np.stack(interp_trajs, axis=0), axis=0)
        N = len(spks)
        for i, target in enumerate(spks):
            ref_loco = (N * ref_full - interp_trajs[i]) / (N - 1)
            q = proj[(target, sid)]
            T_q = q.shape[0]
            dist = _dtw_dist(q, ref_loco)
            atd = dist / max(T_q, T_ref)
            out_rows.append({
                "speaker_id": target, "sentence_id": sid,
                "n_ref_speakers": N - 1,
                "T_query": T_q, "T_ref": T_ref,
                "atd": atd,
            })
    logger.info("ATD compute: %.1fs", time.time() - t0)

    atd_df = pd.DataFrame(out_rows)
    out1 = PROJECT_ROOT / "results/vctk_atd_scores.csv"
    atd_df.to_csv(out1, index=False)
    logger.info("wrote %s (%d rows)", out1, len(atd_df))

    spk_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/vctk_speakers.csv")
    agg = atd_df.groupby("speaker_id")["atd"].agg(["mean", "median", "std", "count"]).reset_index()
    agg = agg.merge(spk_df[["speaker_id", "age", "gender", "accent", "region"]], on="speaker_id")
    agg = agg.rename(columns={"mean": "mean_atd", "median": "median_atd",
                              "std": "std_atd", "count": "n_utts"})
    out2 = PROJECT_ROOT / "results/vctk_speaker_atd.csv"
    agg.to_csv(out2, index=False)
    logger.info("wrote %s", out2)
    print(agg.groupby("accent")["mean_atd"].agg(["mean", "std", "count"]).round(4).to_string())


if __name__ == "__main__":
    main()
