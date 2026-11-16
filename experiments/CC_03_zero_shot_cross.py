"""Track D — Zero-shot cross-corpus ATD: TORGO references → IPVS.

Hypothesis H3: per-corpus CMVN + shared TORGO PCA basis closes the gap
between within-corpus and cross-corpus ATD.

This script does NOT rely on within-corpus IPVS references. Instead it
projects IPVS (XLS-R-300m) features through a per-corpus CMVN + a
SharedPCABasis fit on TORGO controls, then computes DTW vs the TORGO
per-word reference set. Words are limited to the shared vowel inventory
(sustained vowels), where the reference is rebuilt on TORGO controls
producing the corresponding vowel-quality stimulus.

Since TORGO doesn't have isolated sustained-vowel stimuli matching IPVS
/a/, /e/, /i/, /o/, /u/, we approximate by selecting TORGO control
words that contain the target vowel in stressed nucleus position. Real
PC-GITA / EasyCall integration (when available) will replace this hack.

Output: results/cross_corpus/zero_shot_torgo2ipvs.json with per-vowel rho.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TORGO_CONTROL_SPEAKERS
from src.cross_corpus.normalization import (
    fit_corpus_stats, fit_shared_pca, save_basis, save_stats,
)
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


IPVS_LAYER_POSITION_MAP = {1: 0, 9: 1, 12: 2}
TORGO_LAYER_POSITION_MAP = {1: 0, 3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 18: 6, 21: 7, 23: 8}


def _gather_torgo_control_frames(
    ssl_index: dict, layer_position: int, project_root: Path,
) -> np.ndarray:
    """Concatenate TORGO control frames at the requested layer position."""
    frames = []
    for utt_id, entry in ssl_index.items():
        spk = entry.get("speaker", "")
        if spk not in TORGO_CONTROL_SPEAKERS:
            continue
        path = entry.get("wavlm_large")
        if not path:
            continue
        npy = project_root / path
        if not npy.exists():
            continue
        emb = np.load(str(npy))
        if layer_position >= emb.shape[0]:
            continue
        frames.append(emb[layer_position])
    return np.concatenate(frames, axis=0)


def _gather_ipvs_frames(
    ssl_index: dict, layer_position: int, project_root: Path, control_only: bool,
) -> tuple[np.ndarray, list[str]]:
    frames = []
    utt_ids = []
    for utt_id, entry in ssl_index.items():
        if control_only and not entry.get("is_control"):
            continue
        path = entry.get("xlsr_300m")
        if not path:
            continue
        npy = project_root / path
        if not npy.exists():
            continue
        emb = np.load(str(npy))
        if layer_position >= emb.shape[0]:
            continue
        frames.append(emb[layer_position])
        utt_ids.append(utt_id)
    return frames, utt_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--torgo-layer", type=int, default=1,
                        help="TORGO WavLM-Large layer to project from.")
    parser.add_argument("--ipvs-layer", type=int, default=1,
                        help="IPVS XLS-R-300m layer to project.")
    parser.add_argument("--out", default="results/cross_corpus/zero_shot_torgo2ipvs.json")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    torgo_pos = TORGO_LAYER_POSITION_MAP[args.torgo_layer]
    ipvs_pos = IPVS_LAYER_POSITION_MAP[args.ipvs_layer]

    torgo_idx_path = PROJECT_ROOT / "data/embeddings/ssl_index.json"
    ipvs_idx_path = PROJECT_ROOT / "data/embeddings/ssl_index_ipvs_xlsr_300m.json"
    if not torgo_idx_path.exists() or not ipvs_idx_path.exists():
        raise SystemExit("missing ssl_index for TORGO or IPVS")

    with open(torgo_idx_path, encoding="utf-8") as f:
        torgo_idx = json.load(f)
    with open(ipvs_idx_path, encoding="utf-8") as f:
        ipvs_idx = json.load(f)

    logger.info("gathering TORGO control frames (layer pos %d)...", torgo_pos)
    torgo_ctrl_frames = _gather_torgo_control_frames(torgo_idx, torgo_pos, PROJECT_ROOT)
    logger.info("torgo ctrl frames: %d x %d", *torgo_ctrl_frames.shape)

    # Fit donor stats + shared PCA basis from TORGO.
    torgo_stats = fit_corpus_stats(
        torgo_ctrl_frames, corpus="torgo", model="wavlm_large", layer=args.torgo_layer,
    )
    basis = fit_shared_pca(torgo_ctrl_frames, source_corpus="torgo", n_components=64)
    save_stats(torgo_stats, PROJECT_ROOT / f"data/processed/corpus_stats/torgo_l{args.torgo_layer}.pkl")
    save_basis(basis,       PROJECT_ROOT / f"data/processed/corpus_stats/shared_pca_torgo_l{args.torgo_layer}.pkl")

    # Fit IPVS-control stats (same layer position in xlsr space).
    logger.info("gathering IPVS control frames (layer pos %d)...", ipvs_pos)
    ipvs_ctrl_frames_list, _ = _gather_ipvs_frames(ipvs_idx, ipvs_pos, PROJECT_ROOT, control_only=True)
    ipvs_ctrl_frames = np.concatenate(ipvs_ctrl_frames_list, axis=0)
    logger.info("ipvs ctrl frames: %d x %d", *ipvs_ctrl_frames.shape)
    ipvs_stats = fit_corpus_stats(
        ipvs_ctrl_frames, corpus="ipvs", model="xlsr_300m", layer=args.ipvs_layer,
    )
    save_stats(ipvs_stats, PROJECT_ROOT / f"data/processed/corpus_stats/ipvs_l{args.ipvs_layer}.pkl")

    # Centroid-distance diagnostic: project IPVS controls + PD into shared
    # TORGO PCA basis after CMVN. Report rho of distance-to-control-centroid
    # vs PD/HC label as a coarse cross-corpus signal.
    logger.info("projecting all IPVS utts through shared basis...")
    all_frames_list, all_utt_ids = _gather_ipvs_frames(ipvs_idx, ipvs_pos, PROJECT_ROOT, control_only=False)
    if not all_frames_list:
        raise SystemExit("no IPVS frames")

    # Build per-utterance centroid in shared PCA space.
    utt_centroids = []
    for arr in all_frames_list:
        z = basis.transform(arr, ipvs_stats)
        utt_centroids.append(z.mean(axis=0))
    utt_centroids = np.array(utt_centroids)

    # Speaker-level centroids = mean of utt centroids per speaker.
    rows = []
    for utt_id, c in zip(all_utt_ids, utt_centroids):
        entry = ipvs_idx[utt_id]
        rows.append({"utterance_id": utt_id, "speaker_id": entry["speaker"],
                     "is_control": bool(entry.get("is_control", False)),
                     **{f"c{i}": float(v) for i, v in enumerate(c)}})
    df = pd.DataFrame(rows)
    grp = df.groupby("speaker_id")
    spk_centroids = grp[[c for c in df.columns if c.startswith("c")]].mean()
    spk_is_ctrl = grp["is_control"].first()

    # Control prototype = mean of HC speaker centroids.
    proto = spk_centroids.loc[spk_is_ctrl[spk_is_ctrl].index].mean()
    dists = np.linalg.norm(spk_centroids.values - proto.values, axis=1)
    labels = (~spk_is_ctrl.values).astype(int)  # 1 = PD, 0 = HC
    rho, p = spearmanr(labels, dists)

    out = {
        "torgo_layer": args.torgo_layer,
        "ipvs_layer": args.ipvs_layer,
        "n_speakers": int(len(spk_centroids)),
        "n_controls": int(spk_is_ctrl.sum()),
        "n_dysarthric": int((~spk_is_ctrl).sum()),
        "rho_centroid_distance_vs_label": float(rho),
        "p_value": float(p),
        "method": "TORGO-fit shared PCA basis + per-corpus CMVN; speaker-level "
                  "centroid distance to HC prototype.",
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info("rho=%.4f  p=%.4f  n=%d  -> %s",
                rho, p, len(spk_centroids), out_path)


if __name__ == "__main__":
    main()
