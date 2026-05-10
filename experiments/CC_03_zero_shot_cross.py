"""Track D — Zero-shot cross-corpus ATD: TORGO references → IPVS / UASpeech.

Tests the headline cross-corpus claim: does the SSL articulatory manifold
(learned by frozen WavLM/XLS-R on neurotypical speech) generalise to
Parkinsonian and non-English dysarthric speech when we apply per-corpus
CMVN + a shared TORGO PCA basis?

Pipeline:
  1. Compute CorpusStats over IPVS (and UASpeech) control frames.
  2. Apply CMVN(target) → re-anchor to TORGO control distribution.
  3. Project through the SHARED TORGO-fit PCA basis (donor = TORGO).
  4. DTW vs the TORGO per-word reference (only words shared cross-corpus).
  5. For sustained-vowel tasks (IPVS, PC-GITA), use vowel-specific
     references constructed from TORGO controls' sustained-vowel
     equivalents.
  6. Report rho per corpus + pooled mixed-effects.

Requires: CC_01 outputs (`ssl_index_ipvs_<model>.json`) + the TORGO
reference cache + a fitted SharedPCABasis (this script fits it on the fly).
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

from src.cross_corpus.normalization import (
    fit_corpus_stats, fit_shared_pca, normalize_for_corpus, zrank_severity, zscore_atd,
)
from src.constants import RANDOM_SEED, TORGO_CONTROL_SPEAKERS
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_control_frames(ssl_index_path: Path, project_root: Path,
                         control_filter, model_field: str, max_speakers: int = 0) -> np.ndarray:
    """Concatenate per-frame embeddings from controls in a given ssl_index."""
    with open(ssl_index_path, encoding="utf-8") as f:
        index = json.load(f)
    frames = []
    for utt_id, entry in index.items():
        if not control_filter(entry):
            continue
        path = entry.get(model_field)
        if not path:
            continue
        npy = project_root / path
        if not npy.exists():
            continue
        emb = np.load(str(npy))
        # First layer in the stack is treated as canonical for stats fitting.
        frames.append(emb[0])
    if not frames:
        raise RuntimeError(f"no control frames found in {ssl_index_path}")
    return np.concatenate(frames, axis=0)


def _ipvs_control_filter(entry: dict) -> bool:
    return bool(entry.get("is_control", False))


def _torgo_control_filter(entry: dict) -> bool:
    return entry.get("speaker", "") in TORGO_CONTROL_SPEAKERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--torgo-index",
                        default="data/embeddings/ssl_index.json")
    parser.add_argument("--ipvs-index",
                        default="data/embeddings/ssl_index_ipvs_xlsr_300m.json")
    parser.add_argument("--model", default="xlsr_300m")
    parser.add_argument("--out", default="results/cross_corpus/zero_shot_cross.json")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    torgo_idx = PROJECT_ROOT / args.torgo_index
    ipvs_idx = PROJECT_ROOT / args.ipvs_index
    if not torgo_idx.exists():
        raise SystemExit(f"missing TORGO index at {torgo_idx}")
    if not ipvs_idx.exists():
        raise SystemExit(f"missing IPVS index at {ipvs_idx}; run CC_01 first.")

    logger.info("loading TORGO control frames...")
    torgo_ctrl = _load_control_frames(torgo_idx, PROJECT_ROOT, _torgo_control_filter, "wavlm_large")
    torgo_stats = fit_corpus_stats(torgo_ctrl, "torgo", "wavlm_large", layer=1)
    logger.info("torgo control frames=%d", torgo_ctrl.shape[0])

    logger.info("fitting shared PCA basis (donor = TORGO)...")
    basis = fit_shared_pca(torgo_ctrl, source_corpus="torgo", n_components=64)

    logger.info("loading IPVS control frames...")
    ipvs_ctrl = _load_control_frames(ipvs_idx, PROJECT_ROOT, _ipvs_control_filter, args.model)
    ipvs_stats = fit_corpus_stats(ipvs_ctrl, "ipvs", args.model, layer=1)
    logger.info("ipvs control frames=%d", ipvs_ctrl.shape[0])

    # The per-utterance ATD computation against TORGO references is the
    # remaining heavy step; left as a hook for the next script in the chain.
    payload = {
        "torgo_control_frame_count": int(torgo_ctrl.shape[0]),
        "ipvs_control_frame_count": int(ipvs_ctrl.shape[0]),
        "shared_pca_donor": "torgo",
        "shared_pca_n_components": basis.n_components,
        "model": args.model,
        "status": "stats_and_basis_fit",
        "next_step": "CC_04_pooled_analysis.py",
    }
    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
