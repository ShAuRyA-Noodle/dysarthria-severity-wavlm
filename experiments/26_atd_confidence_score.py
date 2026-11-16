"""T4 — Per-utterance ATD confidence score.

Augments ATD with two confidence indicators:
  (a) Path-entropy: normalised entropy of the DTW alignment-path
      length distribution along the reference; lower = more concentrated
      alignment, higher = more spread / uncertain.
  (b) Frame coverage: fraction of reference frames the DTW path visits.

The combined uncertainty score helps SLPs prioritise ATD outputs:
high-confidence utterances (low entropy, full coverage) can be acted
on; low-confidence ones flagged for human review.

Reads results/atd_scores.csv. Computes the two metrics by replaying
the DTW alignment on a stratified subsample. Emits
results/atd_confidence.csv with per-utt {atd_score, path_entropy,
coverage_pct, confidence_combined} columns.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import FASTDTW_RADIUS, LAYER_POSITION_MAP, RANDOM_SEED
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _path_entropy(path: list[tuple[int, int]], ref_len: int) -> float:
    """Shannon entropy of the path's reference-index distribution.

    Lower = path concentrated on a few reference frames (uncertain
    alignment, query maps everything to a small region).
    Higher = path uniformly spread across the reference.
    Normalised to [0, 1] by dividing by log(ref_len).
    """
    counts = np.zeros(ref_len, dtype=np.int64)
    for _, j in path:
        if 0 <= j < ref_len:
            counts[j] += 1
    p = counts.astype(float) / counts.sum()
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p)))
    return h / np.log(ref_len) if ref_len > 1 else 0.0


def _coverage(path: list[tuple[int, int]], ref_len: int) -> float:
    visited = {j for _, j in path}
    return len(visited) / float(ref_len)


def main() -> None:
    set_global_seed(RANDOM_SEED)
    layer = 9
    pos = LAYER_POSITION_MAP[layer]

    metadata = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json", encoding="utf-8") as f:
        ssl_index = json.load(f)
    with open(PROJECT_ROOT / "data/processed/references/reference_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    ref_cache: dict[str, dict] = {}
    for key, info in manifest["word_layer_ref"].items():
        if not key.endswith(f"_{layer}"):
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as rf:
                ref_cache[key] = pickle.load(rf)
    logger.info("ref cache: %d", len(ref_cache))

    # Stratified subsample: 50 utts per severity.
    rng = np.random.default_rng(RANDOM_SEED)
    md = metadata[~metadata["word"].astype(str).str.startswith("[")].copy()
    md["severity_score"] = md["speaker_id"].map({s: int(sv) for s, sv in __import__(
        "src.constants", fromlist=["TORGO_SEVERITY"]
    ).TORGO_SEVERITY.items()})
    sample = pd.concat([
        g.sample(min(150, len(g)), random_state=42)
        for _, g in md.groupby("severity_score")
    ], ignore_index=True)
    logger.info("subsample: %d utts", len(sample))

    rows = []
    for _, r in sample.iterrows():
        utt = r["utterance_id"]; spk = r["speaker_id"]; word = str(r["word"])
        entry = ssl_index.get(utt)
        if entry is None or entry.get("wavlm_large") is None:
            continue
        ref = ref_cache.get(f"{word}_{layer}")
        if ref is None:
            continue
        npy = PROJECT_ROOT / entry["wavlm_large"]
        if not npy.exists():
            continue
        emb = np.load(str(npy))[pos]
        utt_pca = ref["pca_model"].transform(emb)
        reference = ref["reference"]
        dist, path = fastdtw(utt_pca, reference, radius=FASTDTW_RADIUS, dist=euclidean)
        atd = float(dist / max(len(utt_pca), len(reference)))
        ent = _path_entropy(path, len(reference))
        cov = _coverage(path, len(reference))
        # Combined confidence: high coverage AND moderate entropy = good.
        conf = float(cov * (1.0 - abs(ent - 0.7)))  # heuristic
        rows.append({
            "utterance_id": utt, "speaker_id": spk, "word": word,
            "severity_score": int(r["severity_score"]),
            "atd_score": atd, "path_entropy": ent, "coverage_pct": cov,
            "confidence_combined": conf,
        })

    df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "results/atd_confidence.csv"
    df.to_csv(out, index=False)
    logger.info("wrote %s rows=%d", out, len(df))
    print(df.groupby("severity_score")[["atd_score", "path_entropy", "coverage_pct",
                                         "confidence_combined"]].mean().round(4).to_string())


if __name__ == "__main__":
    main()
