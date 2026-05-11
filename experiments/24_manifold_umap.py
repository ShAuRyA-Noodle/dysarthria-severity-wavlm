"""T5 — UMAP / t-SNE visualisation of WavLM frame trajectories on TORGO.

Samples K frames per (speaker, severity) bucket from layer-9 stacked
embeddings, fits a single UMAP, plots colour-coded by severity. Companion
figure for the paper's geometric thesis.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import LAYER_POSITION_MAP, RANDOM_SEED, TORGO_SEVERITY
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SEVERITY_LABEL = {0: "Control", 1: "Mild-Mod", 2: "Severe"}
SEVERITY_COLOR = {0: "#4daf4a", 1: "#ff7f00", 2: "#e41a1c"}


def main() -> None:
    set_global_seed(RANDOM_SEED)
    layer = 9
    pos = LAYER_POSITION_MAP[layer]
    frames_per_speaker = 200

    metadata = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json", encoding="utf-8") as f:
        ssl_index = json.load(f)

    rng = np.random.default_rng(RANDOM_SEED)
    frames_list: list[np.ndarray] = []
    labels: list[int] = []
    speakers: list[str] = []
    for spk in sorted(TORGO_SEVERITY):
        sub = metadata[metadata["speaker_id"] == spk]
        if sub.empty:
            continue
        utts = sub["utterance_id"].sample(min(15, len(sub)), random_state=42).tolist()
        bag: list[np.ndarray] = []
        for u in utts:
            e = ssl_index.get(u)
            if e is None or e.get("wavlm_large") is None:
                continue
            p = PROJECT_ROOT / e["wavlm_large"]
            if not p.exists():
                continue
            emb = np.load(str(p))[pos]
            bag.append(emb)
        if not bag:
            continue
        all_frames = np.concatenate(bag, axis=0)
        n = min(frames_per_speaker, len(all_frames))
        idx = rng.choice(len(all_frames), size=n, replace=False)
        frames_list.append(all_frames[idx])
        labels.extend([TORGO_SEVERITY[spk]] * n)
        speakers.extend([spk] * n)
        logger.info("speaker %s: sampled %d frames", spk, n)

    X = np.concatenate(frames_list, axis=0)
    y = np.array(labels)
    spk_arr = np.array(speakers)
    logger.info("matrix %s, severity dist=%s", X.shape, dict(zip(*np.unique(y, return_counts=True))))

    import umap  # type: ignore
    reducer = umap.UMAP(n_components=2, random_state=RANDOM_SEED, metric="cosine",
                       n_neighbors=30, min_dist=0.3)
    Z = reducer.fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    for sev_val in (0, 1, 2):
        m = y == sev_val
        ax.scatter(Z[m, 0], Z[m, 1], c=SEVERITY_COLOR[sev_val], alpha=0.35, s=8,
                   label=f"{SEVERITY_LABEL[sev_val]} (n={int(m.sum())})")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.set_title("WavLM-Large Layer 9 frame trajectories on TORGO\n(cosine metric, n_neighbors=30)")
    ax.legend(loc="best")
    fig.tight_layout()
    out_root = PROJECT_ROOT / "results/figures"
    out_root.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out_root / f"fig_manifold_umap{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)

    coords = pd.DataFrame({
        "speaker_id": spk_arr, "severity": y, "umap1": Z[:, 0], "umap2": Z[:, 1],
    })
    coords.to_csv(PROJECT_ROOT / "results/manifold_umap_coords.csv", index=False)
    logger.info("saved figures + coords")


if __name__ == "__main__":
    main()
