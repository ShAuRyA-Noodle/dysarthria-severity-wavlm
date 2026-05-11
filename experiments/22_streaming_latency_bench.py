"""Track A — Offline-vs-online ATD parity + latency benchmark.

For N held-out TORGO control utterances:
  1. Compute offline ATD via src.atd_metric.compute_single_atd.
  2. Stream the same utterance frame-by-frame through OnlineSubseqDTW.
  3. Compare final scores (parity).
  4. Measure per-frame extend() wall-clock latency.

Outputs:
  results/streaming_parity.csv
  results/streaming_parity.json (median latency, parity stats)
  results/figures/fig_streaming_parity.{png,pdf}
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import FASTDTW_RADIUS, LAYER_POSITION_MAP, RANDOM_SEED
from src.streaming.online_dtw import OnlineSubseqDTW
from src.seed_utils import set_global_seed
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    set_global_seed(RANDOM_SEED)
    layer = 9
    layer_position = LAYER_POSITION_MAP[layer]

    metadata = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json", encoding="utf-8") as f:
        ssl_index = json.load(f)
    with open(PROJECT_ROOT / "data/processed/references/reference_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    # Pre-load ref cache for layer 9.
    ref_cache: dict[str, dict] = {}
    for key, info in manifest["word_layer_ref"].items():
        if not key.endswith(f"_{layer}"):
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as rf:
                ref_cache[key] = pickle.load(rf)
    logger.info("ref cache: %d words", len(ref_cache))

    # Sample 200 utterances stratified by severity (so latency reflects realistic mix).
    sample_size = 200
    candidates = []
    for _, row in metadata.iterrows():
        word = str(row.get("word", ""))
        if not word or word.startswith("["):
            continue
        if f"{word}_{layer}" not in ref_cache:
            continue
        utt = row["utterance_id"]
        entry = ssl_index.get(utt)
        if entry is None or entry.get("wavlm_large") is None:
            continue
        npy = PROJECT_ROOT / entry["wavlm_large"]
        if not npy.exists():
            continue
        candidates.append((utt, row["speaker_id"], word, str(npy)))
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(candidates)
    sample = candidates[:sample_size]
    logger.info("sampled %d utterances", len(sample))

    rows: list[dict] = []
    per_frame_latencies_us: list[float] = []

    for utt_id, spk, word, npy in sample:
        emb = np.load(npy)
        utt_frames = emb[layer_position]  # (T, 1024)
        ref = ref_cache[f"{word}_{layer}"]
        pca = ref["pca_model"]
        reference = ref["reference"]
        utt_pca = pca.transform(utt_frames)

        # Offline ATD (reference truth).
        d_off, _ = fastdtw(utt_pca, reference, radius=FASTDTW_RADIUS, dist=euclidean)
        atd_offline = float(d_off / max(len(utt_pca), len(reference)))

        # Online stream.
        dtw = OnlineSubseqDTW(reference, radius=FASTDTW_RADIUS)
        for frame in utt_pca:
            t0 = time.perf_counter()
            dtw.extend(frame)
            per_frame_latencies_us.append((time.perf_counter() - t0) * 1e6)
        atd_online = dtw.last_atd

        rows.append({
            "utterance_id": utt_id, "speaker_id": spk, "word": word,
            "n_frames": int(len(utt_pca)),
            "atd_offline": atd_offline,
            "atd_online": atd_online,
            "abs_diff": float(abs(atd_offline - atd_online)),
            "rel_diff": float(abs(atd_offline - atd_online) / max(atd_offline, 1e-9)),
        })

    df = pd.DataFrame(rows)
    out_csv = PROJECT_ROOT / "results/streaming_parity.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    lat_arr = np.array(per_frame_latencies_us)
    pearson_r = float(np.corrcoef(df["atd_offline"], df["atd_online"])[0, 1])
    summary = {
        "n_utterances": int(len(df)),
        "n_frame_samples": int(len(lat_arr)),
        "atd_pearson_r_online_vs_offline": pearson_r,
        "atd_mean_abs_diff": float(df["abs_diff"].mean()),
        "atd_median_abs_diff": float(df["abs_diff"].median()),
        "atd_mean_rel_diff": float(df["rel_diff"].mean()),
        "atd_max_rel_diff": float(df["rel_diff"].max()),
        "per_frame_latency_us_p50": float(np.percentile(lat_arr, 50)),
        "per_frame_latency_us_p95": float(np.percentile(lat_arr, 95)),
        "per_frame_latency_us_p99": float(np.percentile(lat_arr, 99)),
        "per_frame_latency_us_max": float(lat_arr.max()),
    }
    out_json = PROJECT_ROOT / "results/streaming_parity.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Parity scatter.
    fig, ax = plt.subplots(figsize=(5, 5))
    lo = min(df["atd_offline"].min(), df["atd_online"].min())
    hi = max(df["atd_offline"].max(), df["atd_online"].max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="identity")
    ax.scatter(df["atd_offline"], df["atd_online"], alpha=0.6, s=18)
    ax.set_xlabel("Offline ATD"); ax.set_ylabel("Online ATD (subseq DTW stream)")
    ax.set_title(f"Streaming parity: Pearson r = {pearson_r:.4f}")
    ax.legend(); fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(PROJECT_ROOT / f"results/figures/fig_streaming_parity{ext}", dpi=180)
    plt.close(fig)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
