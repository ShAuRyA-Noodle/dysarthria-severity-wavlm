"""Track D — Within-corpus ATD for IPVS / EasyCall / PC-GITA.

Pipeline:
  1. Load corpus metadata (renames `task` → `word` for IPVS so reference
     lookups work).
  2. Load corpus-internal references built by
     `scripts/build_ipvs_references.py` (or analogous for other corpora).
  3. For each utterance: compute ATD vs same-corpus per-task reference.
  4. Aggregate to speaker level and report Spearman rho vs severity.

Tests Hypothesis H1 (within-corpus rho > 0 in PD speech).
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

from src.constants import RANDOM_SEED
from src.cross_corpus.corpus_registry import REGISTRY, get
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


IPVS_LAYER_POSITION_MAP = {1: 0, 9: 1, 12: 2}


def _compute_atd_for_corpus(
    metadata_df: pd.DataFrame,
    ssl_index_path: Path,
    reference_manifest_path: Path,
    model_name: str,
    layer: int,
    layer_position: int,
    word_col: str,
) -> pd.DataFrame:
    """Compute per-utterance ATD against same-corpus references."""
    from fastdtw import fastdtw
    from scipy.spatial.distance import euclidean

    with open(ssl_index_path, encoding="utf-8") as f:
        ssl_index = json.load(f)
    with open(reference_manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    word_refs = manifest.get("word_layer_ref", {})

    # Pre-load references for the target layer only.
    import pickle
    ref_cache: dict[str, dict] = {}
    for key, info in word_refs.items():
        if not key.endswith(f"_{layer}"):
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as rf:
                ref_cache[key] = pickle.load(rf)
    logger.info("loaded %d references for layer %d", len(ref_cache), layer)

    rows = []
    for _, row in metadata_df.iterrows():
        utt_id = str(row["utterance_id"])
        spk = str(row["speaker_id"])
        word = str(row[word_col])
        entry = ssl_index.get(utt_id)
        if entry is None or entry.get(model_name) is None:
            continue
        ref_key = f"{word}_{layer}"
        ref_data = ref_cache.get(ref_key)
        if ref_data is None:
            continue
        emb_path = PROJECT_ROOT / entry[model_name]
        if not emb_path.exists():
            continue
        emb = np.load(str(emb_path))  # (n_layers, T, D)
        utt_frames = emb[layer_position]
        utt_pca = ref_data["pca_model"].transform(utt_frames)
        ref_traj = ref_data["reference"]
        distance, _ = fastdtw(utt_pca, ref_traj, dist=euclidean)
        atd = float(distance / max(len(utt_pca), len(ref_traj)))
        rows.append({
            "utterance_id": utt_id, "speaker_id": spk, "word": word,
            "atd_score": atd,
            "is_control": bool(row.get("is_control", False)),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, choices=sorted(REGISTRY.keys()))
    parser.add_argument("--task", default=None,
                        help="Optional task subset (e.g. sustained_vowel_a).")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--model", default="xlsr_300m")
    parser.add_argument("--ssl-index", default=None)
    parser.add_argument("--refs-manifest", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    cfg = get(args.corpus)
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata" / cfg.metadata_path
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata for corpus {args.corpus} at {metadata_path}")
    df = pd.read_csv(metadata_path)
    df["is_control"] = df["is_control"].astype(bool)

    if args.corpus == "ipvs":
        # IPVS uses `task` as prompt identity. Layer 9 → position 1.
        df = df.copy()
        df["word"] = df["task"]
        layer_position = IPVS_LAYER_POSITION_MAP[args.layer]
        ssl_index_path = (
            Path(args.ssl_index) if args.ssl_index
            else PROJECT_ROOT / f"data/embeddings/ssl_index_ipvs_{args.model}.json"
        )
        refs_manifest_path = (
            Path(args.refs_manifest) if args.refs_manifest
            else PROJECT_ROOT / "data/processed/references_ipvs/reference_manifest.json"
        )
    else:
        from src.atd_metric import LAYER_POSITION_MAP
        layer_position = LAYER_POSITION_MAP.get(args.layer, args.layer)
        ssl_index_path = (
            Path(args.ssl_index) if args.ssl_index
            else PROJECT_ROOT / "data/embeddings/ssl_index.json"
        )
        refs_manifest_path = (
            Path(args.refs_manifest) if args.refs_manifest
            else PROJECT_ROOT / "data/processed/references/reference_manifest.json"
        )

    if args.task:
        df = df[df.get("task", "") == args.task]

    logger.info("corpus=%s task=%s n_utts=%d controls=%d",
                args.corpus, args.task, len(df), int(df["is_control"].sum()))

    atd_df = _compute_atd_for_corpus(
        df, ssl_index_path, refs_manifest_path,
        args.model, args.layer, layer_position, "word",
    )
    n_atd = atd_df["atd_score"].notna().sum()
    logger.info("computed ATD for %d / %d utterances", n_atd, len(df))

    # Save per-utterance + speaker-level.
    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / f"results/cross_corpus/{args.corpus}_atd_layer{args.layer}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atd_df.to_csv(out_path, index=False)

    speaker_atd = atd_df.groupby("speaker_id").agg(
        mean_atd=("atd_score", "mean"),
        n_utts=("atd_score", "count"),
        is_control=("is_control", "first"),
    ).reset_index()
    spk_out = out_path.with_name(out_path.stem.replace("_atd_", "_speaker_atd_") + ".csv")
    speaker_atd.to_csv(spk_out, index=False)
    logger.info("speaker-level -> %s", spk_out)

    # Hypothesis H1 test: binary HC vs PD discrimination.
    if speaker_atd["is_control"].nunique() == 2:
        rho, p = spearmanr(
            (~speaker_atd["is_control"]).astype(int),  # 0=control, 1=PD
            speaker_atd["mean_atd"],
        )
        logger.info("H1 (binary PD vs HC): rho=%.4f  p=%.4f  n=%d",
                    rho, p, len(speaker_atd))
        # Effect size: mean-ATD gap normalised by pooled std.
        ctrl_atd = speaker_atd.loc[speaker_atd["is_control"], "mean_atd"].to_numpy()
        pd_atd = speaker_atd.loc[~speaker_atd["is_control"], "mean_atd"].to_numpy()
        if len(ctrl_atd) > 1 and len(pd_atd) > 1:
            pooled = np.sqrt(0.5 * (ctrl_atd.std(ddof=1) ** 2 + pd_atd.std(ddof=1) ** 2))
            d = (pd_atd.mean() - ctrl_atd.mean()) / pooled if pooled > 0 else 0.0
            logger.info("Cohen's d (PD - HC)/pooled_sd = %.3f", d)

    json_out = out_path.with_suffix(".json")
    summary = {
        "corpus": args.corpus,
        "task_subset": args.task,
        "layer": args.layer,
        "model": args.model,
        "n_utterances_with_atd": int(n_atd),
        "n_speakers": int(len(speaker_atd)),
        "n_controls": int(speaker_atd["is_control"].sum()),
        "n_dysarthric": int((~speaker_atd["is_control"]).sum()),
    }
    if speaker_atd["is_control"].nunique() == 2:
        summary["rho_binary"] = float(rho)
        summary["p_binary"] = float(p)
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary -> %s", json_out)


if __name__ == "__main__":
    main()
