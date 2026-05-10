"""Track D — Within-corpus ATD for IPVS / EasyCall / PC-GITA.

Each corpus contributes its own controls to its own reference set; ATD
is computed against same-corpus references so channel/recording/language
mismatch does not contaminate the metric. Tests Hypothesis H1
(within-corpus rho > 0.5, p < 0.05 in PC-GITA / IPVS).

Skeleton — fills in once IPVS reference set is built. See WORLD_CLASS_ROADMAP.md
Track D for the full architecture.
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
from src.cross_corpus.normalization import zrank_severity, zscore_atd
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True, choices=sorted(REGISTRY.keys()))
    parser.add_argument("--task", default=None,
                        help="Optional subset: 'sustained_vowels' | 'ddk' | 'commands' | ...")
    parser.add_argument("--layer", type=int, default=9)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    cfg = get(args.corpus)
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata" / cfg.metadata_path
    if not metadata_path.exists():
        raise SystemExit(f"missing metadata for corpus {args.corpus} at {metadata_path}")
    df = pd.read_csv(metadata_path)
    if args.task:
        df = df[df.get("task", "") == args.task]
        logger.info("filtered to task=%s; rows=%d", args.task, len(df))

    # NOTE: per-utterance ATD computation hooks into:
    #   src.atd_metric.compute_single_atd if model == 'wavlm_large' for English corpora,
    #   or a corpus-specific reference cache for IPVS / PC-GITA / EasyCall.
    # Expected output frame: per-utterance row with atd_score + speaker_id +
    # severity. Aggregate to speaker-level mean and report Spearman rho.

    # Placeholder aggregation pseudo-code (real wiring lands once ssl_index_<corpus>
    # exists from CC_01).
    if "atd_score" not in df.columns:
        logger.warning("no atd_score column yet; CC_02 expects CC_01 outputs and an "
                       "ATD-computation pass. Skeleton run only.")
        result = {
            "corpus": args.corpus,
            "task_subset": args.task,
            "n_utterances": len(df),
            "status": "skeleton_only",
        }
    else:
        valid = df.dropna(subset=["atd_score"])
        speaker_atd = valid.groupby("speaker_id")["atd_score"].mean().to_dict()
        sev = valid.drop_duplicates("speaker_id").set_index("speaker_id")[cfg.severity_field]
        ids = sorted(set(speaker_atd) & set(sev.index))
        rho, p = spearmanr([sev[i] for i in ids], [speaker_atd[i] for i in ids])
        result = {
            "corpus": args.corpus,
            "task_subset": args.task,
            "n_speakers": len(ids),
            "n_utterances": int(valid.shape[0]),
            "rho_within": float(rho),
            "p_value": float(p),
            "severity_field": cfg.severity_field,
            "speaker_atd_min": float(min(speaker_atd.values())) if speaker_atd else None,
            "speaker_atd_max": float(max(speaker_atd.values())) if speaker_atd else None,
        }

    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "results" / "cross_corpus" / f"within_{args.corpus}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
