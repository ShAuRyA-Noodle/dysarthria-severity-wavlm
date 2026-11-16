"""Track D — Layer sweep across corpora.

For each corpus and SSL model, sweeps transformer layers and reports
within-corpus Spearman rho. Used to verify that the optimal layer is
stable across languages/corpora (or to characterise the layer drift if
not).

Expected output: results/cross_corpus/layer_sweep_<model>.csv with columns
{corpus, language, model, layer, n_speakers, rho_within, p_value}.
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


def _per_layer_speaker_atd_paths(corpus: str, layer: int, results_dir: Path) -> Path | None:
    """Locate per-corpus per-layer speaker-ATD CSV."""
    candidates = [
        results_dir / "cross_corpus" / f"{corpus}_speaker_atd_layer{layer}.csv",
        results_dir / f"{corpus}_speaker_atd_layer{layer}.csv",
    ]
    if corpus == "torgo":
        candidates.insert(0, results_dir / f"speaker_atd_layer{layer}.csv")
    elif corpus == "uaspeech":
        candidates.insert(0, results_dir / f"uaspeech_speaker_atd_layer{layer}.csv")
    for p in candidates:
        if p.exists():
            return p
    return None


def _coerce_severity(df: pd.DataFrame, severity_field: str) -> np.ndarray:
    for col in (severity_field, "severity_score", "intelligibility", "severity"):
        if col in df.columns:
            return df[col].astype(float).to_numpy()
    raise KeyError(f"no severity column in {list(df.columns)}")


def _coerce_atd(df: pd.DataFrame) -> np.ndarray:
    for col in ("mean_atd", "atd_score", "atd"):
        if col in df.columns:
            return df[col].astype(float).to_numpy()
    raise KeyError(f"no atd column in {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpora", default="torgo,uaspeech",
                        help="Comma-separated corpus names.")
    parser.add_argument("--layers", default="1,3,6,9,12,15,18,21,23",
                        help="Comma-separated layer indices.")
    parser.add_argument("--model", default="wavlm_large")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    results_dir = PROJECT_ROOT / "results"
    layers = [int(x) for x in args.layers.split(",")]
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]

    rows: list[dict] = []
    for corpus in corpora:
        if corpus not in REGISTRY:
            logger.warning("skipping unknown corpus %s", corpus); continue
        cfg = get(corpus)
        for L in layers:
            path = _per_layer_speaker_atd_paths(corpus, L, results_dir)
            if path is None:
                rows.append({"corpus": corpus, "language": cfg.language, "model": args.model,
                             "layer": L, "n_speakers": 0, "rho_within": None,
                             "p_value": None, "status": "missing"})
                continue
            df = pd.read_csv(path)
            try:
                sev = _coerce_severity(df, cfg.severity_field)
                atd = _coerce_atd(df)
            except KeyError as exc:
                rows.append({"corpus": corpus, "language": cfg.language, "model": args.model,
                             "layer": L, "n_speakers": 0, "rho_within": None,
                             "p_value": None, "status": f"schema: {exc}"})
                continue
            rho, p = spearmanr(sev, atd)
            rows.append({
                "corpus": corpus, "language": cfg.language, "model": args.model,
                "layer": L, "n_speakers": int(len(df)),
                "rho_within": float(rho), "p_value": float(p), "status": "ok",
            })

    df_out = pd.DataFrame(rows)
    out_path = Path(args.out) if args.out else (
        PROJECT_ROOT / "results" / "cross_corpus" / f"layer_sweep_{args.model}.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)
    logger.info("wrote %s rows=%d", out_path, len(df_out))

    # Summary: best layer per corpus.
    print("\nbest layer per corpus:")
    for corpus, sub in df_out.dropna(subset=["rho_within"]).groupby("corpus"):
        best = sub.loc[sub["rho_within"].idxmax()]
        print(f"  {corpus}: layer={int(best.layer)} rho={best.rho_within:.4f} p={best.p_value:.4f}")


if __name__ == "__main__":
    main()
