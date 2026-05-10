"""Bootstrap CIs + Bonferroni correction for the IPVS vowel headline result.

Loads results/cross_corpus/ipvs_speaker_atd_vowel_<v>_layer1.csv for
v in {a, e, i, o, u}. For each vowel runs n_resamples bootstrap pairs of
(severity_label, mean_atd) and reports the 95% Spearman CI. Applies
Bonferroni correction across the 5 vowels (alpha = 0.05 / 5 = 0.01) and
Benjamini-Hochberg FDR. Writes a forest-plot of per-vowel rho + 95% CI.

Run::

    python experiments/17_ipvs_vowel_bootstrap.py
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _bootstrap_spearman_ci(
    labels: np.ndarray, scores: np.ndarray, n_resamples: int = 10000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Return (point_rho, ci_low, ci_high) via paired bootstrap."""
    rng = rng or np.random.default_rng(RANDOM_SEED)
    n = len(labels)
    point, _ = spearmanr(labels, scores)
    rhos = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(labels[idx], scores[idx])
        rhos[i] = r if np.isfinite(r) else 0.0
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment."""
    p = np.asarray(p_values)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / (np.arange(n) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.minimum(adj, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--n-resamples", type=int, default=10000)
    parser.add_argument("--out-json", default="results/cross_corpus/ipvs_vowel_bootstrap.json")
    parser.add_argument("--out-fig", default="results/figures/fig_ipvs_vowel_forest")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    rng = np.random.default_rng(RANDOM_SEED)
    vowels = ["a", "e", "i", "o", "u"]
    per_vowel = []
    for v in vowels:
        path = PROJECT_ROOT / f"results/cross_corpus/ipvs_speaker_atd_vowel_{v}_layer1.csv"
        if not path.exists():
            logger.warning("missing %s — skipping vowel %s", path, v)
            continue
        df = pd.read_csv(path)
        labels = (~df["is_control"].astype(bool)).astype(int).to_numpy()
        scores = df["mean_atd"].to_numpy()
        point, lo, hi = _bootstrap_spearman_ci(labels, scores, args.n_resamples, rng)
        _, p_raw = spearmanr(labels, scores)
        per_vowel.append({
            "vowel": v, "n": int(len(labels)),
            "rho": point, "ci_low": lo, "ci_high": hi, "p_raw": float(p_raw),
        })
        logger.info("/%s/  n=%d  rho=%.3f  95%%CI [%.3f, %.3f]  p_raw=%.4f",
                    v, len(labels), point, lo, hi, p_raw)

    p_raw = np.array([r["p_raw"] for r in per_vowel])
    p_bonf = np.minimum(p_raw * len(per_vowel), 1.0)
    p_bh = _bh_fdr(p_raw)
    for r, b, f in zip(per_vowel, p_bonf, p_bh):
        r["p_bonferroni"] = float(b)
        r["p_bh_fdr"] = float(f)

    summary = {
        "n_resamples": args.n_resamples,
        "alpha_uncorrected": 0.05,
        "alpha_bonferroni": 0.05 / len(per_vowel),
        "per_vowel": per_vowel,
    }
    out_json = PROJECT_ROOT / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary -> %s", out_json)

    # Forest plot.
    fig, ax = plt.subplots(figsize=(6, 3.4))
    labels = [f"/{r['vowel']}/  p_BH={r['p_bh_fdr']:.3f}" for r in per_vowel]
    rhos = [r["rho"] for r in per_vowel]
    los = [r["ci_low"] for r in per_vowel]
    his = [r["ci_high"] for r in per_vowel]
    y = np.arange(len(per_vowel))
    ax.errorbar(rhos, y, xerr=[np.array(rhos) - np.array(los),
                               np.array(his) - np.array(rhos)],
                fmt="o", capsize=4, color="tab:blue", markersize=7)
    ax.axvline(0, color="k", linestyle="--", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"Spearman $\rho$ (PD vs HC) with 95% bootstrap CI")
    ax.set_title("IPVS sustained-vowel ATD discrimination (XLS-R-300m, layer 1)")
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(PROJECT_ROOT / (args.out_fig + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("forest plot -> %s.{png,pdf}", args.out_fig)


if __name__ == "__main__":
    main()
