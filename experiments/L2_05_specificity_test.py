"""ATD specificity test on L2-ARCTIC.

Core question: does ATD pick up accent-driven pronunciation deviation, or
is it dysarthric-temporal-disruption-specific?

Tests:
  1. Spearman + Pearson corr(speaker_mean_atd, speaker_PER) across n=24 L2
     speakers. Significant positive rho => ATD is generic deviation
     detector (bad for our specificity claim). NS or negative => ATD
     specific to dysarthric profile (good).
  2. Kruskal-Wallis H-test of mean_ATD across 6 L1 groups. If significant
     => ATD distinguishes accents (bad). NS => doesn't (good).
  3. Effect-size comparison: L2-ARCTIC ATD distribution vs TORGO control
     distribution vs TORGO dysarthric distribution. Cohen's d for each
     pair. Want d(L2 vs TORGO_ctrl) small + d(TORGO_dys vs everything)
     large.

Output:
  results/l2arctic_specificity.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal, mannwhitneyu

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def main() -> None:
    spk_df = pd.read_csv(PROJECT_ROOT / "results/l2arctic_speaker_atd.csv")
    logger.info("L2-ARCTIC speakers: %d", len(spk_df))

    # Test 1: corr(ATD, PER)
    rho_atd_per, p_atd_per = spearmanr(spk_df["mean_atd"], spk_df["per"])
    r_atd_per, rp_atd_per = pearsonr(spk_df["mean_atd"], spk_df["per"])

    # Test 2: Kruskal-Wallis across L1
    groups = [spk_df[spk_df["l1"] == g]["mean_atd"].to_numpy() for g in spk_df["l1"].unique()]
    H_stat, p_kw = kruskal(*groups)

    l1_means = spk_df.groupby("l1")["mean_atd"].agg(["mean", "std", "count"]).to_dict("index")

    # Test 3: cross-corpus effect sizes
    block = {}
    torgo_spk_p = PROJECT_ROOT / "results/speaker_atd.csv"
    if torgo_spk_p.exists():
        tspk = pd.read_csv(torgo_spk_p)
        ctrl = tspk[tspk["severity_score"] == 0]["mean_atd"].to_numpy()
        dys = tspk[tspk["severity_score"] > 0]["mean_atd"].to_numpy()
        l2 = spk_df["mean_atd"].to_numpy()
        block = {
            "torgo_control_n": int(len(ctrl)),
            "torgo_control_mean": float(ctrl.mean()) if len(ctrl) else None,
            "torgo_dys_n": int(len(dys)),
            "torgo_dys_mean": float(dys.mean()) if len(dys) else None,
            "l2_arctic_n": int(len(l2)),
            "l2_arctic_mean": float(l2.mean()),
            "cohen_d_l2_vs_torgo_ctrl": _cohens_d(l2, ctrl) if len(ctrl) else None,
            "cohen_d_l2_vs_torgo_dys": _cohens_d(l2, dys) if len(dys) else None,
            "cohen_d_torgo_ctrl_vs_dys": _cohens_d(ctrl, dys) if len(ctrl) and len(dys) else None,
        }
        # Mann-Whitney
        if len(ctrl):
            mu1, mp1 = mannwhitneyu(l2, ctrl, alternative="two-sided")
            block["mw_l2_vs_torgo_ctrl_p"] = float(mp1)
        if len(dys):
            mu2, mp2 = mannwhitneyu(l2, dys, alternative="two-sided")
            block["mw_l2_vs_torgo_dys_p"] = float(mp2)
    else:
        logger.warning("results/speaker_atd.csv missing — skipping cross-corpus block")

    out = {
        "n_l2_speakers": int(len(spk_df)),
        "test_1_corr_atd_vs_per": {
            "spearman_rho": float(rho_atd_per), "spearman_p": float(p_atd_per),
            "pearson_r": float(r_atd_per), "pearson_p": float(rp_atd_per),
            "interpretation": (
                "H0: ATD independent of per-speaker phone-error-rate (accent strength). "
                "Significant rho => ATD is generic deviation detector. "
                "NS rho => specific to dysarthric temporal disruption (desired)."
            ),
        },
        "test_2_kruskal_wallis_across_l1": {
            "H": float(H_stat), "p": float(p_kw),
            "n_groups": int(spk_df["l1"].nunique()),
            "group_means": {k: {kk: float(vv) if not pd.isna(vv) else None for kk, vv in v.items()}
                            for k, v in l1_means.items()},
            "interpretation": (
                "H0: mean ATD same across 6 L1 groups (Arabic, Chinese, Hindi, "
                "Korean, Spanish, Vietnamese). NS => ATD does not distinguish "
                "accents (desired)."
            ),
        },
        "test_3_cross_corpus_effect_sizes": block,
        "summary_table": spk_df[["speaker_id", "l1", "gender", "n_utts", "mean_atd", "per"]]
            .sort_values("per").to_dict(orient="records"),
    }
    out_path = PROJECT_ROOT / "results/l2arctic_specificity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "summary_table"}, indent=2))


if __name__ == "__main__":
    main()
