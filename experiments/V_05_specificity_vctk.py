"""VCTK accent specificity test.

Kruskal-Wallis across accent groups. H0: mean ATD same across accents.
NS = ATD does not distinguish native-English accents (desired).

Output:
  results/vctk_specificity.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    df = pd.read_csv(PROJECT_ROOT / "results/vctk_speaker_atd.csv")
    logger.info("VCTK speakers: %d", len(df))

    # Restrict to accents with >=4 speakers for KW power.
    counts = df["accent"].value_counts()
    keep = counts[counts >= 4].index.tolist()
    sub = df[df["accent"].isin(keep)].copy()
    logger.info("retained accents (n>=4): %s", keep)

    groups = [sub[sub["accent"] == a]["mean_atd"].to_numpy() for a in keep]
    H_stat, p_kw = kruskal(*groups)

    # Per-accent stats
    accent_stats = sub.groupby("accent")["mean_atd"].agg(["mean", "median", "std", "count"]).round(4)
    age_corr = float("nan"); age_p = float("nan")
    if "age" in df.columns and df["age"].notna().sum() > 10:
        valid = df.dropna(subset=["age"])
        r_age, p_age = spearmanr(valid["age"], valid["mean_atd"])
        age_corr = float(r_age); age_p = float(p_age)

    # Gender effect
    male = df[df["gender"] == "M"]["mean_atd"].to_numpy()
    female = df[df["gender"] == "F"]["mean_atd"].to_numpy()
    gender_block = None
    if len(male) > 1 and len(female) > 1:
        from scipy.stats import mannwhitneyu
        u_g, p_g = mannwhitneyu(male, female, alternative="two-sided")
        gender_block = {
            "n_male": int(len(male)), "n_female": int(len(female)),
            "male_mean": float(male.mean()), "female_mean": float(female.mean()),
            "mw_p": float(p_g),
        }

    out = {
        "n_speakers": int(len(df)),
        "n_accents_total": int(df["accent"].nunique()),
        "n_accents_retained_n_ge_4": len(keep),
        "kruskal_wallis_across_accents": {
            "H": float(H_stat), "p": float(p_kw),
            "n_groups": len(keep), "groups": keep,
            "per_accent_stats": accent_stats.to_dict("index"),
            "interpretation": (
                "H0: ATD distribution identical across native-English accents. "
                "NS p => ATD does not distinguish accents (specificity desired)."
            ),
        },
        "spearman_atd_vs_age": {"rho": age_corr, "p": age_p},
        "gender_effect": gender_block,
    }
    out_path = PROJECT_ROOT / "results/vctk_specificity.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
