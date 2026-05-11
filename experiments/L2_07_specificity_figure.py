"""Figure: per-population z-ATD distribution across 4 corpora.

Box plot: TORGO ctrl, TORGO mild, TORGO severe, UA ctrl, UA mild, UA mid,
UA severe, L2-ARCTIC (accented healthy), VCTK (native-accented healthy).

Output:
  results/figures/fig_specificity_4corpus.png
  results/figures/fig_specificity_4corpus.pdf
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ORDER = [
    ("TORGO_sev0",     "TORGO\nctrl",      "tab:blue"),
    ("UASpeech_sev0",  "UA\nctrl",         "tab:blue"),
    ("VCTK",           "VCTK\nnative-acc", "tab:cyan"),
    ("L2ARCTIC",       "L2-ARCTIC\nL2-acc","tab:cyan"),
    ("TORGO_sev1",     "TORGO\nmild",      "tab:orange"),
    ("UASpeech_sev1",  "UA\nmild",         "tab:orange"),
    ("UASpeech_sev2",  "UA\nmid",          "tab:orange"),
    ("TORGO_sev2",     "TORGO\nsevere",    "tab:red"),
    ("UASpeech_sev3",  "UA\nsevere",       "tab:red"),
]


def _load_all() -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    p = PROJECT_ROOT / "results/atd_scores.csv"
    if p.exists():
        d = pd.read_csv(p).dropna(subset=["atd_score"])
        parts.append(pd.DataFrame({
            "corpus": "TORGO", "atd": d["atd_score"], "severity": d["severity_score"].astype(int),
        }))
    p = PROJECT_ROOT / "results/uaspeech_atd_scores.csv"
    if p.exists():
        d = pd.read_csv(p).dropna(subset=["atd_score"])
        parts.append(pd.DataFrame({
            "corpus": "UASpeech", "atd": d["atd_score"],
            "severity": pd.to_numeric(d["severity_proxy"], errors="coerce").fillna(0).astype(int),
        }))
    p = PROJECT_ROOT / "results/l2arctic_atd_scores.csv"
    if p.exists():
        d = pd.read_csv(p).dropna(subset=["atd"])
        parts.append(pd.DataFrame({"corpus": "L2ARCTIC", "atd": d["atd"], "severity": 0}))
    p = PROJECT_ROOT / "results/vctk_atd_scores.csv"
    if p.exists():
        d = pd.read_csv(p).dropna(subset=["atd"])
        parts.append(pd.DataFrame({"corpus": "VCTK", "atd": d["atd"], "severity": 0}))
    df = pd.concat(parts, ignore_index=True)
    df["atd_z"] = df.groupby("corpus")["atd"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0))

    def _label(r):
        if r["corpus"] in ("TORGO", "UASpeech"):
            return f"{r['corpus']}_sev{int(r['severity'])}"
        return r["corpus"]

    df["population"] = df.apply(_label, axis=1)
    return df


def main() -> None:
    df = _load_all()
    logger.info("n_obs=%d, populations=%s", len(df), df["population"].value_counts().to_dict())

    fig, ax = plt.subplots(figsize=(11, 5.2))
    data, labels, colors = [], [], []
    for pop, lab, col in ORDER:
        sub = df[df["population"] == pop]["atd_z"].to_numpy()
        if len(sub) < 5:
            continue
        data.append(sub); labels.append(lab); colors.append(col)
    bp = ax.boxplot(data, labels=labels, showfliers=False, patch_artist=True,
                    widths=0.6, medianprops={"color": "black", "linewidth": 1.5})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.55)

    ax.axhline(0, color="grey", linewidth=0.6, linestyle=":")
    ax.set_ylabel("ATD z-score (within-corpus normalised)", fontsize=11)
    ax.set_title("Per-population ATD distribution across 4 corpora\n"
                 "Healthy populations (blue + cyan) hug zero; dysarthric populations (orange + red) elevate", fontsize=11)
    ax.tick_params(axis="x", labelsize=9)

    # Annotate dysarthric region.
    ax.axhspan(0.4, 2.5, color="tab:red", alpha=0.05, zorder=0)
    ax.text(0.985, 0.96, "dysarthric region\n(z > 0.4)", transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color="tab:red",
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "tab:red", "alpha": 0.6})

    fig.tight_layout()
    out_dir = PROJECT_ROOT / "results/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".pdf"):
        fig.savefig(out_dir / f"fig_specificity_4corpus{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote results/figures/fig_specificity_4corpus.{png,pdf}")


if __name__ == "__main__":
    main()
