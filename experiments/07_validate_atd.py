"""
Experiment 07 — Deep ATD validation with diagnostic plots.

Generates 3 publication-quality figures:
1. ATD vs Severity scatter (speaker level)
2. ATD distribution by severity (violin + box)
3. Per-speaker ATD profile (bar chart)

Usage:
    python experiments/07_validate_atd.py
"""

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

TORGO_GENDER = {
    "F01": "F", "F03": "F", "F04": "F",
    "M01": "M", "M02": "M", "M03": "M", "M04": "M", "M05": "M",
    "FC01": "F", "FC02": "F", "FC03": "F",
    "MC01": "M", "MC02": "M", "MC03": "M", "MC04": "M",
}

SEV_LABELS = {0: "Control", 1: "Mild-Mod", 2: "Severe"}
SEV_COLORS = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}


def plot_atd_vs_severity(speaker_df: pd.DataFrame) -> Path:
    """Plot 1: ATD vs severity scatter at speaker level."""
    fig, ax = plt.subplots(figsize=(8, 6))

    speaker_df = speaker_df.copy()
    speaker_df["gender"] = speaker_df["speaker_id"].map(TORGO_GENDER)
    speaker_df["sev_label"] = speaker_df["severity_score"].map(SEV_LABELS)

    # Jitter for x
    np.random.seed(42)
    jitter = np.random.uniform(-0.15, 0.15, len(speaker_df))
    x_jittered = speaker_df["severity_score"] + jitter

    # Color by gender
    colors_gender = {"F": "#e74c3c", "M": "#3498db"}
    for gender, group in speaker_df.groupby("gender"):
        mask = speaker_df["gender"] == gender
        ax.scatter(
            x_jittered[mask], speaker_df.loc[mask, "mean_atd"],
            c=colors_gender[gender], label=f"{gender}",
            s=100, edgecolors="black", linewidth=0.5, zorder=5, alpha=0.85,
        )

    # Add speaker labels
    for i, row in speaker_df.iterrows():
        ax.annotate(
            row["speaker_id"], (x_jittered[i], row["mean_atd"]),
            fontsize=7, ha="center", va="bottom", xytext=(0, 6),
            textcoords="offset points",
        )

    # Regression line
    rho, p_val = spearmanr(speaker_df["severity_score"], speaker_df["mean_atd"])
    z = np.polyfit(speaker_df["severity_score"], speaker_df["mean_atd"], 1)
    poly = np.poly1d(z)
    x_line = np.linspace(-0.2, 2.2, 50)
    ax.plot(x_line, poly(x_line), "k--", alpha=0.5, linewidth=1.5)

    ax.set_xlabel("Severity Level", fontsize=13)
    ax.set_ylabel("Mean ATD Score", fontsize=13)
    ax.set_title(f"ATD vs Dysarthria Severity (Speaker Level)\n"
                 f"Spearman rho = {rho:.3f}, p = {p_val:.4f}", fontsize=14)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Control (0)", "Mild-Mod (1)", "Severe (2)"])
    ax.legend(title="Gender", fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = FIGURES_DIR / "atd_vs_severity.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_atd_distribution(atd_df: pd.DataFrame) -> Path:
    """Plot 2: ATD distribution by severity (violin + box)."""
    valid = atd_df.dropna(subset=["atd_score"]).copy()
    valid["sev_label"] = valid["severity_score"].map(SEV_LABELS)

    fig, ax = plt.subplots(figsize=(9, 6))

    order = ["Control", "Mild-Mod", "Severe"]
    palette = [SEV_COLORS[0], SEV_COLORS[1], SEV_COLORS[2]]

    sns.violinplot(
        data=valid, x="sev_label", y="atd_score", order=order,
        palette=palette, inner=None, alpha=0.3, ax=ax,
    )
    sns.boxplot(
        data=valid, x="sev_label", y="atd_score", order=order,
        palette=palette, width=0.2, fliersize=2, ax=ax,
    )

    # Group stats
    for i, sev in enumerate([0, 1, 2]):
        g = valid[valid["severity_score"] == sev]["atd_score"]
        ax.text(i, g.quantile(0.95), f"n={len(g)}\nmean={g.mean():.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Severity Level", fontsize=13)
    ax.set_ylabel("ATD Score (utterance level)", fontsize=13)
    ax.set_title("ATD Distribution by Dysarthria Severity", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out = FIGURES_DIR / "atd_distribution.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def plot_speaker_profile(speaker_df: pd.DataFrame) -> Path:
    """Plot 3: Per-speaker ATD bar chart, sorted by severity."""
    df = speaker_df.copy()
    df["sev_label"] = df["severity_score"].map(SEV_LABELS)
    df = df.sort_values(["severity_score", "mean_atd"])

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [SEV_COLORS[s] for s in df["severity_score"]]
    bars = ax.bar(range(len(df)), df["mean_atd"], color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.85)

    # Error bars
    ax.errorbar(
        range(len(df)), df["mean_atd"], yerr=df["std_atd"],
        fmt="none", ecolor="black", capsize=3, alpha=0.6,
    )

    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["speaker_id"], rotation=45, ha="right", fontsize=11)
    ax.set_xlabel("Speaker", fontsize=13)
    ax.set_ylabel("Mean ATD Score", fontsize=13)
    ax.set_title("Per-Speaker ATD Profile (sorted by severity)", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=SEV_COLORS[k], label=v)
                       for k, v in SEV_LABELS.items()]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=11)

    fig.tight_layout()
    out = FIGURES_DIR / "speaker_atd_profile.png"
    fig.savefig(str(out), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


def main() -> None:
    """Generate all diagnostic plots."""
    print(f"{'='*60}")
    print(f"  ATD Deep Validation — Diagnostic Plots")
    print(f"{'='*60}\n")

    atd_path = PROJECT_ROOT / "results" / "atd_scores.csv"
    speaker_path = PROJECT_ROOT / "results" / "speaker_atd.csv"

    if not atd_path.exists() or not speaker_path.exists():
        print("Run 06_compute_atd.py first!")
        sys.exit(1)

    atd_df = pd.read_csv(atd_path)
    speaker_df = pd.read_csv(speaker_path)
    print(f"Loaded: {len(atd_df)} utterance scores, {len(speaker_df)} speakers\n")

    print("Plot 1: ATD vs Severity scatter...")
    plot_atd_vs_severity(speaker_df)

    print("Plot 2: ATD distribution by severity...")
    plot_atd_distribution(atd_df)

    print("Plot 3: Per-speaker ATD profile...")
    plot_speaker_profile(speaker_df)

    print(f"\nAll plots saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
