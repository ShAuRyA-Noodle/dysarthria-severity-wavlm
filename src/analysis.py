"""Analysis utilities for ATD metric — confound tests, statistical analysis."""

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)


def speaking_rate_confound_test(
    atd_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: str = "results",
    project_root: Path = None,
) -> dict:
    """
    Test whether ATD is confounded by speaking rate (duration).

    Uses duration_sec as a speaking rate proxy.
    If |Pearson r| < 0.4: ATD is NOT a proxy for speaking rate.

    Args:
        atd_df: Utterance-level ATD scores.
        metadata_df: Metadata with duration_sec.
        output_dir: Output directory for figures.
        project_root: Project root.

    Returns:
        Dict with confound analysis results.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # Merge ATD with metadata to get duration
    merged = atd_df.merge(
        metadata_df[["utterance_id", "duration_sec"]],
        on="utterance_id",
        how="left",
        suffixes=("", "_meta"),
    )

    # Drop NaN ATD and duration
    valid = merged.dropna(subset=["atd_score", "duration_sec"])
    valid = valid[valid["duration_sec"] > 0]

    print(f"  Valid utterances for confound test: {len(valid)}")

    # Compute Pearson correlation
    r, p = pearsonr(valid["atd_score"], valid["duration_sec"])
    rho, rho_p = spearmanr(valid["atd_score"], valid["duration_sec"])

    confounded = abs(r) >= 0.4
    result = {
        "atd_duration_pearson_r": round(float(r), 4),
        "atd_duration_pearson_p": round(float(p), 6),
        "atd_duration_spearman_rho": round(float(rho), 4),
        "atd_duration_spearman_p": round(float(rho_p), 6),
        "n_utterances": len(valid),
        "is_confounded": bool(confounded),
        "verdict": "ATD is confounded by duration" if confounded else "ATD is NOT a proxy for speaking rate",
    }

    # Save JSON
    json_path = project_root / output_dir / "atd_confound_analysis.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved analysis to: {json_path}")

    # Scatter plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = valid["severity_score"].map({0: "#4CAF50", 1: "#FF9800", 2: "#F44336"})
    ax.scatter(valid["duration_sec"], valid["atd_score"], c=colors, alpha=0.3, s=10)

    # Regression line
    z = np.polyfit(valid["duration_sec"], valid["atd_score"], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(valid["duration_sec"].min(), valid["duration_sec"].max(), 100)
    ax.plot(x_range, p_line(x_range), "k--", linewidth=2, alpha=0.7)

    ax.set_xlabel("Duration (sec)", fontsize=12)
    ax.set_ylabel("ATD Score", fontsize=12)
    ax.set_title(f"ATD vs Speaking Duration (Pearson r = {r:.3f})", fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", label="Control (sev=0)"),
        Patch(facecolor="#FF9800", label="Mild (sev=1)"),
        Patch(facecolor="#F44336", label="Severe (sev=2)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right")

    fig_path = project_root / output_dir / "figures" / "confound_speaking_rate.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to: {fig_path}")

    return result
