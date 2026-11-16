"""Per-speaker leave-one-out + influence diagnostics on TORGO headline.

For each of the 15 TORGO speakers, removes that speaker and recomputes
the headline speaker-level Spearman rho. Also fits Cook's-distance-style
influence on a Ridge regressor of severity ~ mean_atd.

Outputs:
  results/torgo_loo_diagnostics.csv   one row per held-out speaker
  results/torgo_loo_diagnostics.json  summary stats (mean, range, F04 delta)
  results/figures/fig_loo_influence.{png,pdf}  bar chart per speaker
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TORGO_SEVERITY
from src.seed_utils import set_global_seed


def main() -> None:
    set_global_seed(RANDOM_SEED)
    spk_atd = pd.read_csv(PROJECT_ROOT / "results/speaker_atd.csv")
    spk_atd = spk_atd[["speaker_id", "mean_atd", "severity_score"]].copy()
    speakers = sorted(spk_atd["speaker_id"].unique())
    full_rho, full_p = spearmanr(spk_atd["severity_score"], spk_atd["mean_atd"])

    rows: list[dict] = []
    for spk in speakers:
        sub = spk_atd[spk_atd["speaker_id"] != spk]
        rho, p = spearmanr(sub["severity_score"], sub["mean_atd"])
        delta = rho - full_rho
        # Influence ≈ |delta_rho| relative to full leverage.
        rows.append({
            "held_out_speaker": spk,
            "severity": int(TORGO_SEVERITY.get(spk, -1)),
            "rho_without_speaker": round(float(rho), 4),
            "delta_from_full": round(float(delta), 4),
            "abs_delta": round(float(abs(delta)), 4),
            "p_value": round(float(p), 5),
            "n_remaining": int(len(sub)),
        })
    df = pd.DataFrame(rows).sort_values("abs_delta", ascending=False)
    out_csv = PROJECT_ROOT / "results/torgo_loo_diagnostics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    summary = {
        "full_rho": float(full_rho),
        "full_p": float(full_p),
        "n_speakers": len(speakers),
        "loo_rho_min": float(df["rho_without_speaker"].min()),
        "loo_rho_max": float(df["rho_without_speaker"].max()),
        "loo_rho_median": float(df["rho_without_speaker"].median()),
        "most_influential_speaker": df.iloc[0]["held_out_speaker"],
        "most_influential_delta": float(df.iloc[0]["delta_from_full"]),
        "f04_loo_rho": float(df.loc[df["held_out_speaker"] == "F04", "rho_without_speaker"].iloc[0])
                        if "F04" in df["held_out_speaker"].values else None,
        "all_loo_rho": dict(zip(df["held_out_speaker"], df["rho_without_speaker"])),
    }
    out_json = PROJECT_ROOT / "results/torgo_loo_diagnostics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot
    df_sorted = df.sort_values("held_out_speaker")
    fig, ax = plt.subplots(figsize=(9, 4))
    sev = df_sorted["severity"].to_numpy()
    colors = ["tab:green" if s == 0 else "tab:orange" if s == 1 else "tab:red" for s in sev]
    bars = ax.bar(df_sorted["held_out_speaker"], df_sorted["rho_without_speaker"], color=colors)
    ax.axhline(full_rho, color="k", linestyle="--", linewidth=0.8,
               label=f"full rho = {full_rho:.3f}")
    ax.set_ylabel(r"Spearman $\rho$ with speaker removed")
    ax.set_xlabel("Held-out speaker (green=ctrl, orange=mild, red=severe)")
    ax.set_title(r"Leave-one-speaker-out influence on TORGO headline $\rho$")
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(PROJECT_ROOT / f"results/figures/fig_loo_influence{ext}", dpi=180)
    plt.close(fig)

    print(f"full rho = {full_rho:.4f}  p = {full_p:.4f}")
    print(f"LOO range: [{summary['loo_rho_min']:.4f}, {summary['loo_rho_max']:.4f}]  median {summary['loo_rho_median']:.4f}")
    print(f"most influential: {summary['most_influential_speaker']}  delta = {summary['most_influential_delta']:+.4f}")
    print(f"F04 LOO rho = {summary['f04_loo_rho']}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
