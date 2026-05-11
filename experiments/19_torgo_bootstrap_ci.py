"""10k-resample paired bootstrap CI for the TORGO headline speaker-level rho.

Adds the bootstrap CI to paper_numbers.json under
``atd_validation.bootstrap_ci_10k`` and writes a histogram of the
resampled rho distribution.
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

from src.constants import RANDOM_SEED
from src.seed_utils import set_global_seed


def main() -> None:
    set_global_seed(RANDOM_SEED)
    n_resamples = 10000

    spk = pd.read_csv(PROJECT_ROOT / "results/speaker_atd.csv")
    sev = spk["severity_score"].to_numpy()
    atd = spk["mean_atd"].to_numpy()
    point_rho, point_p = spearmanr(sev, atd)

    rng = np.random.default_rng(RANDOM_SEED)
    rhos = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(sev), size=len(sev))
        r, _ = spearmanr(sev[idx], atd[idx])
        rhos[i] = r if np.isfinite(r) else 0.0
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    median = float(np.median(rhos))
    pct_above_zero = float(np.mean(rhos > 0))

    out = {
        "n_speakers": int(len(spk)),
        "n_resamples": n_resamples,
        "point_rho": float(point_rho),
        "point_p_value": float(point_p),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "bootstrap_median_rho": median,
        "bootstrap_pct_above_zero": pct_above_zero,
    }
    out_path = PROJECT_ROOT / "results/torgo_bootstrap_ci.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 3.4))
    ax.hist(rhos, bins=40, color="tab:blue", alpha=0.7, edgecolor="white")
    ax.axvline(point_rho, color="k", linestyle="--", linewidth=1,
               label=f"point rho = {point_rho:.3f}")
    ax.axvline(lo, color="tab:red", linestyle=":", linewidth=1, label=f"95% CI low = {lo:.3f}")
    ax.axvline(hi, color="tab:red", linestyle=":", linewidth=1, label=f"95% CI high = {hi:.3f}")
    ax.set_xlabel(r"Bootstrap Spearman $\rho$")
    ax.set_ylabel("Count (n=10000 resamples)")
    ax.set_title("TORGO headline rho — 10k paired bootstrap")
    ax.legend(loc="upper left")
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(PROJECT_ROOT / f"results/figures/fig_torgo_bootstrap{ext}", dpi=180)
    plt.close(fig)

    print(f"point rho={point_rho:.4f}  p={point_p:.4f}")
    print(f"10k bootstrap 95% CI: [{lo:.4f}, {hi:.4f}]")
    print(f"median={median:.4f}  pct above zero={pct_above_zero:.4f}")


if __name__ == "__main__":
    main()
