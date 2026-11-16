"""LOSO jackknife CI on the TORGO headline rho.

Reads 15 leave-one-speaker-out rho values from results/torgo_loo_diagnostics.json
and produces a proper jackknife confidence interval that addresses the
F04-leverage objection directly. Reported alongside the 10k-bootstrap CI.

  jackknife_se = sqrt((n-1)/n * sum((r_i - r_bar)**2))
  bias_corrected_rho = n * full_rho - (n-1) * mean(r_i)
  95% jackknife CI = full_rho +/- 1.96 * jackknife_se

Also: percentile interval = [min(r_i), max(r_i)] for the most conservative
reading (worst-case-speaker-removed bracket).

Output:
  results/torgo_loso_jackknife_ci.json
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    set_global_seed(42)
    src = PROJECT_ROOT / "results/torgo_loo_diagnostics.json"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    d = json.loads(src.read_text(encoding="utf-8"))

    full_rho = float(d["full_rho"])
    full_p = float(d["full_p"])
    rhos = np.array(list(d["all_loo_rho"].values()), dtype=np.float64)
    n = int(len(rhos))
    assert n == int(d["n_speakers"]), "speaker count mismatch"

    r_bar = float(rhos.mean())
    jackknife_se = float(np.sqrt((n - 1) / n * np.sum((rhos - r_bar) ** 2)))
    z = 1.96
    jk_low = full_rho - z * jackknife_se
    jk_high = full_rho + z * jackknife_se
    bias_corrected = n * full_rho - (n - 1) * r_bar
    perc_low = float(rhos.min())
    perc_high = float(rhos.max())
    iqr_low = float(np.percentile(rhos, 25))
    iqr_high = float(np.percentile(rhos, 75))

    out = {
        "method": "leave-one-speaker-out jackknife on TORGO Spearman rho",
        "n_speakers": n,
        "full_rho": full_rho,
        "full_p": full_p,
        "loso_rhos_sorted": sorted([float(r) for r in rhos.tolist()]),
        "loso_mean": r_bar,
        "loso_median": float(np.median(rhos)),
        "jackknife_se": jackknife_se,
        "jackknife_95_ci": [float(jk_low), float(jk_high)],
        "jackknife_95_ci_width": float(jk_high - jk_low),
        "bias_corrected_rho": float(bias_corrected),
        "percentile_min_max_bracket": [perc_low, perc_high],
        "iqr_25_75": [iqr_low, iqr_high],
        "comparison_to_bootstrap": {
            "bootstrap_95_ci": [0.382, 0.921],
            "bootstrap_95_ci_width": 0.539,
            "jackknife_is_tighter": (jk_high - jk_low) < 0.539,
            "ci_width_ratio_jk_over_boot": (jk_high - jk_low) / 0.539,
        },
        "interpretation": (
            "Jackknife CI quantifies how much each speaker shifts the "
            "headline rho. The percentile bracket [min, max] shows the "
            "rho if the most-favourable or least-favourable speaker were "
            "removed. Both reported alongside the bootstrap CI to address "
            "the F04-leverage objection raised by the independent audit."
        ),
    }
    out_path = PROJECT_ROOT / "results/torgo_loso_jackknife_ci.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
