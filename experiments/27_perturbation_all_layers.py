"""T1 — Run perturbation smoke at every WavLM target layer.

Sequentially runs 15_perturbation_validation.py --smoke at each
extracted layer in {1, 3, 6, 9, 12, 15, 18, 21, 23} and aggregates
the per-family Spearman slope into one table. Reveals the layer-wise
specificity hierarchy: which transformer depth encodes which
acoustic-articulatory axis.

CPU and GPU time budget: 9 layers x ~10 min/layer ≈ 1.5 hr.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TARGET_LAYERS
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _summarize(csv_path: Path, layer: int) -> dict:
    df = pd.read_csv(csv_path)
    out: dict = {"layer": int(layer), "n_rows": int(len(df)), "by_family": {}}
    for fam, sub in df.groupby("family"):
        x = sub["intensity"].to_numpy()
        if fam in ("time_stretch", "formant_shift"):
            x = np.abs(x - 1.0)
        elif fam == "pitch_shift":
            x = np.abs(x)
        elif fam == "pink_noise":
            x = -x
        if len(sub) < 3:
            continue
        rho, p = spearmanr(x, sub["delta_atd"])
        out["by_family"][fam] = {
            "n": int(len(sub)),
            "rho": float(rho),
            "p": float(p),
            "mean_delta_atd": float(sub["delta_atd"].mean()),
        }
    return out


def main() -> None:
    set_global_seed(RANDOM_SEED)
    layers = list(TARGET_LAYERS)
    runner = PROJECT_ROOT / "experiments/15_perturbation_validation.py"
    if not runner.exists():
        raise SystemExit(f"missing {runner}")

    all_results: list[dict] = []
    started = time.time()
    for layer in layers:
        out_csv = PROJECT_ROOT / f"results/perturbation_atd_layer{layer}.csv"
        if out_csv.exists():
            logger.info("[skip] layer %d csv exists at %s", layer, out_csv)
        else:
            logger.info("[run ] layer %d", layer)
            cmd = [sys.executable, str(runner), "--smoke", "--layer", str(layer),
                   "--out", str(out_csv.relative_to(PROJECT_ROOT))]
            rc = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False).returncode
            if rc != 0:
                logger.error("layer %d run failed rc=%d", layer, rc)
                continue
        if out_csv.exists():
            all_results.append(_summarize(out_csv, layer))

    elapsed = (time.time() - started) / 60
    logger.info("all layers complete; elapsed=%.1f min", elapsed)

    # Save flat layer-hierarchy table.
    rows: list[dict] = []
    for r in all_results:
        for fam, s in r["by_family"].items():
            rows.append({"layer": r["layer"], "family": fam,
                         "n": s["n"], "rho": s["rho"], "p": s["p"],
                         "mean_delta_atd": s["mean_delta_atd"]})
    table = pd.DataFrame(rows)
    table.to_csv(PROJECT_ROOT / "results/perturbation_layer_hierarchy.csv", index=False)
    with open(PROJECT_ROOT / "results/perturbation_layer_hierarchy.json",
              "w", encoding="utf-8") as f:
        json.dump({"per_layer": all_results,
                   "elapsed_min": elapsed}, f, indent=2)
    logger.info("wrote per-layer hierarchy")
    print(table.pivot_table(index="layer", columns="family", values="rho").round(3).to_string())


if __name__ == "__main__":
    main()
