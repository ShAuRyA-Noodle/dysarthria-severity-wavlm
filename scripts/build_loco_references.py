"""Build leave-one-control-out (LOCO) reference cache for Track C.

For each held-out TORGO control speaker s in {FC01..MC04}, rebuild every
word reference using the OTHER 6 controls only. Stored under
``data/processed/references_loco/<held_out>/<word>_layer<L>.pkl``,
matching the schema produced by ``src.reference_builder.build_word_reference``.

This is required for Track C's perturbation experiment: when perturbing
speaker s's audio, the reference must be free of s's contribution to
avoid leakage of clean baseline into the metric.

Run::

    python scripts/build_loco_references.py [--layer 9] [--force]

~2.3 hours one-time on RTX 4080 (CPU-bound for DTW + PCA fit).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TARGET_LAYERS, TORGO_CONTROL_SPEAKERS
from src.reference_builder import build_all_references
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", default="9", help="Comma-separated layer indices.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if existing LOCO outputs are present.")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)
    layers = [int(s) for s in args.layers.split(",")]

    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json", encoding="utf-8") as f:
        ssl_index = json.load(f)

    out_root = PROJECT_ROOT / "data" / "processed" / "references_loco"
    out_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    for held_out in sorted(TORGO_CONTROL_SPEAKERS):
        target_dir = out_root / held_out
        if target_dir.exists() and any(target_dir.iterdir()) and not args.force:
            logger.info("[skip] LOCO refs for %s already exist at %s", held_out, target_dir)
            continue

        logger.info("[build] held-out=%s, layers=%s", held_out, layers)
        sub_meta = metadata_df.copy()
        # Mask the held-out speaker's controls so they cannot contribute.
        ctrl_mask = sub_meta["speaker_id"].isin(TORGO_CONTROL_SPEAKERS - {held_out})
        # Existing reference_builder uses metadata_df["is_control"] as the gate.
        sub_meta["is_control"] = ctrl_mask
        manifest = build_all_references(
            ssl_index=ssl_index,
            metadata_df=sub_meta,
            model_name="wavlm_large",
            target_layers=layers,
            output_dir=str(target_dir.relative_to(PROJECT_ROOT)).replace("\\", "/") + "/",
            min_control_utterances=2,
            project_root=PROJECT_ROOT,
        )
        logger.info("[done] %s: words_with_refs=%d total_refs=%d",
                    held_out, len(manifest["words_with_references"]),
                    manifest["total_references"])

    logger.info("total elapsed=%.1f min", (time.time() - started) / 60)


if __name__ == "__main__":
    main()
