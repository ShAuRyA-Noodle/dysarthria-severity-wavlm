"""Build IPVS within-corpus reference trajectories (Track D Hypothesis H1).

IPVS has no shared word inventory with TORGO. Each speaker repeats the
same prompt types (sustained vowels /a/, /e/, /i/, /o/, /u/; read passage;
DDK syllables). We treat each TASK as a "word" and build a per-task
reference trajectory from IPVS healthy controls (HC_Young + HC_Old).

Outputs land at ``data/processed/references_ipvs/`` with the same
schema as TORGO (per-task .pkl plus reference_manifest.json), so
``src.atd_metric.compute_single_atd`` works without modification.

We restrict to tasks with sufficient classified data (sustained_vowel_a,
sustained_vowel_e, read_passage) to avoid the "unknown" 642-row bucket
that needs prefix-table refinement (Track D follow-up).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED
from src.reference_builder import build_all_references
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# IPVS used layers {1, 9, 12} in CC_01 — position 0/1/2 in stacked arrays.
IPVS_LAYER_POSITION_MAP = {1: 0, 9: 1, 12: 2}

# Tasks that have a fixed semantic content shared across speakers.
# 'read_passage' is a standardized passage per speaker; include despite
# variant length. Drop 'unknown' bucket until task-prefix table refined.
ELIGIBLE_TASKS = (
    "sustained_vowel_a",
    "sustained_vowel_e",
    "sustained_vowel_i",
    "sustained_vowel_o",
    "sustained_vowel_u",
    "read_passage",
    "ddk_pataka",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--layers", default="1,9,12",
                        help="Transformer layer indices to build references for.")
    parser.add_argument("--ssl-index",
                        default="data/embeddings/ssl_index_ipvs_xlsr_300m.json")
    parser.add_argument("--model", default="xlsr_300m")
    parser.add_argument("--output-dir", default="data/processed/references_ipvs/")
    parser.add_argument("--metadata",
                        default="data/processed/metadata/metadata_ipvs.csv")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    layers = [int(x) for x in args.layers.split(",")]
    metadata_path = PROJECT_ROOT / args.metadata
    ssl_index_path = PROJECT_ROOT / args.ssl_index
    if not ssl_index_path.exists():
        raise SystemExit(f"missing IPVS ssl_index at {ssl_index_path}; run CC_01 first.")

    metadata_df = pd.read_csv(metadata_path)
    # IPVS metadata uses `task` as the prompt-identity field. The reference
    # builder expects `word`. Map directly without renaming the source CSV.
    metadata_df = metadata_df.copy()
    metadata_df["word"] = metadata_df["task"]
    metadata_df["is_control"] = metadata_df["is_control"].astype(bool)
    # Drop unmapped rows: keep only eligible tasks.
    before = len(metadata_df)
    metadata_df = metadata_df[metadata_df["task"].isin(ELIGIBLE_TASKS)]
    logger.info("metadata: filtered %d -> %d rows (eligible tasks only)",
                before, len(metadata_df))
    logger.info("controls available: %d", int(metadata_df["is_control"].sum()))
    logger.info("tasks present: %s",
                dict(metadata_df["task"].value_counts()))

    with open(ssl_index_path, encoding="utf-8") as f:
        ssl_index = json.load(f)

    manifest = build_all_references(
        ssl_index=ssl_index,
        metadata_df=metadata_df,
        model_name=args.model,
        target_layers=layers,
        output_dir=args.output_dir,
        min_control_utterances=2,
        project_root=PROJECT_ROOT,
        layer_position_map=IPVS_LAYER_POSITION_MAP,
    )

    summary_path = PROJECT_ROOT / args.output_dir / "build_summary.json"
    summary = {
        "model": args.model,
        "layers": layers,
        "tasks_eligible": list(ELIGIBLE_TASKS),
        "tasks_with_references": manifest["words_with_references"],
        "tasks_skipped": manifest["words_skipped"],
        "total_references": manifest["total_references"],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    logger.info("summary -> %s", summary_path)


if __name__ == "__main__":
    main()
