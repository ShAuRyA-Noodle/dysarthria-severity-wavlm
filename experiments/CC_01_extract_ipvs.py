"""Track D — Extract WavLM-Large + XLS-R-300m features for IPVS.

Walks data/raw/ipvs/, builds metadata_ipvs.csv via src.cross_corpus.ipvs_parser,
then extracts both English-pretrained (WavLM-Large) and multilingual
(XLS-R-300m) features at the same layer set as TORGO. Writes
ssl_index_ipvs.json mirroring the TORGO ssl_index schema.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TARGET_LAYERS, TARGET_SAMPLE_RATE
from src.cross_corpus.corpus_registry import get as get_corpus
from src.cross_corpus.ipvs_parser import build_ipvs_metadata
from src.cross_corpus.multilingual_extractor import MultilingualExtractor
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=["wavlm_large", "xlsr_300m", "xlsr_1b"],
                        default="xlsr_300m", help="Multilingual SSL backbone.")
    parser.add_argument("--layers", default="1,9,12",
                        help="Comma-separated transformer layer indices.")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    cfg = get_corpus("ipvs")
    raw_dir = PROJECT_ROOT / "data" / "raw" / cfg.raw_subdir
    if not raw_dir.exists():
        raise SystemExit(f"IPVS not found at {raw_dir}; run download first.")

    # 1. Metadata.
    metadata_path = PROJECT_ROOT / "data" / "processed" / "metadata" / cfg.metadata_path
    logger.info("building metadata at %s", metadata_path)
    df = build_ipvs_metadata(raw_dir, metadata_path)
    logger.info("ipvs metadata rows=%d speakers=%d", len(df), df["speaker_id"].nunique())

    # 2. SSL extraction.
    layers = [int(s) for s in args.layers.split(",")]
    extractor = MultilingualExtractor(model_name=args.model, device="auto", use_fp16=True)
    out_dir = PROJECT_ROOT / "data" / "embeddings" / args.model / "ipvs"
    out_dir.mkdir(parents=True, exist_ok=True)

    index: dict[str, dict] = {}
    started = time.time()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract ipvs"):
        wav_path = Path(row["filepath_processed"])
        try:
            wave, sr = sf.read(wav_path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("read failed %s: %s", wav_path, exc)
            continue
        if wave.ndim == 2:
            wave = wave.mean(axis=1)
        wave = wave.astype(np.float32)
        if sr != TARGET_SAMPLE_RATE:
            # Skip resampling here — assume IPVS already 16 kHz; if not, log + skip.
            logger.debug("skip non-16k file %s (%d Hz)", wav_path, sr)
            continue

        feats = extractor.extract_all_layers(wave, sr, target_layers=layers)
        stacked = np.stack([feats[L] for L in layers], axis=0)
        speaker_dir = out_dir / row["speaker_id"]
        speaker_dir.mkdir(parents=True, exist_ok=True)
        npy_path = speaker_dir / f"{row['utterance_id']}.npy"
        np.save(str(npy_path), stacked)

        index[row["utterance_id"]] = {
            "speaker": row["speaker_id"],
            "task": row.get("task", ""),
            "is_control": bool(row.get("is_control", False)),
            "shape": list(stacked.shape),
            "layer_indices": layers,
            "model": args.model,
            args.model: str(npy_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        }

    index_path = PROJECT_ROOT / "data" / "embeddings" / f"ssl_index_ipvs_{args.model}.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    logger.info("done. n=%d elapsed=%.1f min  index=%s",
                len(index), (time.time() - started) / 60, index_path)


if __name__ == "__main__":
    main()
