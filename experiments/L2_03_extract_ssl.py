"""Extract WavLM-Large Layer-9 frame trajectories for L2-ARCTIC annotated subset.

Restricts to the 3,599 manually-annotated utterances (~150 per speaker) since
those are the ones with ground-truth phone-error rates. Each WAV is resampled
44.1 -> 16 kHz, normalised, and frame embeddings at layer 9 are saved as
float16 numpy arrays.

Output:
  data/processed/features/l2arctic/{speaker}/{sentence_id}.npy
    array of shape (T, 1024) float16

  data/processed/features/l2arctic/extraction_log.json
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio.functional as TAF
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

from src.ssl_extractor import SSLExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

OUT_ROOT = PROJECT_ROOT / "data/processed/features/l2arctic"
LAYER = 9


def _load_resample(wav_path: Path) -> np.ndarray:
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = TAF.resample(t, orig_freq=sr, new_freq=16000)
        audio = t.squeeze(0).numpy()
    # Normalise to -20 dBFS-ish equivalent for consistency.
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def main() -> None:
    set_global_seed(42)
    md_path = PROJECT_ROOT / "data/processed/metadata/metadata_l2arctic.csv"
    md = pd.read_csv(md_path)
    sub = md[md["has_annotation"]].copy()
    logger.info("annotated utts: %d / %d total", len(sub), len(md))

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    extractor = SSLExtractor(model_name="wavlm_large", device="auto", use_fp16=True)

    log: dict = {"n_ok": 0, "n_skip": 0, "n_err": 0, "errors": []}
    t0 = time.time()

    for _, row in tqdm(sub.iterrows(), total=len(sub), desc="extract L9"):
        spk = row["speaker_id"]; sid = row["sentence_id"]
        out_dir = OUT_ROOT / spk
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{sid}.npy"
        if out_path.exists() and out_path.stat().st_size > 0:
            log["n_skip"] += 1
            continue
        try:
            wav = _load_resample(PROJECT_ROOT / row["wav_path"])
            with torch.no_grad():
                feats = extractor.extract_all_layers(wav, sample_rate=16000, target_layers=[LAYER])
            arr = feats[LAYER].astype(np.float16)
            np.save(out_path, arr)
            log["n_ok"] += 1
        except Exception as exc:  # noqa: BLE001
            log["n_err"] += 1
            log["errors"].append({"speaker": spk, "sid": sid, "err": str(exc)})

    log["wall_time_s"] = time.time() - t0
    (OUT_ROOT / "extraction_log.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    logger.info("ok=%d skip=%d err=%d  time=%.1fs", log["n_ok"], log["n_skip"], log["n_err"], log["wall_time_s"])


if __name__ == "__main__":
    main()
