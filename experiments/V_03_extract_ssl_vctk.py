"""Extract WavLM L9 frame trajectories for VCTK Corpus 0.92.

Restricted to top-50 most-shared sentences across the 110 speakers, mic1
(DPA 4035 omni) only. FLAC -> 16 kHz -> WavLM-Large layer 9 frame-level
embeddings, saved as float16.

Output:
  data/processed/features/vctk/{speaker_id}/{sentence_id}.npy
  data/processed/features/vctk/extraction_log.json
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

OUT_ROOT = PROJECT_ROOT / "data/processed/features/vctk"
LAYER = 9


def _load_resample(p: Path) -> np.ndarray:
    audio, sr = sf.read(str(p), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = TAF.resample(t, orig_freq=sr, new_freq=16000)
        audio = t.squeeze(0).numpy()
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak * 0.9
    return audio


def main() -> None:
    set_global_seed(42)
    md = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_vctk.csv")
    logger.info("VCTK utterances: %d (speakers=%d, sentences=%d)",
                len(md), md["speaker_id"].nunique(), md["sentence_id"].nunique())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    extractor = SSLExtractor(model_name="wavlm_large", device="auto", use_fp16=True)

    log: dict = {"n_ok": 0, "n_skip": 0, "n_err": 0, "errors": []}
    t0 = time.time()
    for _, row in tqdm(md.iterrows(), total=len(md), desc="extract L9"):
        spk = row["speaker_id"]; sid = str(row["sentence_id"])
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
    logger.info("ok=%d skip=%d err=%d  time=%.1fs",
                log["n_ok"], log["n_skip"], log["n_err"], log["wall_time_s"])


if __name__ == "__main__":
    main()
