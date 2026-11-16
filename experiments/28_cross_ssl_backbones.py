"""T2 — Cross-SSL-backbone comparison on IPVS sustained-vowel /a/.

For each backbone in {wavlm_large, xlsr_300m, hubert_large,
wav2vec2_large_xlsr_53}:
  1. Extract layer-1 features for all IPVS sustained_vowel_a utterances
     (resampling to 16 kHz on the fly when needed).
  2. Build a per-corpus IPVS-control reference (DTW-barycentre).
  3. Compute ATD for all 99 vowel-a utterances against the reference.
  4. Aggregate to speaker, Spearman PD-vs-HC.

Reports which SSL backbone yields the strongest cross-language PD
signal on this single, well-defined task. Real models from HF, real
audio, no synthetic.
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import FASTDTW_RADIUS, RANDOM_SEED, TARGET_SAMPLE_RATE
from src.cross_corpus.multilingual_extractor import MODEL_REGISTRY, MultilingualExtractor
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BACKBONES = ["wavlm_large", "xlsr_300m", "hubert_large"]
LAYER = 1


def _load_wave(p: Path) -> np.ndarray:
    wave, sr = sf.read(p)
    if wave.ndim == 2:
        wave = wave.mean(axis=1)
    wave = wave.astype(np.float32)
    if sr != TARGET_SAMPLE_RATE:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
    return wave


def _backbone_score(backbone: str, md: pd.DataFrame) -> dict:
    """Run extract + reference + ATD + Spearman for one backbone."""
    if backbone not in MODEL_REGISTRY:
        return {"backbone": backbone, "status": "unknown"}
    started = time.time()
    extractor = MultilingualExtractor(model_name=backbone, device="auto", use_fp16=True)
    feats: dict[str, np.ndarray] = {}
    for _, r in md.iterrows():
        p = Path(r["filepath_processed"])
        if not p.exists():
            continue
        try:
            wave = _load_wave(p)
        except Exception:  # noqa: BLE001
            continue
        out = extractor.extract_all_layers(wave, TARGET_SAMPLE_RATE, [LAYER])
        feats[str(r["utterance_id"])] = out[LAYER]
    logger.info("[%s] extracted features for %d utts in %.1f min",
                backbone, len(feats), (time.time() - started) / 60)

    # Build reference from controls only.
    control_utts = md[md["is_control"].astype(bool)]["utterance_id"].astype(str).tolist()
    control_trajs = [feats[u] for u in control_utts if u in feats]
    if len(control_trajs) < 2:
        return {"backbone": backbone, "status": "too_few_controls",
                "n_controls": len(control_trajs)}
    cat = np.concatenate(control_trajs, axis=0)
    n_comp = min(64, cat.shape[0], cat.shape[1])
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    pca.fit(cat)
    reduced = [pca.transform(t) for t in control_trajs]
    lengths = [len(t) for t in reduced]
    template = reduced[int(np.argsort(lengths)[len(lengths) // 2])]
    ref = template  # median-length trajectory as reference

    # ATD per utt + speaker aggregation.
    rows: list[dict] = []
    for _, r in md.iterrows():
        u = str(r["utterance_id"])
        if u not in feats:
            continue
        z = pca.transform(feats[u])
        dist, _ = fastdtw(z, ref, radius=FASTDTW_RADIUS, dist=euclidean)
        atd = float(dist / max(len(z), len(ref)))
        rows.append({"utterance_id": u, "speaker_id": str(r["speaker_id"]),
                     "is_control": bool(r["is_control"]), "atd": atd})
    df = pd.DataFrame(rows)
    if df.empty:
        return {"backbone": backbone, "status": "no_atd"}
    spk = df.groupby("speaker_id").agg(mean_atd=("atd", "mean"),
                                       is_control=("is_control", "first")).reset_index()
    rho, p = spearmanr((~spk["is_control"]).astype(int).to_numpy(), spk["mean_atd"].to_numpy())
    return {
        "backbone": backbone, "status": "ok",
        "n_utts": int(len(df)),
        "n_speakers": int(len(spk)),
        "n_controls": int(spk["is_control"].sum()),
        "n_pd": int((~spk["is_control"]).sum()),
        "rho": float(rho), "p_value": float(p),
        "elapsed_min": (time.time() - started) / 60,
    }


def main() -> None:
    set_global_seed(RANDOM_SEED)
    md = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_ipvs.csv")
    md = md[md["task"] == "sustained_vowel_a"].copy()
    md["is_control"] = md["is_control"].astype(bool)
    logger.info("ipvs sustained_vowel_a: %d utts  ctrl=%d", len(md), int(md["is_control"].sum()))

    results = []
    for bb in BACKBONES:
        try:
            r = _backbone_score(bb, md)
        except Exception as exc:  # noqa: BLE001
            logger.exception("backbone %s failed: %s", bb, exc)
            r = {"backbone": bb, "status": f"error: {exc}"}
        results.append(r)
        logger.info("backbone=%s result=%s", bb, r)

    out = PROJECT_ROOT / "results/cross_corpus/ssl_backbone_compare.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"task": "ipvs_sustained_vowel_a",
                   "layer": LAYER,
                   "by_backbone": results}, f, indent=2)
    print(json.dumps({"summary": [{k: r[k] for k in ("backbone", "status", "rho", "p_value")
                                  if k in r} for r in results]}, indent=2))


if __name__ == "__main__":
    main()
