"""Frozen WavLM-Large layer-9 mean-pool features for the severity harness.

One utterance -> one 1024-d vector = time-mean of WavLM-Large transformer layer 9
(the project's headline analysis layer). Speaker features are the mean over that
speaker's utterance vectors (done at eval time).

Reuse / cost controls:
  * TORGO reuses the already-validated cache
    ``data/processed/features/torgo_utt_mean_l9.npy`` (no re-extraction).
  * Every other corpus is extracted once and cached per corpus at
    ``data/veritas/features/<corpus>_utt_l9.npz``; re-runs load the cache and
    only extract utterances not yet present.
  * Long clips (IPVS read passages up to ~70 s) are mean-pooled in fixed windows
    with frame-count weighting, so memory stays bounded and the result equals a
    single-pass time-mean up to chunk-boundary framing.

Model loads are OFFLINE (importing ``src.veritas`` activates the HF-cache
bootstrap). Nothing is mocked; features come from real audio only.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.veritas.corpora.base import PROJECT_ROOT, UtteranceRecord

FEATURE_LAYER = 9
FEATURE_CACHE_DIR = PROJECT_ROOT / "data" / "veritas" / "features"

# Legacy TORGO cache (utterance-level L9 mean-pool) — reused as-is.
_TORGO_NPY = PROJECT_ROOT / "data" / "processed" / "features" / "torgo_utt_mean_l9.npy"
_TORGO_META = PROJECT_ROOT / "data" / "processed" / "features" / "torgo_utt_mean_l9_meta.csv"


def _load_audio_mono16k(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


def _meanpool_utterance(extractor, audio: np.ndarray, sr: int, chunk_s: float = 20.0) -> np.ndarray:
    """Time-mean of WavLM-L9 frames, computed in memory-safe windows."""
    step = int(chunk_s * sr)
    if len(audio) <= step:
        feats = extractor.extract(audio, sr)  # (T, 1024)
        return feats.mean(axis=0).astype(np.float32)
    acc = np.zeros(1024, dtype=np.float64)
    n_frames = 0
    for i in range(0, len(audio), step):
        seg = audio[i:i + step]
        if len(seg) < int(0.2 * sr):  # skip a tiny trailing sliver
            continue
        feats = extractor.extract(seg, sr)
        acc += feats.sum(axis=0)
        n_frames += feats.shape[0]
    if n_frames == 0:  # degenerate: fall back to whole-clip
        feats = extractor.extract(audio, sr)
        return feats.mean(axis=0).astype(np.float32)
    return (acc / n_frames).astype(np.float32)


def _make_extractor(device: str):
    # Imported lazily so corpora that are fully cached never load the backbone.
    from src.veritas.wavlm_features import WavLMFeatures

    return WavLMFeatures(layer=FEATURE_LAYER, device=device)


def _records_meta(records: list[UtteranceRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "utterance_id": [r.utterance_id for r in records],
            "speaker_id": [r.speaker_id for r in records],
            "corpus": [r.corpus for r in records],
            "etiology": [r.etiology for r in records],
            "language": [r.language for r in records],
            "severity_rank": [float(r.severity_rank) for r in records],
            "is_control": [bool(r.is_control) for r in records],
        }
    )


def _torgo_features(records: list[UtteranceRecord]) -> tuple[np.ndarray, pd.DataFrame]:
    """Reuse the validated legacy TORGO L9 cache; align to records by utt id."""
    X = np.load(_TORGO_NPY)
    meta = pd.read_csv(_TORGO_META)
    idx = {u: i for i, u in enumerate(meta["utterance_id"].tolist())}
    rows, feats = [], []
    for r in records:
        j = idx.get(r.utterance_id)
        if j is None:
            continue  # utterance not in cache -> drop (never fabricate)
        feats.append(X[j])
        rows.append(r)
    Xout = np.stack(feats, axis=0).astype(np.float32)
    return Xout, _records_meta(rows)


def build_corpus_features(
    records: list[UtteranceRecord],
    device: str = "cuda:0",
    chunk_s: float = 20.0,
    log=print,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return (X (N,1024), meta) of per-utterance WavLM-L9 mean-pool features.

    Uses the corpus name from the records to pick the TORGO fast-path or a
    per-corpus npz cache. Only uncached utterances are extracted.
    """
    if not records:
        return np.zeros((0, 1024), dtype=np.float32), _records_meta([])
    corpus = records[0].corpus

    if corpus == "torgo" and _TORGO_NPY.exists() and _TORGO_META.exists():
        X, meta = _torgo_features(records)
        log(f"[features:{corpus}] reused legacy cache -> X={X.shape}")
        return X, meta

    FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = FEATURE_CACHE_DIR / f"{corpus}_utt_l9.npz"
    cached: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        cached = {u: v for u, v in zip(z["utt_ids"].tolist(), z["feats"])}
        log(f"[features:{corpus}] loaded {len(cached)} cached vectors")

    todo = [r for r in records if r.utterance_id not in cached]
    if todo:
        log(f"[features:{corpus}] extracting {len(todo)} / {len(records)} utterances on {device}")
        extractor = _make_extractor(device)
        t0 = time.time()
        n_fail = 0
        for k, r in enumerate(todo):
            try:
                audio, sr = _load_audio_mono16k(r.wav_path)
                cached[r.utterance_id] = _meanpool_utterance(extractor, audio, sr, chunk_s)
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                log(f"  [warn] {corpus} {r.utterance_id}: {e}")
            if (k + 1) % 200 == 0:
                log(f"  {corpus}: {k + 1}/{len(todo)} ({time.time() - t0:.0f}s, fail={n_fail})")
        # Persist the merged cache.
        utt_ids = list(cached.keys())
        feats = np.stack([cached[u] for u in utt_ids], axis=0).astype(np.float32)
        np.savez(cache, utt_ids=np.array(utt_ids, dtype=object), feats=feats)
        log(f"[features:{corpus}] cached {len(utt_ids)} vectors ({n_fail} failed) -> {cache.name}")

    rows, feats = [], []
    for r in records:
        v = cached.get(r.utterance_id)
        if v is None:
            continue
        feats.append(v)
        rows.append(r)
    X = np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, 1024), np.float32)
    return X, _records_meta(rows)
