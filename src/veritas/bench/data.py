"""Speaker-level data views for the validity battery (REAL features only).

A :class:`CorpusBundle` holds, for one corpus:
  * utterance-level frozen-WavLM-L9 mean-pool features (reuses the project's
    validated caches via ``src.veritas.corpora.features``) + aligned metadata,
  * one :class:`SpeakerView` per speaker carrying the pooled features, a
    silence-trimmed ("speech-only") pooled variant, mean speech energy, and the
    mean silence frame-fraction.

Two signal sources, both REAL:
  * fast path (TORGO / any corpus with cached per-frame WavLM embeddings + MFA
    TextGrids): silence fraction comes from the MFA speech span vs the stored
    frame count; the speech-only pooled feature slices those cached frames — no
    re-extraction.
  * generic path (audio only): a deterministic energy VAD defines the speech
    mask; energy is the speech-frame RMS; the speech-only pooled feature is
    re-extracted from the trimmed audio.

The cheap signals (silence fraction, energy) are cached per corpus under
``data/veritas/bench/``. The expensive speech-only pooled feature (needs an
embedding/extractor pass) is computed on demand only for bundles that a
pooled-feature metric will silence-test. Nothing is fabricated: a speaker with
no resolvable audio/features is dropped, not invented.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from src.veritas.corpora import features as feat_mod
from src.veritas.corpora.base import PROJECT_ROOT, UtteranceRecord

# WavLM-Large: 50 fps (20 ms hop); layer 9 lives at stacked-array position 3.
FPS = 50
_L9_POS = 3
_SSL_INDEX = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
_TEXTGRID_DIR = PROJECT_ROOT / "data" / "processed" / "textgrids"
_BENCH_CACHE = PROJECT_ROOT / "data" / "veritas" / "bench"


@dataclass
class SpeakerView:
    """Everything a metric or test needs about ONE speaker."""

    speaker_id: str
    corpus: str
    language: str
    severity: float
    is_control: bool
    records: list[UtteranceRecord]
    utt_ids: list[str]
    pooled_full: np.ndarray          # (n_utts, D) full-frame mean-pool
    pooled_speech: np.ndarray        # (n_utts, D) speech-only mean-pool (== full if unavailable)
    energy: float                    # mean speech log-RMS (dB)
    silence_fraction: float          # mean per-utt fraction of silence frames

    @property
    def n_utts(self) -> int:
        return len(self.utt_ids)


@dataclass
class CorpusBundle:
    """A corpus loaded for the battery: utterance features + speaker views."""

    name: str
    language: str
    views: list[SpeakerView]
    X_utt: np.ndarray                # (N, D) utterance-level pooled features
    meta: pd.DataFrame               # utterance_id, speaker_id, group, severity_rank, is_control
    silence_available: bool = True
    speech_pool_available: bool = True
    notes: list[str] = field(default_factory=list)

    @property
    def n_speakers(self) -> int:
        return len(self.views)


# ---------------------------------------------------------------------------
# index / textgrid / audio helpers
# ---------------------------------------------------------------------------
def _load_ssl_index() -> dict:
    if not _SSL_INDEX.exists():
        return {}
    with open(_SSL_INDEX, encoding="utf-8") as f:
        return json.load(f)


def _speech_span_from_textgrid(tg_path: Path, T: int) -> tuple[int, int] | None:
    """(f0, f1) speech-frame span from an MFA TextGrid, or None if no phones."""
    from src.phoneme.aligner import parse_textgrid

    phones = parse_textgrid(tg_path)  # already drops sil/sp/spn/empty
    if not phones:
        return None
    onset = min(p[1] for p in phones)
    offset = max(p[2] for p in phones)
    f0 = max(0, min(int(np.floor(onset * FPS)), T))
    f1 = max(0, min(int(np.ceil(offset * FPS)), T))
    if f1 <= f0:
        return None
    return f0, f1


def _energy_vad_mask(audio: np.ndarray, sr: int, win_s: float = 0.025,
                     hop_s: float = 0.010, rel_db: float = 30.0) -> np.ndarray:
    """Deterministic energy VAD -> per-hop speech mask (within rel_db of max)."""
    win = max(1, int(win_s * sr))
    hop = max(1, int(hop_s * sr))
    if len(audio) < win:
        return np.ones(1, dtype=bool)
    n = 1 + (len(audio) - win) // hop
    rms = np.empty(n, dtype=np.float64)
    for i in range(n):
        seg = audio[i * hop: i * hop + win]
        rms[i] = np.sqrt(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    db = 20.0 * np.log10(rms + 1e-12)
    return db >= (db.max() - rel_db)


def _rms_db(x: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    return float(20.0 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-12))


def _read_mono(path: str) -> tuple[np.ndarray, int] | None:
    p = Path(path)
    if not p.exists():
        return None
    audio, sr = sf.read(str(p), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.ascontiguousarray(audio, dtype=np.float32), int(sr)


# ---------------------------------------------------------------------------
# cheap signals: silence fraction + speech energy (cached per corpus)
# ---------------------------------------------------------------------------
def _compute_silence_energy(
    records: list[UtteranceRecord], corpus: str, log=print
) -> dict[str, dict]:
    """Per-utterance {silence_fraction, energy_db, f0, f1, T}. Cached to disk.

    Cheap: silence span from the stored frame count + MFA TextGrid (no embedding
    load); energy from the small audio file. Audio-VAD fallback when no TextGrid.
    """
    _BENCH_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = _BENCH_CACHE / f"{corpus}_silence_energy.npz"
    cached: dict[str, dict] = {}
    if cache_path.exists():
        z = np.load(cache_path, allow_pickle=True)
        cached = {u: v for u, v in zip(z["utt_ids"].tolist(), z["payload"].tolist())}
        log(f"[bench.data:{corpus}] loaded {len(cached)} cached silence/energy rows")

    todo = [r for r in records if r.utterance_id not in cached]
    if todo:
        ssl_index = _load_ssl_index()
        n_fast = n_vad = n_fail = 0
        for k, r in enumerate(todo):
            row = _sil_energy_fast(r, ssl_index)
            if row is not None:
                n_fast += 1
            else:
                row = _sil_energy_audio(r)
                if row is not None:
                    n_vad += 1
            if row is None:
                n_fail += 1
                continue
            cached[r.utterance_id] = row
            if (k + 1) % 1000 == 0:
                log(f"  {corpus} silence/energy {k + 1}/{len(todo)} "
                    f"(fast={n_fast}, vad={n_vad}, fail={n_fail})")
        utt_ids = list(cached.keys())
        payload = np.array([cached[u] for u in utt_ids], dtype=object)
        np.savez(cache_path, utt_ids=np.array(utt_ids, dtype=object), payload=payload)
        log(f"[bench.data:{corpus}] cached {len(utt_ids)} silence/energy rows "
            f"(fast={n_fast}, vad={n_vad}, fail={n_fail})")
    return cached


def _sil_energy_fast(r: UtteranceRecord, ssl_index: dict) -> dict | None:
    """Silence fraction from stored frame count + TextGrid; energy from audio."""
    entry = ssl_index.get(r.utterance_id)
    if entry is None or "shape" not in entry:
        return None
    T = int(entry["shape"][1])
    tg_path = _TEXTGRID_DIR / r.speaker_id / f"{r.utterance_id}.TextGrid"
    if not tg_path.exists():
        return None
    span = _speech_span_from_textgrid(tg_path, T)
    if span is None:
        return None
    f0, f1 = span
    sil_frac = float(1.0 - (f1 - f0) / T)
    mono = _read_mono(r.wav_path)
    if mono is None:
        return None
    audio, sr = mono
    s0, s1 = int(f0 / FPS * sr), int(f1 / FPS * sr)
    seg = audio[s0:s1] if 0 <= s0 < s1 <= len(audio) else audio
    return {"silence_fraction": sil_frac, "energy_db": _rms_db(seg),
            "f0": f0, "f1": f1, "T": T, "path": "fast"}


def _sil_energy_audio(r: UtteranceRecord) -> dict | None:
    """Generic path: energy VAD -> silence fraction + speech-frame RMS energy."""
    mono = _read_mono(r.wav_path)
    if mono is None:
        return None
    audio, sr = mono
    mask = _energy_vad_mask(audio, sr)
    hop, win = max(1, int(0.010 * sr)), max(1, int(0.025 * sr))
    keep = np.zeros(len(audio), dtype=bool)
    for i in np.where(mask)[0]:
        keep[i * hop: i * hop + win] = True
    sil_frac = float(1.0 - keep.mean()) if keep.size else float("nan")
    seg = audio[keep] if keep.any() else audio
    return {"silence_fraction": sil_frac, "energy_db": _rms_db(seg),
            "f0": -1, "f1": -1, "T": -1, "path": "vad"}


# ---------------------------------------------------------------------------
# expensive signal: speech-only pooled feature (on demand)
# ---------------------------------------------------------------------------
def _compute_speech_pool(
    records: list[UtteranceRecord], corpus: str, sil_energy: dict[str, dict],
    device: str, log=print
) -> dict[str, np.ndarray]:
    """Per-utterance speech-only mean-pool L9 feature (1024-d).

    Fast path slices cached per-frame embeddings by the MFA span; generic path
    re-extracts WavLM on the VAD-trimmed audio.
    """
    ssl_index = _load_ssl_index()
    out: dict[str, np.ndarray] = {}
    extractor = None
    n_slice = n_ext = n_fail = 0
    for r in records:
        row = sil_energy.get(r.utterance_id)
        entry = ssl_index.get(r.utterance_id)
        vec = None
        if row is not None and row.get("path") == "fast" and entry is not None \
                and entry.get("wavlm_large") is not None:
            emb_path = PROJECT_ROOT / entry["wavlm_large"]
            if emb_path.exists():
                frames = np.load(str(emb_path))[_L9_POS]
                f0, f1 = int(row["f0"]), int(row["f1"])
                seg = frames[f0:f1]
                if seg.shape[0] >= 1:
                    vec = seg.mean(axis=0).astype(np.float32)
                    n_slice += 1
        if vec is None:
            mono = _read_mono(r.wav_path)
            if mono is not None:
                if extractor is None:
                    from src.veritas.wavlm_features import WavLMFeatures
                    extractor = WavLMFeatures(layer=feat_mod.FEATURE_LAYER, device=device)
                    log(f"[bench.data:{corpus}] built WavLM extractor for speech-pool re-extraction")
                audio, sr = mono
                mask = _energy_vad_mask(audio, sr)
                hop, win = max(1, int(0.010 * sr)), max(1, int(0.025 * sr))
                keep = np.zeros(len(audio), dtype=bool)
                for i in np.where(mask)[0]:
                    keep[i * hop: i * hop + win] = True
                speech_audio = audio[keep] if keep.sum() >= win else audio
                vec = extractor.extract(speech_audio, sr).mean(axis=0).astype(np.float32)
                n_ext += 1
        if vec is None:
            n_fail += 1
            continue
        out[r.utterance_id] = vec
    log(f"[bench.data:{corpus}] speech-pool: sliced={n_slice}, re-extracted={n_ext}, "
        f"failed={n_fail}")
    return out


# ---------------------------------------------------------------------------
# bundle builder
# ---------------------------------------------------------------------------
def build_corpus_bundle(
    records: list[UtteranceRecord],
    device: str = "cuda:0",
    with_silence: bool = True,
    need_speech_pool: bool = False,
    max_utts_per_speaker: int | None = None,
    seed: int = 42,
    log=print,
) -> CorpusBundle:
    """Build a :class:`CorpusBundle` from real utterance records.

    Parameters
    ----------
    with_silence
        Compute silence fraction + speech energy (enables silence/energy tests).
    need_speech_pool
        Also compute the speech-only pooled feature (needed only when a
        pooled-feature metric will run the silence test on this bundle).
    max_utts_per_speaker
        Seeded per-speaker utterance cap (keeps the transfer/probe protocol
        comparable to the committed cross-corpus harness).
    """
    if not records:
        raise ValueError("build_corpus_bundle: empty records (fail-loud, no fabrication)")
    corpus = records[0].corpus

    X, meta = feat_mod.build_corpus_features(records, device=device, log=log)
    if len(meta) == 0:
        raise ValueError(f"{corpus}: 0 usable feature rows (fail-loud)")
    meta = meta.reset_index(drop=True)
    meta["group"] = meta["corpus"] + "::" + meta["speaker_id"]
    if max_utts_per_speaker is not None:
        X, meta = _cap_per_speaker(X, meta, max_utts_per_speaker, seed)

    rec_by_utt = {r.utterance_id: r for r in records}
    keep_recs = [rec_by_utt[u] for u in meta["utterance_id"] if u in rec_by_utt]

    sil_energy: dict[str, dict] = {}
    speech_pool: dict[str, np.ndarray] = {}
    silence_available = speech_pool_available = False
    if with_silence:
        sil_energy = _compute_silence_energy(keep_recs, corpus, log=log)
        silence_available = len(sil_energy) > 0
        if need_speech_pool:
            speech_pool = _compute_speech_pool(keep_recs, corpus, sil_energy, device, log=log)
            speech_pool_available = len(speech_pool) > 0

    views: list[SpeakerView] = []
    for spk, idx in meta.groupby("speaker_id").groups.items():
        idx = list(idx)
        sub = meta.iloc[idx]
        utt_ids = sub["utterance_id"].tolist()
        pooled_full = X[idx]
        recs = [rec_by_utt[u] for u in utt_ids if u in rec_by_utt]
        speech_rows, energies, sil_fracs = [], [], []
        for u, pf in zip(utt_ids, pooled_full):
            row = sil_energy.get(u)
            if row is not None:
                if np.isfinite(row["energy_db"]):
                    energies.append(float(row["energy_db"]))
                if np.isfinite(row["silence_fraction"]):
                    sil_fracs.append(float(row["silence_fraction"]))
            sp = speech_pool.get(u)
            speech_rows.append(np.asarray(sp, dtype=np.float32) if sp is not None else pf)
        views.append(SpeakerView(
            speaker_id=str(spk), corpus=corpus,
            language=str(sub["language"].iloc[0]),
            severity=float(sub["severity_rank"].iloc[0]),
            is_control=bool(sub["is_control"].iloc[0]),
            records=recs, utt_ids=utt_ids,
            pooled_full=pooled_full.astype(np.float32),
            pooled_speech=np.stack(speech_rows, axis=0).astype(np.float32),
            energy=float(np.mean(energies)) if energies else float("nan"),
            silence_fraction=float(np.mean(sil_fracs)) if sil_fracs else float("nan"),
        ))

    views.sort(key=lambda v: (v.severity, v.speaker_id))
    notes = []
    if with_silence and not silence_available:
        notes.append("silence/energy signals unavailable for this corpus "
                     "(no embeddings/TextGrids and no resolvable audio)")
    if with_silence and need_speech_pool and not speech_pool_available:
        notes.append("speech-only pooled feature unavailable; silence test on "
                     "pooled-feature metrics will be N/A for this corpus")
    return CorpusBundle(
        name=corpus, language=str(meta["language"].iloc[0]),
        views=views, X_utt=X.astype(np.float32),
        meta=meta[["utterance_id", "speaker_id", "group", "severity_rank", "is_control"]].copy(),
        silence_available=silence_available,
        speech_pool_available=speech_pool_available,
        notes=notes,
    )


def _cap_per_speaker(X: np.ndarray, meta: pd.DataFrame, cap: int, seed: int):
    """Seeded subsample of at most ``cap`` utterances per speaker (group)."""
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, idx in meta.groupby("group").groups.items():
        idx = np.asarray(idx)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.extend(idx.tolist())
    keep = sorted(keep)
    return X[keep], meta.iloc[keep].reset_index(drop=True)
