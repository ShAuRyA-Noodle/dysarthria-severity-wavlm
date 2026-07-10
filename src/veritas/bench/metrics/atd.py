"""Reference metric #2 (reference-based): the ATD trajectory-deviation score.

ATD = length-normalised DTW distance between a speaker's frozen-WavLM (PCA-64)
trajectory and a per-word CONTROL reference. This is the project's headline
metric. The adapter reuses the committed artefacts exactly — cached per-frame
WavLM embeddings, the global + leave-one-control-out (LOCO) reference manifests,
and MFA TextGrids — so it reproduces the published leakage and silence numbers.

Scoring modes (all speaker-mean of per-utterance ATD; higher = more severe):
  * ``naive``      : score every speaker vs the GLOBAL control reference (a
                     control speaker's own data is IN its reference -> leaks).
  * ``loco`` /
    ``predictive`` : controls scored vs a leave-one-control-out reference
                     (speaker-disjoint); dysarthric speakers vs the global
                     control reference (already disjoint).
  * ``feature_view="speech_only"`` : references are rebuilt and utterances scored
                     on MFA speech-span frames only (silence removed).

Worked example of a **reference**-kind metric for the docs.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

from src.constants import LAYER_POSITION_MAP
from src.veritas.bench.data import SpeakerView
from src.veritas.bench.metric import VeritasMetric
from src.veritas.corpora.base import PROJECT_ROOT

_SSL_INDEX = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
_GLOBAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"
_LOCO_ROOT = PROJECT_ROOT / "data" / "processed" / "references_loco"
_TEXTGRID_DIR = PROJECT_ROOT / "data" / "processed" / "textgrids"

_FPS = 50
_PCA_N = 64
_RADIUS = 1            # published headline uses fastdtw default radius=1
_MIN_CONTROL_UTTS = 2
_MIN_FRAMES = 2


class ATDReferenceDistance(VeritasMetric):
    kind = "reference"
    has_representation = True

    def __init__(self, layer: int = 9):
        if layer not in LAYER_POSITION_MAP:
            raise ValueError(f"layer {layer} not in {sorted(LAYER_POSITION_MAP)}")
        self.layer = int(layer)
        self.layer_pos = LAYER_POSITION_MAP[layer]
        self.name = f"atd_wavlm_l{layer}"
        self.representation_desc = f"frozen WavLM-Large layer-{layer} PCA-64 trajectory vs control reference"
        with open(_SSL_INDEX, encoding="utf-8") as f:
            self._ssl_index = json.load(f)
        self._global_cache: dict[str, dict] | None = None
        self._loco_caches: dict[str, dict] | None = None
        self._memo: dict[tuple[str, str], dict[str, float]] = {}

    # -- artefact loading ------------------------------------------------------
    def _load_global(self) -> dict[str, dict]:
        if self._global_cache is None:
            self._global_cache = _load_manifest_cache(_GLOBAL_MANIFEST, self.layer)
        return self._global_cache

    def _load_loco(self, control_speakers: set[str]) -> dict[str, dict]:
        if self._loco_caches is None:
            self._loco_caches = {}
        for spk in sorted(control_speakers):
            if spk in self._loco_caches:
                continue
            man = _LOCO_ROOT / spk / "reference_manifest.json"
            if not man.exists():
                raise FileNotFoundError(
                    f"LOCO reference manifest missing for {spk}: {man}. "
                    f"ATD leakage test needs data/processed/references_loco/<control>/."
                )
            self._loco_caches[spk] = _load_manifest_cache(man, self.layer)
        return self._loco_caches

    def _frames(self, utt_id: str) -> np.ndarray | None:
        entry = self._ssl_index.get(utt_id)
        if entry is None or entry.get("wavlm_large") is None:
            return None
        p = PROJECT_ROOT / entry["wavlm_large"]
        if not p.exists():
            return None
        return np.load(str(p))[self.layer_pos]

    def _word(self, utt_id: str) -> str | None:
        entry = self._ssl_index.get(utt_id)
        return None if entry is None else entry.get("word") or None

    # -- public scoring API ----------------------------------------------------
    def predictive_scores(self, views, *, feature_view: str = "full") -> dict[str, float]:
        return self._scores(views, mode="loco", feature_view=feature_view)

    def naive_scores(self, views, *, feature_view: str = "full") -> dict[str, float]:
        return self._scores(views, mode="naive", feature_view=feature_view)

    def loco_scores(self, views, *, feature_view: str = "full") -> dict[str, float]:
        return self._scores(views, mode="loco", feature_view=feature_view)

    # -- core ------------------------------------------------------------------
    def _scores(self, views: list[SpeakerView], mode: str, feature_view: str) -> dict[str, float]:
        key = (mode, feature_view)
        want = {v.speaker_id for v in views}
        if key in self._memo and set(self._memo[key]) == want:
            return dict(self._memo[key])
        if feature_view == "full":
            out = self._scores_full(views, mode)
        elif feature_view == "speech_only":
            out = self._scores_speech(views, mode)
        else:
            raise ValueError(f"unknown feature_view {feature_view!r}")
        self._memo[key] = out
        return out

    def _scores_full(self, views: list[SpeakerView], mode: str) -> dict[str, float]:
        """Score using the committed manifests (exact published reproduction)."""
        global_cache = self._load_global()
        controls = {v.speaker_id for v in views if v.is_control}
        loco_caches = self._load_loco(controls) if mode == "loco" else {}

        per_spk: dict[str, list[float]] = {v.speaker_id: [] for v in views}
        for v in views:
            for utt_id in v.utt_ids:
                word = self._word(utt_id)
                if word is None:
                    continue
                if mode == "loco" and v.is_control:
                    ref = loco_caches.get(v.speaker_id, {}).get(word)
                else:
                    ref = global_cache.get(word)
                if ref is None:
                    continue
                frames = self._frames(utt_id)
                if frames is None or frames.shape[0] < _MIN_FRAMES:
                    continue
                per_spk[v.speaker_id].append(_score_against_ref(frames, ref))
        return _speaker_means(per_spk, self.name, mode, "full")

    def _scores_speech(self, views: list[SpeakerView], mode: str) -> dict[str, float]:
        """Rebuild references on MFA speech-span frames and score speech frames."""
        # 1. slice every utterance's trajectory to its speech span (cache).
        slices: dict[str, np.ndarray] = {}
        for v in views:
            for utt_id in v.utt_ids:
                tr = self._speech_frames(utt_id, v.speaker_id)
                if tr is not None:
                    slices[utt_id] = tr

        # 2. control utterances grouped by word (for reference construction).
        word_to_ctrl: dict[str, list[tuple[str, str]]] = {}
        for v in views:
            if not v.is_control:
                continue
            for utt_id in v.utt_ids:
                word = self._word(utt_id)
                if word is not None and utt_id in slices:
                    word_to_ctrl.setdefault(word, []).append((utt_id, v.speaker_id))

        def build_ref(word: str, exclude: str | None):
            trajs = [slices[u] for u, s in word_to_ctrl.get(word, [])
                     if u in slices and (exclude is None or s != exclude)]
            if len(trajs) < _MIN_CONTROL_UTTS:
                return None
            return _build_reference(trajs)

        global_ref: dict[str, dict | None] = {}
        if mode == "naive":
            for word in word_to_ctrl:
                global_ref[word] = build_ref(word, exclude=None)

        per_spk: dict[str, list[float]] = {v.speaker_id: [] for v in views}
        for v in views:
            for utt_id in v.utt_ids:
                if utt_id not in slices:
                    continue
                word = self._word(utt_id)
                if word is None:
                    continue
                if mode == "loco" and v.is_control:
                    ref = build_ref(word, exclude=v.speaker_id)
                else:
                    ref = global_ref.get(word) if mode == "naive" else build_ref(word, exclude=None)
                if ref is None:
                    continue
                per_spk[v.speaker_id].append(_score_against_ref(slices[utt_id], ref))
        return _speaker_means(per_spk, self.name, mode, "speech_only")

    def _speech_frames(self, utt_id: str, speaker_id: str) -> np.ndarray | None:
        frames = self._frames(utt_id)
        if frames is None:
            return None
        tg = _TEXTGRID_DIR / speaker_id / f"{utt_id}.TextGrid"
        if not tg.exists():
            return None
        span = _speech_span(tg, frames.shape[0])
        if span is None:
            return None
        f0, f1 = span
        seg = frames[f0:f1]
        return seg if seg.shape[0] >= _MIN_FRAMES else None


# --- module helpers -----------------------------------------------------------
def _load_manifest_cache(manifest_path: Path, layer: int) -> dict[str, dict]:
    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)
    cache: dict[str, dict] = {}
    for key, info in man.get("word_layer_ref", {}).items():
        word, lyr = key.rsplit("_", 1)
        if int(lyr) != layer:
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as fh:
                cache[word] = pickle.load(fh)
    return cache


def _score_against_ref(utt_frames: np.ndarray, ref_data: dict) -> float:
    pca = ref_data["pca_model"]
    reference = ref_data["reference"]
    ur = pca.transform(utt_frames)
    dist, _ = fastdtw(ur, reference, radius=_RADIUS, dist=euclidean)
    return float(dist / max(len(ur), len(reference)))


def _speech_span(tg_path: Path, T: int) -> tuple[int, int] | None:
    from src.phoneme.aligner import parse_textgrid

    phones = parse_textgrid(tg_path)
    if not phones:
        return None
    onset = min(p[1] for p in phones)
    offset = max(p[2] for p in phones)
    f0 = max(0, min(int(np.floor(onset * _FPS)), T))
    f1 = max(0, min(int(np.ceil(offset * _FPS)), T))
    return (f0, f1) if f1 > f0 else None


def _warp_to_template(traj: np.ndarray, path: list, tlen: int) -> np.ndarray:
    D = traj.shape[1]
    warped = np.zeros((tlen, D))
    counts = np.zeros(tlen)
    for ti, si in path:
        if ti < tlen and si < len(traj):
            warped[ti] += traj[si]
            counts[ti] += 1
    counts = np.maximum(counts, 1)
    warped /= counts[:, None]
    return warped


def _build_reference(trajectories: list[np.ndarray]) -> dict | None:
    """Concat-fit PCA-64, median-length template, DTW-warp others, average.

    Identical algorithm to ``src.reference_builder.build_word_reference`` but on
    caller-supplied (speech-only) trajectories.
    """
    trajs = [t for t in trajectories if t is not None and t.shape[0] >= 1]
    if len(trajs) < _MIN_CONTROL_UTTS:
        return None
    all_frames = np.concatenate(trajs, axis=0)
    n_comp = min(_PCA_N, all_frames.shape[0], all_frames.shape[1])
    if n_comp < 1:
        return None
    pca = PCA(n_components=n_comp, random_state=42)
    pca.fit(all_frames)
    reduced = [pca.transform(t) for t in trajs]
    lengths = [len(t) for t in reduced]
    median_len = int(np.median(lengths))
    ci = min(range(len(lengths)), key=lambda i: abs(lengths[i] - median_len))
    template = reduced[ci]
    aligned = [template]
    for i, tr in enumerate(reduced):
        if i == ci:
            continue
        _, path = fastdtw(template, tr, radius=_RADIUS, dist=euclidean)
        aligned.append(_warp_to_template(tr, path, len(template)))
    return {"reference": np.mean(aligned, axis=0), "pca_model": pca}


def _speaker_means(per_spk: dict[str, list[float]], name: str, mode: str, fv: str) -> dict[str, float]:
    out: dict[str, float] = {}
    empty = []
    for spk, vals in per_spk.items():
        if vals:
            out[spk] = float(np.mean(vals))
        else:
            empty.append(spk)
            out[spk] = float("nan")
    if empty:
        # fail-loud: surface which speakers produced no scored utterances
        print(f"[warn] {name} {mode}/{fv}: {len(empty)} speaker(s) with no scored "
              f"utterances -> NaN: {empty[:5]}")
    return out
