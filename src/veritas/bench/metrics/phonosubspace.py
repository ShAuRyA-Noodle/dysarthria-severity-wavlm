"""Reference metric #3 (reference-based): a phonological-subspace d' metric.

A faithful REIMPLEMENTATION, from the published description, of the currently
strongest training-free frozen-SSL dysarthria metric family (Muller et al.,
2026): phone-level embeddings are taken from a frozen SSL model via MFA
alignment; a direction for each binary phonological contrast (nasality, voicing,
stridency, sonorance, continuancy) is estimated **from healthy controls only**;
each speaker is then scored by the d' separation of their own phone embeddings
along those directions. Degraded contrasts => lower d' => more severe.

    score = -mean_c d'_c          (higher = more severe, per the bench contract)
    d'_c  = |mu+ - mu-| / sqrt(0.5 * (var+ + var-))   projected on direction w_c
    w_c   = mean(class+) - mean(class-)               estimated on CONTROLS

THIS IS NOT THE AUTHORS' CODE. It is an independent reimplementation used to
*audit* the metric family, and is labelled as such wherever it is reported. It
is scored on the same frozen features and alignments as every other metric here,
so the comparison is apples-to-apples.

Why it is a ``reference`` metric (this is the whole point of the audit):
the contrast directions are estimated from the healthy-control pool. When a
CONTROL speaker is scored against directions that were fitted using that same
speaker's data, the score is inflated by speaker identity. The published form
does not exclude the test speaker; the LOCO form does. The battery's leakage
test measures exactly that gap -- a confound the original work does not report
controlling for.

  * ``naive`` : directions fitted on ALL controls (the as-published form; a
                control speaker's own data is in its own direction -> leaks).
  * ``loco``  : directions re-fitted excluding the test control speaker
                (speaker-disjoint). Dysarthric speakers are unaffected (their
                data never enters the control directions).

The metric is silence-free by construction (it only ever reads frames inside
aligned phone intervals), so ``feature_view`` is accepted and ignored.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.constants import LAYER_POSITION_MAP
from src.phoneme.per_phoneme_atd import (
    ARPABET_AFFRICATES,
    ARPABET_FRICATIVES,
    ARPABET_GLIDES,
    ARPABET_LIQUIDS,
    ARPABET_NASALS,
    ARPABET_VOICED_STOPS,
    ARPABET_VOICELESS_STOPS,
    ARPABET_VOWELS,
)
from src.veritas.bench.data import SpeakerView
from src.veritas.bench.metric import VeritasMetric
from src.veritas.corpora.base import PROJECT_ROOT

_SSL_INDEX = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
_TEXTGRID_DIR = PROJECT_ROOT / "data" / "processed" / "textgrids"

_FPS = 50
_MIN_TOKENS_PER_CLASS = 5     # Muller et al. require >=5 tokens/class for a valid d'
_MIN_FRAMES_PER_PHONE = 1

_VOICED_FRICATIVES = frozenset({"V", "DH", "Z", "ZH"})
_VOICELESS_FRICATIVES = frozenset({"F", "TH", "S", "SH", "HH"})
_STOPS = ARPABET_VOICELESS_STOPS | ARPABET_VOICED_STOPS
_OBSTRUENTS = _STOPS | ARPABET_FRICATIVES | ARPABET_AFFRICATES
_SONORANTS = ARPABET_VOWELS | ARPABET_NASALS | ARPABET_LIQUIDS | ARPABET_GLIDES
_STRIDENT = frozenset({"S", "Z", "SH", "ZH"}) | ARPABET_AFFRICATES
_NONSTRIDENT_FRIC = frozenset({"F", "V", "TH", "DH"})

# Binary phonological contrasts: name -> (positive class, negative class)
CONTRASTS: dict[str, tuple[frozenset, frozenset]] = {
    "nasality":    (ARPABET_NASALS, _STOPS),
    "voicing":     (ARPABET_VOICED_STOPS | _VOICED_FRICATIVES,
                    ARPABET_VOICELESS_STOPS | _VOICELESS_FRICATIVES),
    "stridency":   (_STRIDENT, _NONSTRIDENT_FRIC),
    "sonorance":   (_SONORANTS, _OBSTRUENTS),
    "continuancy": (ARPABET_FRICATIVES, _STOPS),
}


def _base_phone(label: str) -> str:
    """Strip the MFA/ARPABET stress digit: 'AA1' -> 'AA'."""
    return label.rstrip("012").upper()


class PhonologicalSubspace(VeritasMetric):
    """Muller-style phonological-subspace d' severity metric (reimplementation)."""

    kind = "reference"
    has_representation = True

    def __init__(self, layer: int = 9):
        if layer not in LAYER_POSITION_MAP:
            raise ValueError(f"layer {layer} not in {sorted(LAYER_POSITION_MAP)}")
        self.layer = int(layer)
        self.layer_pos = LAYER_POSITION_MAP[layer]
        self.name = f"phono_subspace_l{layer}"
        self.representation_desc = (
            f"frozen WavLM-Large layer-{layer} phone-level embeddings; "
            f"d' along control-estimated phonological-contrast directions"
        )
        with open(_SSL_INDEX, encoding="utf-8") as f:
            self._ssl_index = json.load(f)
        self._phone_cache: dict[str, dict[str, np.ndarray] | None] = {}
        self._memo: dict[tuple, dict[str, float]] = {}

    # -- feature access --------------------------------------------------------
    def _frames(self, utt_id: str) -> np.ndarray | None:
        entry = self._ssl_index.get(utt_id)
        if entry is None or entry.get("wavlm_large") is None:
            return None
        p = PROJECT_ROOT / entry["wavlm_large"]
        if not p.exists():
            return None
        arr = np.load(str(p), mmap_mode="r")     # read only the needed layer
        return np.asarray(arr[self.layer_pos])

    def _phone_embeddings(self, utt_id: str, speaker_id: str) -> dict[str, list[np.ndarray]] | None:
        """{base_phone: [embedding, ...]} — mean-pooled frames inside each phone."""
        from src.phoneme.aligner import parse_textgrid

        tg = _TEXTGRID_DIR / speaker_id / f"{utt_id}.TextGrid"
        if not tg.exists():
            return None
        frames = self._frames(utt_id)
        if frames is None:
            return None
        try:
            phones = parse_textgrid(tg)
        except Exception:  # noqa: BLE001 — a corrupt TextGrid must not kill the run
            return None
        if not phones:
            return None

        T = frames.shape[0]
        out: dict[str, list[np.ndarray]] = {}
        for label, t0, t1 in phones:
            f0 = max(0, min(int(np.floor(t0 * _FPS)), T))
            f1 = max(0, min(int(np.ceil(t1 * _FPS)), T))
            if f1 - f0 < _MIN_FRAMES_PER_PHONE:
                continue
            out.setdefault(_base_phone(label), []).append(
                frames[f0:f1].mean(axis=0).astype(np.float64))
        return out or None

    def _speaker_phones(self, v: SpeakerView) -> dict[str, np.ndarray]:
        """{base_phone: (n_tokens, D)} pooled over all of a speaker's utterances."""
        if v.speaker_id in self._phone_cache:
            cached = self._phone_cache[v.speaker_id]
            return cached or {}
        acc: dict[str, list[np.ndarray]] = {}
        for utt_id in v.utt_ids:
            per_utt = self._phone_embeddings(utt_id, v.speaker_id)
            if not per_utt:
                continue
            for ph, embs in per_utt.items():
                acc.setdefault(ph, []).extend(embs)
        stacked = {ph: np.stack(e) for ph, e in acc.items() if e}
        self._phone_cache[v.speaker_id] = stacked or None
        return stacked

    # -- contrast directions ---------------------------------------------------
    @staticmethod
    def _class_tokens(phones: dict[str, np.ndarray], cls: frozenset) -> np.ndarray | None:
        rows = [phones[ph] for ph in phones if ph in cls]
        if not rows:
            return None
        return np.concatenate(rows, axis=0)

    def _direction(
        self, control_phones: list[dict[str, np.ndarray]], contrast: str
    ) -> np.ndarray | None:
        """w = mean(class+) - mean(class-), estimated on the CONTROL pool only."""
        pos_cls, neg_cls = CONTRASTS[contrast]
        pos, neg = [], []
        for ph in control_phones:
            p = self._class_tokens(ph, pos_cls)
            n = self._class_tokens(ph, neg_cls)
            if p is not None:
                pos.append(p)
            if n is not None:
                neg.append(n)
        if not pos or not neg:
            return None
        P = np.concatenate(pos, 0)
        N = np.concatenate(neg, 0)
        if len(P) < _MIN_TOKENS_PER_CLASS or len(N) < _MIN_TOKENS_PER_CLASS:
            return None
        w = P.mean(0) - N.mean(0)
        nrm = np.linalg.norm(w)
        return None if nrm < 1e-9 else w / nrm

    @staticmethod
    def _dprime(phones: dict[str, np.ndarray], contrast: str, w: np.ndarray) -> float | None:
        """d' between the two classes for ONE speaker, projected on direction w."""
        pos_cls, neg_cls = CONTRASTS[contrast]
        P = PhonologicalSubspace._class_tokens(phones, pos_cls)
        N = PhonologicalSubspace._class_tokens(phones, neg_cls)
        if P is None or N is None:
            return None
        if len(P) < _MIN_TOKENS_PER_CLASS or len(N) < _MIN_TOKENS_PER_CLASS:
            return None
        p, n = P @ w, N @ w
        pooled = 0.5 * (p.var(ddof=1) + n.var(ddof=1))
        if not np.isfinite(pooled) or pooled < 1e-12:
            return None
        return float(abs(p.mean() - n.mean()) / np.sqrt(pooled))

    # -- scoring ---------------------------------------------------------------
    def _scores(self, views: list[SpeakerView], mode: str) -> dict[str, float]:
        key = (mode,)
        want = {v.speaker_id for v in views}
        if key in self._memo and set(self._memo[key]) == want:
            return dict(self._memo[key])

        phones = {v.speaker_id: self._speaker_phones(v) for v in views}
        controls = [v for v in views if v.is_control]

        # directions fitted on the whole control pool (the as-published form)
        global_dirs: dict[str, np.ndarray | None] = {
            c: self._direction([phones[v.speaker_id] for v in controls if phones[v.speaker_id]], c)
            for c in CONTRASTS
        }

        out: dict[str, float] = {}
        for v in views:
            ph = phones[v.speaker_id]
            if not ph:
                out[v.speaker_id] = float("nan")
                continue
            # LOCO: for a control speaker, re-fit directions WITHOUT their own data.
            if mode == "loco" and v.is_control:
                pool = [phones[c.speaker_id] for c in controls
                        if c.speaker_id != v.speaker_id and phones[c.speaker_id]]
                dirs = {c: self._direction(pool, c) for c in CONTRASTS}
            else:
                dirs = global_dirs

            ds = [d for c, w in dirs.items() if w is not None
                  for d in [self._dprime(ph, c, w)] if d is not None]
            # higher = more severe  =>  negate mean d' (degraded contrast = low d')
            out[v.speaker_id] = float(-np.mean(ds)) if ds else float("nan")

        self._memo[key] = out
        return dict(out)

    def predictive_scores(self, views, *, feature_view: str = "full") -> dict[str, float]:
        return self._scores(views, mode="loco")      # speaker-disjoint

    def naive_scores(self, views, *, feature_view: str = "full") -> dict[str, float]:
        return self._scores(views, mode="naive")     # as published (leaks)

    def loco_scores(self, views, *, feature_view: str = "full") -> dict[str, float] | None:
        return self._scores(views, mode="loco")
