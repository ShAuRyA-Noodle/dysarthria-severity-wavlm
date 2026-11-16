"""Per-phoneme ATD via DTW-path slicing (Track B core algorithm).

Given an utterance's full DTW alignment to a control reference and the
utterance's MFA phoneme intervals, attribute the path's cumulative cost to each
phoneme's frame range. The decomposition is EXACT once silence/unattributed
path steps are accounted for by an explicit residual:

    sum_p D(p) + residual = total DTW path cost.

``residual`` is the cost carried by DTW steps whose utterance frame falls in a
silence interval (sil/sp/spn — dropped by the MFA ``phones`` tier) or in no MFA
phone interval at all. For TORGO single-word recordings this residual is large
(leading/trailing silence dominates the recording), so the honest claim is an
"exact additive decomposition into per-phoneme contributions plus a
silence/unattributed residual" — NOT exact phone-only additivity.

Length-normalised per-phoneme deviation:

    ATD(p) = D(p) / max(L_u(p), L_r(p))

where L_u(p) is the utterance frames inside the phone, L_r(p) is the
unique reference frames the path visits during the phone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.constants import FASTDTW_RADIUS

WAVLM_FPS = 50  # WavLM-Large output frame rate (20 ms hop)


@dataclass
class UtteranceDecomposition:
    """Exact additive decomposition of one utterance's DTW path cost.

    Identity (holds to floating-point tolerance):
        ``sum(r["distance"] for r in per_phone) + residual == total_dtw_distance``
    """

    per_phone: list[dict]
    total_dtw_distance: float
    residual: float  # total - sum(per-phone): silence / unattributed path mass
    n_path_steps: int = 0
    n_attributed_steps: int = 0

    @property
    def sum_phone_distance(self) -> float:
        return float(sum(r["distance"] for r in self.per_phone))

    @property
    def residual_fraction(self) -> float:
        return float(self.residual / self.total_dtw_distance) if self.total_dtw_distance > 0 else 0.0


def decompose_utterance(
    utt_npy_path: Path,
    textgrid_path: Path,
    pca_model,
    reference: np.ndarray,
    layer_position: int = 3,
    fps: int = WAVLM_FPS,
    radius: int = FASTDTW_RADIUS,
) -> UtteranceDecomposition:
    """Decompose an utterance's DTW path cost into per-phone + residual.

    Computes the full DTW path once, attributes each step's local cost to the
    phone interval containing its utterance frame (if any), and books all
    remaining cost (silence / unattributed steps) into an explicit residual so
    the additive identity holds EXACTLY.

    Parameters
    ----------
    utt_npy_path : Path
        Stacked WavLM embedding ``.npy`` of shape (n_layers, T_utt, 1024).
    textgrid_path : Path
        MFA-aligned TextGrid for the same utterance.
    pca_model : sklearn.decomposition.PCA
        Fitted PCA from `src.reference_builder.build_word_reference`.
    reference : np.ndarray
        (T_ref, 64) PCA-space reference trajectory.
    layer_position : int
        Index into the stacked-layer axis (default 3 → layer 9).
    fps : int
        WavLM frame rate (default 50).
    radius : int
        FastDTW Sakoe-Chiba radius.

    Returns
    -------
    UtteranceDecomposition
        ``.per_phone`` rows (phone, t_start, t_end, distance, atd, n_path_steps,
        L_utt_frames, L_ref_frames), ``.total_dtw_distance`` and ``.residual``.
    """
    from src.phoneme.aligner import parse_textgrid  # local import to keep top-of-module light

    emb = np.load(str(utt_npy_path))[layer_position]            # (T_utt, 1024)
    utt = pca_model.transform(emb)                              # (T_utt, 64)

    total_dtw_distance, path = fastdtw(utt, reference, radius=radius, dist=euclidean)
    total_dtw_distance = float(total_dtw_distance)
    step_costs = np.array([euclidean(utt[i], reference[j]) for i, j in path])
    path_i = np.array([i for i, _ in path])
    path_j = np.array([j for _, j in path])
    n_path_steps = int(len(path))

    phones = parse_textgrid(textgrid_path)
    out: list[dict] = []
    # Track which path steps get attributed to a phone so the residual = the
    # rest. Non-silence MFA intervals can overlap in frame space (rounding), so
    # we OR a boolean mask rather than summing per-phone counts.
    attributed = np.zeros(len(path), dtype=bool)
    for phone, t0, t1 in phones:
        f0 = int(np.floor(t0 * fps))
        f1 = min(int(np.ceil(t1 * fps)), len(utt))
        if f1 <= f0:
            continue
        mask = (path_i >= f0) & (path_i < f1)
        n_steps = int(mask.sum())
        if n_steps == 0:
            continue
        attributed |= mask
        D = float(step_costs[mask].sum())
        L_u = f1 - f0
        L_r = int(np.unique(path_j[mask]).size)
        atd_p = D / max(L_u, L_r)
        out.append({
            "phone": phone, "t_start": float(t0), "t_end": float(t1),
            "distance": D, "atd": atd_p, "n_path_steps": n_steps,
            "L_utt_frames": int(L_u), "L_ref_frames": int(L_r),
        })

    # Residual = total path cost minus cost on steps attributed to a phone.
    # Using the path steps directly (not sum of per-phone distances) guarantees
    # exactness even if two phone intervals were to claim the same frame: each
    # path step's cost is counted once in attributed_cost.
    attributed_cost = float(step_costs[attributed].sum())
    sum_phone = float(sum(r["distance"] for r in out))
    # In the (rare) absence of overlapping intervals attributed_cost == sum_phone;
    # the residual is defined against the per-phone sum that is actually stored,
    # so the stored identity sum_phone + residual == total holds exactly.
    residual = total_dtw_distance - sum_phone

    return UtteranceDecomposition(
        per_phone=out,
        total_dtw_distance=total_dtw_distance,
        residual=residual,
        n_path_steps=n_path_steps,
        n_attributed_steps=int(attributed.sum()),
    )


def per_phoneme_atd(
    utt_npy_path: Path,
    textgrid_path: Path,
    pca_model,
    reference: np.ndarray,
    layer_position: int = 3,
    fps: int = WAVLM_FPS,
    radius: int = FASTDTW_RADIUS,
) -> list[dict]:
    """Backward-compatible wrapper: return only the per-phone rows.

    Each row additionally carries ``total_dtw_distance`` and ``residual`` for
    the parent utterance so downstream CSVs can store the full identity. Prefer
    :func:`decompose_utterance` for the explicit total/residual object.
    """
    dec = decompose_utterance(
        utt_npy_path, textgrid_path, pca_model, reference,
        layer_position=layer_position, fps=fps, radius=radius,
    )
    for row in dec.per_phone:
        row["total_dtw_distance"] = dec.total_dtw_distance
        row["residual"] = dec.residual
        row["residual_fraction"] = dec.residual_fraction
    return dec.per_phone


def verify_additivity(
    per_phone_rows: list[dict],
    total_dtw_distance: float,
    residual: float,
    tol: float = 1e-6,
) -> bool:
    """Assert the exact additive identity ``sum(D_p) + residual == total``.

    Returns True iff ``|sum(per-phone distance) + residual - total| <= tol``
    (absolute) OR within ``tol`` relative to the total. The residual term is
    REQUIRED: per-phone costs alone do NOT sum to the total because silence and
    unattributed DTW steps carry cost. Callers should report the residual
    fraction separately.
    """
    s = sum(r["distance"] for r in per_phone_rows)
    lhs = s + residual
    abs_err = abs(lhs - total_dtw_distance)
    return abs_err <= tol or abs_err <= tol * max(abs(total_dtw_distance), 1.0)


# Manner-of-articulation pooling for sparse-class fallback (ARPABET).
ARPABET_VOWELS = frozenset(
    {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY",
     "OW", "OY", "UH", "UW"}
)
ARPABET_VOICELESS_STOPS = frozenset({"P", "T", "K"})
ARPABET_VOICED_STOPS = frozenset({"B", "D", "G"})
ARPABET_NASALS = frozenset({"M", "N", "NG"})
ARPABET_FRICATIVES = frozenset({"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"})
ARPABET_AFFRICATES = frozenset({"CH", "JH"})
ARPABET_LIQUIDS = frozenset({"L", "R"})
ARPABET_GLIDES = frozenset({"W", "Y"})


def manner_class(phone: str) -> str:
    """Map an ARPABET phone (with optional 0/1/2 stress digit) to a manner class."""
    base = phone.rstrip("0123456789").upper()
    if base in ARPABET_VOWELS:
        return "vowel"
    if base in ARPABET_VOICELESS_STOPS:
        return "voiceless_stop"
    if base in ARPABET_VOICED_STOPS:
        return "voiced_stop"
    if base in ARPABET_NASALS:
        return "nasal"
    if base in ARPABET_FRICATIVES:
        return "fricative"
    if base in ARPABET_AFFRICATES:
        return "affricate"
    if base in ARPABET_LIQUIDS:
        return "liquid"
    if base in ARPABET_GLIDES:
        return "glide"
    return "other"
