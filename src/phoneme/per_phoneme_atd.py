"""Per-phoneme ATD via DTW-path slicing (Track B core algorithm).

Given an utterance's full DTW alignment to a control reference and the
utterance's MFA phoneme intervals, attribute the path's cumulative cost
to each phoneme's frame range. The decomposition is exact:

    sum_p D(p) = total DTW distance.

Length-normalised per-phoneme deviation:

    ATD(p) = D(p) / max(L_u(p), L_r(p))

where L_u(p) is the utterance frames inside the phone, L_r(p) is the
unique reference frames the path visits during the phone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

from src.constants import FASTDTW_RADIUS

WAVLM_FPS = 50  # WavLM-Large output frame rate (20 ms hop)


def per_phoneme_atd(
    utt_npy_path: Path,
    textgrid_path: Path,
    pca_model,
    reference: np.ndarray,
    layer_position: int = 3,
    fps: int = WAVLM_FPS,
    radius: int = FASTDTW_RADIUS,
) -> list[dict]:
    """Compute per-phoneme deviation by slicing the DTW path.

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
    list of dicts with keys: phone, t_start, t_end, distance, atd, n_path_steps,
    L_utt_frames, L_ref_frames. Sum of `distance` over all phones equals
    the total DTW distance (exact additive identity).
    """
    from src.phoneme.aligner import parse_textgrid  # local import to keep top-of-module light

    emb = np.load(str(utt_npy_path))[layer_position]            # (T_utt, 1024)
    utt = pca_model.transform(emb)                              # (T_utt, 64)

    _, path = fastdtw(utt, reference, radius=radius, dist=euclidean)
    step_costs = np.array([euclidean(utt[i], reference[j]) for i, j in path])
    path_i = np.array([i for i, _ in path])
    path_j = np.array([j for _, j in path])

    phones = parse_textgrid(textgrid_path)
    out: list[dict] = []
    for phone, t0, t1 in phones:
        f0 = int(np.floor(t0 * fps))
        f1 = min(int(np.ceil(t1 * fps)), len(utt))
        if f1 <= f0:
            continue
        mask = (path_i >= f0) & (path_i < f1)
        n_steps = int(mask.sum())
        if n_steps == 0:
            continue
        D = float(step_costs[mask].sum())
        L_u = f1 - f0
        L_r = int(np.unique(path_j[mask]).size)
        atd_p = D / max(L_u, L_r)
        out.append({
            "phone": phone, "t_start": float(t0), "t_end": float(t1),
            "distance": D, "atd": atd_p, "n_path_steps": n_steps,
            "L_utt_frames": int(L_u), "L_ref_frames": int(L_r),
        })
    return out


def verify_additivity(
    per_phone_rows: list[dict], total_dtw_distance: float, tol: float = 1e-3
) -> bool:
    """Sum-of-phone-costs ≈ total DTW distance (modulo silence frames)."""
    s = sum(r["distance"] for r in per_phone_rows)
    return abs(s - total_dtw_distance) / max(total_dtw_distance, 1e-9) < tol


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
