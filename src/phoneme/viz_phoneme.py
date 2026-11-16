"""Phoneme-level visualization (Track B): heatmap + waveform overlay.

Produces two paper-ready figure types:

1. ``plot_phoneme_heatmap(df, out_path)`` — heatmap of (phoneme x speaker)
   z-scored ATD. Phonemes ordered by manner-of-articulation, speakers
   ordered by severity. Highlights cells with |z| > 2 in red.

2. ``plot_waveform_overlay(wav, sr, textgrid, deviations, out_path)`` —
   waveform with phone-interval shading proportional to per-phone
   z-score (deeper red = larger deviation). Prints phoneme labels
   below the waveform. Used as a clinical inspection tool.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.phoneme.per_phoneme_atd import (
    ARPABET_AFFRICATES,
    ARPABET_FRICATIVES,
    ARPABET_GLIDES,
    ARPABET_LIQUIDS,
    ARPABET_NASALS,
    ARPABET_VOICED_STOPS,
    ARPABET_VOICELESS_STOPS,
    ARPABET_VOWELS,
    manner_class,
)

# Manner ordering for the heatmap rows (clinical convention).
MANNER_ORDER = (
    "voiceless_stop", "voiced_stop", "fricative", "affricate",
    "nasal", "liquid", "glide", "vowel", "other",
)


def _order_phonemes_by_manner(phones: list[str]) -> list[str]:
    buckets: dict[str, list[str]] = {m: [] for m in MANNER_ORDER}
    for ph in phones:
        m = manner_class(ph)
        buckets.setdefault(m, []).append(ph)
    out: list[str] = []
    for m in MANNER_ORDER:
        out.extend(sorted(buckets.get(m, [])))
    return out


def plot_phoneme_heatmap(
    df: pd.DataFrame,
    out_path: Path,
    *,
    speaker_col: str = "speaker_id",
    severity_col: str = "severity_score",
    phone_col: str = "phone",
    z_col: str = "z_atd",
    z_clip: float = 4.0,
    title: str | None = "Per-phoneme z-scored ATD",
) -> Path:
    """Heatmap of z-scored ATD over (speaker × phoneme).

    Parameters
    ----------
    df : DataFrame
        Per-(speaker, phoneme) z-scored ATD. Must have columns
        speaker_id, severity_score, phone, z_atd.
    out_path : Path
        Output figure path; PNG and PDF written side-by-side.
    z_clip : float
        Heatmap colour scale clipped at ±z_clip to prevent outliers
        dominating the colormap.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    speakers = (
        df.drop_duplicates(speaker_col)
        .sort_values(severity_col)
        [speaker_col].tolist()
    )
    phonemes = _order_phonemes_by_manner(sorted(df[phone_col].unique()))

    pivot = df.pivot_table(
        index=phone_col, columns=speaker_col, values=z_col, aggfunc="mean"
    ).reindex(index=phonemes, columns=speakers)

    fig, ax = plt.subplots(figsize=(max(6, 0.4 * len(speakers) + 4),
                                    max(4, 0.18 * len(phonemes) + 2)))
    im = ax.imshow(
        pivot.values, aspect="auto", cmap="RdBu_r",
        vmin=-z_clip, vmax=z_clip, interpolation="nearest",
    )
    ax.set_xticks(range(len(speakers)))
    ax.set_xticklabels(speakers, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(phonemes)))
    ax.set_yticklabels(phonemes, fontsize=7)
    ax.set_xlabel("Speaker (ordered by severity)")
    ax.set_ylabel("Phoneme (manner-grouped)")
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("z-score (control distribution)")

    # Mark |z|>2 cells with a star.
    starred = np.where(np.abs(pivot.values) > 2)
    for r, c in zip(*starred):
        ax.text(c, r, "*", ha="center", va="center", fontsize=10,
                color="black", fontweight="bold")

    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_waveform_overlay(
    wav: np.ndarray,
    sr: int,
    phones: list[tuple[str, float, float]],
    deviations: dict[str, float],
    out_path: Path,
    *,
    title: str | None = None,
) -> Path:
    """Waveform with per-phone shading proportional to z-deviation.

    Parameters
    ----------
    wav : np.ndarray
        1-D mono waveform.
    sr : int
        Sample rate.
    phones : list of (phone_label, t_start, t_end)
        Output of `src.phoneme.aligner.parse_textgrid`.
    deviations : dict
        Map phone label → z-score (signed). Missing phones → no shading.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(len(wav)) / float(sr)

    fig, ax = plt.subplots(figsize=(max(8, len(phones) * 0.5), 4))
    ax.plot(t, wav, color="0.3", linewidth=0.6)

    for phone, t0, t1 in phones:
        z = deviations.get(phone, 0.0)
        # Map |z| to alpha (0..0.6); sign to red vs blue.
        alpha = float(min(0.6, abs(z) / 4.0))
        color = "tab:red" if z >= 0 else "tab:blue"
        ax.axvspan(t0, t1, color=color, alpha=alpha)
        ax.text((t0 + t1) / 2, ax.get_ylim()[1] * 0.85, phone,
                ha="center", va="center", fontsize=7)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path
