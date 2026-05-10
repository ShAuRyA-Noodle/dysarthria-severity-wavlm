"""Per-phoneme z-norm distributions built from controls only (Track B)."""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.phoneme.per_phoneme_atd import manner_class

MIN_TOKENS_FULL_PHONE = 30  # below this, fall back to manner pooling


@dataclass
class PhonemeNorms:
    """Mean/std of per-phoneme ATD over control utterances only.

    Used to z-score test (dysarthric) per-phoneme ATDs into a per-phone
    distribution: |z| > 2 highlights phones the patient deviated on.
    """

    by_phone: dict[str, tuple[float, float, int]] = field(default_factory=dict)
    by_manner: dict[str, tuple[float, float, int]] = field(default_factory=dict)

    def zscore(self, phone: str, atd: float) -> float:
        """Return z-score; falls back to manner pooling if phone is sparse."""
        rec = self.by_phone.get(phone)
        if rec is None or rec[2] < MIN_TOKENS_FULL_PHONE:
            rec = self.by_manner.get(manner_class(phone))
        if rec is None:
            return float("nan")
        mu, sd, _ = rec
        if sd <= 1e-9:
            return 0.0
        return (atd - mu) / sd


def fit_norms_from_dataframe(
    per_phoneme_df: pd.DataFrame, is_control_field: str = "is_control"
) -> PhonemeNorms:
    """Build PhonemeNorms from a DataFrame produced by per_phoneme_atd loops.

    Expected columns: phone, atd, is_control. Returns an empty PhonemeNorms
    if the DataFrame is empty or lacks the required columns.
    """
    required = {"phone", "atd", is_control_field}
    if per_phoneme_df.empty or not required.issubset(per_phoneme_df.columns):
        return PhonemeNorms()
    ctrl = per_phoneme_df[per_phoneme_df[is_control_field].astype(bool)]
    by_phone: dict[str, tuple[float, float, int]] = {}
    for phone, sub in ctrl.groupby("phone"):
        atds = sub["atd"].astype(float).to_numpy()
        if len(atds) < 2:
            continue
        by_phone[phone] = (float(atds.mean()), float(atds.std(ddof=1)), int(len(atds)))

    by_manner: dict[str, tuple[float, float, int]] = {}
    ctrl_with_manner = ctrl.assign(manner=ctrl["phone"].map(manner_class))
    for manner, sub in ctrl_with_manner.groupby("manner"):
        atds = sub["atd"].astype(float).to_numpy()
        if len(atds) < 5:
            continue
        by_manner[manner] = (float(atds.mean()), float(atds.std(ddof=1)), int(len(atds)))

    return PhonemeNorms(by_phone=by_phone, by_manner=by_manner)


def save_norms(norms: PhonemeNorms, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(norms, f)


def load_norms(path: Path) -> PhonemeNorms:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_norms_json(norms: PhonemeNorms, path: Path) -> None:
    """Export a human-readable JSON copy alongside the pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "by_phone": {p: {"mean": m, "std": s, "n": n} for p, (m, s, n) in norms.by_phone.items()},
        "by_manner": {p: {"mean": m, "std": s, "n": n} for p, (m, s, n) in norms.by_manner.items()},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
