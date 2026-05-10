"""Smoke tests for Track B phoneme-level localization scaffolds."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_phoneme_module_imports() -> None:
    from src.phoneme import aligner, per_phoneme_atd, phoneme_norms  # noqa: F401


def test_manner_class_mapping() -> None:
    from src.phoneme.per_phoneme_atd import manner_class

    assert manner_class("AA1") == "vowel"
    assert manner_class("EH0") == "vowel"
    assert manner_class("P") == "voiceless_stop"
    assert manner_class("B") == "voiced_stop"
    assert manner_class("M") == "nasal"
    assert manner_class("S") == "fricative"
    assert manner_class("CH") == "affricate"
    assert manner_class("L") == "liquid"
    assert manner_class("Y") == "glide"
    assert manner_class("XX") == "other"


def test_phoneme_norms_zscore_falls_back_to_manner() -> None:
    from src.phoneme.phoneme_norms import PhonemeNorms

    norms = PhonemeNorms(
        by_phone={"AA1": (0.5, 0.1, 100)},
        by_manner={"vowel": (0.5, 0.2, 500), "voiceless_stop": (1.0, 0.3, 50)},
    )
    # Sufficient phone tokens → use phone-specific stats.
    assert norms.zscore("AA1", 0.6) == pytest.approx(1.0, abs=1e-6)
    # Unknown phone → fall back to manner.
    assert norms.zscore("EH1", 0.7) == pytest.approx(1.0, abs=1e-6)


def test_phoneme_norms_fit_from_dataframe() -> None:
    from src.phoneme.phoneme_norms import fit_norms_from_dataframe

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(40):
        rows.append({"phone": "AA1", "atd": float(rng.normal(0.5, 0.1)), "is_control": True})
    for _ in range(40):
        rows.append({"phone": "P", "atd": float(rng.normal(0.8, 0.2)), "is_control": True})
    df = pd.DataFrame(rows)
    norms = fit_norms_from_dataframe(df)
    assert "AA1" in norms.by_phone
    assert norms.by_phone["AA1"][2] == 40
    # vowel manner pool aggregates AA1.
    assert "vowel" in norms.by_manner
    assert norms.by_manner["vowel"][2] >= 40


def test_aligner_mfa_available_returns_bool() -> None:
    from src.phoneme.aligner import mfa_available

    assert isinstance(mfa_available(), bool)


def test_aligner_run_mfa_align_raises_without_mfa() -> None:
    from src.phoneme import aligner as A

    if A.mfa_available():
        pytest.skip("MFA installed locally — skip negative test")

    with pytest.raises(RuntimeError, match="MFA not found"):
        A.run_mfa_align(
            corpus_dir=Path("/nonexistent"),
            dictionary_path=Path("/nonexistent.dict"),
            acoustic_model="english_us_arpa",
            out_dir=Path("/tmp/mfa_out"),
        )
