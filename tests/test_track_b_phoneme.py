"""Smoke + correctness tests for Track B phoneme-level localization."""
from __future__ import annotations

import json
import pickle
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


# ──────────────────────────────────────────────────────────────────────────
# DEFECT 2: exact additive decomposition on REAL phoneme alignment data.
# ──────────────────────────────────────────────────────────────────────────

_LAYER = 9
_DATA = PROJECT_ROOT / "data"
_TEXTGRIDS = _DATA / "processed" / "textgrids"
_REF_MANIFEST = _DATA / "processed" / "references" / "reference_manifest.json"
_SSL_INDEX = _DATA / "embeddings" / "ssl_index.json"
_METADATA = _DATA / "processed" / "metadata" / "metadata_unified.csv"


def _real_phoneme_cases(n: int = 4) -> list[dict]:
    """Collect up to ``n`` REAL utterances that have TextGrid + reference + npy.

    Returns dicts with the loaded pca_model, reference, npy path, textgrid path.
    Empty list if the corpus is not present in this checkout.
    """
    if not (_TEXTGRIDS.exists() and _REF_MANIFEST.exists()
            and _SSL_INDEX.exists() and _METADATA.exists()):
        return []
    from src.constants import LAYER_POSITION_MAP  # noqa: F401 (validate import)

    with open(_REF_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)
    with open(_SSL_INDEX, encoding="utf-8") as f:
        ssl_index = json.load(f)
    meta = pd.read_csv(_METADATA)

    cases: list[dict] = []
    ref_load_cache: dict[str, dict] = {}
    for r in meta.itertuples(index=False):
        if len(cases) >= n:
            break
        utt = getattr(r, "utterance_id", None)
        spk = getattr(r, "speaker_id", None)
        word = getattr(r, "word", None)
        if not utt or not spk or not word:
            continue
        tg = _TEXTGRIDS / str(spk) / f"{utt}.TextGrid"
        if not tg.exists():
            continue
        key = f"{word}_{_LAYER}"
        info = manifest.get("word_layer_ref", {}).get(key)
        if info is None:
            continue
        entry = ssl_index.get(utt)
        if entry is None or entry.get("wavlm_large") is None:
            continue
        npy = PROJECT_ROOT / entry["wavlm_large"]
        if not npy.exists():
            continue
        if key not in ref_load_cache:
            pkl = PROJECT_ROOT / info["path"]
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                ref_load_cache[key] = pickle.load(f)
        ref_data = ref_load_cache[key]
        cases.append({
            "utt": utt, "word": word,
            "pca_model": ref_data["pca_model"],
            "reference": ref_data["reference"],
            "npy": npy, "tg": tg,
        })
    return cases


def test_phoneme_decomposition_exact_additive_identity_on_real_data() -> None:
    """sum(per-phone) + residual == total DTW cost, EXACTLY, on real utterances.

    This is the corrected DEFECT-2 identity. It is NOT phone-only additive
    (silence carries cost); the residual term is required and is reported.
    """
    from src.constants import LAYER_POSITION_MAP
    from src.phoneme.per_phoneme_atd import decompose_utterance, verify_additivity

    cases = _real_phoneme_cases(n=4)
    if not cases:
        pytest.skip("real TORGO phoneme alignment data not present in this checkout")

    lp = LAYER_POSITION_MAP[_LAYER]
    seen_nonzero_residual = False
    for c in cases:
        dec = decompose_utterance(
            utt_npy_path=c["npy"],
            textgrid_path=c["tg"],
            pca_model=c["pca_model"],
            reference=c["reference"],
            layer_position=lp,
        )
        # The stored identity must hold to floating-point tolerance.
        recombined = dec.sum_phone_distance + dec.residual
        assert recombined == pytest.approx(dec.total_dtw_distance, rel=1e-9, abs=1e-6), (
            f"{c['utt']}: sum_phone={dec.sum_phone_distance} + "
            f"residual={dec.residual} != total={dec.total_dtw_distance}"
        )
        assert verify_additivity(dec.per_phone, dec.total_dtw_distance, dec.residual)
        # total must be stored and positive; residual must be real (not silently 0).
        assert dec.total_dtw_distance > 0
        assert np.isfinite(dec.residual)
        assert 0.0 <= dec.residual_fraction <= 1.0 + 1e-9
        if dec.residual > 1e-6:
            seen_nonzero_residual = True

    # On TORGO single-word recordings silence dominates, so at least one real
    # utterance MUST carry a non-trivial residual — proving the residual term is
    # load-bearing and phone-only additivity would be FALSE.
    assert seen_nonzero_residual, (
        "no real utterance carried any silence/unattributed residual — "
        "the residual term would be vacuous"
    )


def test_phoneme_per_phoneme_atd_rows_carry_total_and_residual() -> None:
    """The backward-compatible row API must expose total + residual per row."""
    from src.constants import LAYER_POSITION_MAP
    from src.phoneme.per_phoneme_atd import per_phoneme_atd

    cases = _real_phoneme_cases(n=1)
    if not cases:
        pytest.skip("real TORGO phoneme alignment data not present in this checkout")
    c = cases[0]
    rows = per_phoneme_atd(
        utt_npy_path=c["npy"],
        textgrid_path=c["tg"],
        pca_model=c["pca_model"],
        reference=c["reference"],
        layer_position=LAYER_POSITION_MAP[_LAYER],
    )
    assert rows, "expected at least one phone row for a real utterance"
    total = rows[0]["total_dtw_distance"]
    residual = rows[0]["residual"]
    for row in rows:
        assert row["total_dtw_distance"] == total
        assert row["residual"] == residual
    sum_phone = sum(r["distance"] for r in rows)
    assert sum_phone + residual == pytest.approx(total, rel=1e-9, abs=1e-6)


def test_phoneme_additivity_fails_without_residual_term() -> None:
    """Guard: per-phone sum alone does NOT equal total (residual is required).

    Confirms verify_additivity correctly REJECTS the old (residual=0) claim on
    real data where silence carries cost.
    """
    from src.constants import LAYER_POSITION_MAP
    from src.phoneme.per_phoneme_atd import decompose_utterance, verify_additivity

    cases = _real_phoneme_cases(n=4)
    if not cases:
        pytest.skip("real TORGO phoneme alignment data not present in this checkout")

    lp = LAYER_POSITION_MAP[_LAYER]
    rejected_at_least_once = False
    for c in cases:
        dec = decompose_utterance(
            utt_npy_path=c["npy"], textgrid_path=c["tg"],
            pca_model=c["pca_model"], reference=c["reference"], layer_position=lp,
        )
        # Old false claim: sum(per-phone) == total with NO residual term.
        if dec.residual > 1e-3:
            assert not verify_additivity(dec.per_phone, dec.total_dtw_distance, residual=0.0), (
                f"{c['utt']}: phone-only sum wrongly accepted as == total "
                f"(residual={dec.residual} is non-trivial)"
            )
            rejected_at_least_once = True
    assert rejected_at_least_once, (
        "no real utterance had a non-trivial residual to reject — "
        "the negative test is not discriminating"
    )
