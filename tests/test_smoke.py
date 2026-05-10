"""Smoke tests — verify imports and core invariants without needing TORGO data.

Run with::

    pytest tests/test_smoke.py -q

These tests must pass on CPU-only machines and complete in <30 seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_constants_module_imports() -> None:
    from src.constants import (
        LAYER_POSITION_MAP,
        RANDOM_SEED,
        TARGET_LAYERS,
        TORGO_CONTROL_SPEAKERS,
        TORGO_DYSARTHRIC_SPEAKERS,
        TORGO_SEVERITY,
    )

    assert RANDOM_SEED == 42
    assert TARGET_LAYERS == [1, 3, 6, 9, 12, 15, 18, 21, 23]
    assert len(TORGO_SEVERITY) == 15
    assert TORGO_CONTROL_SPEAKERS == frozenset(
        {"FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"}
    )
    assert TORGO_DYSARTHRIC_SPEAKERS == frozenset(
        {"F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"}
    )
    assert TORGO_CONTROL_SPEAKERS.isdisjoint(TORGO_DYSARTHRIC_SPEAKERS)
    for layer, pos in LAYER_POSITION_MAP.items():
        assert TARGET_LAYERS[pos] == layer


def test_seed_utils_is_idempotent() -> None:
    from src.seed_utils import set_global_seed

    set_global_seed(42)
    a = np.random.rand(8)
    set_global_seed(42)
    b = np.random.rand(8)
    np.testing.assert_array_equal(a, b)


def test_auto_device_returns_known_value() -> None:
    from src.seed_utils import auto_device

    assert auto_device() in {"cuda", "cpu"}


def test_atd_metric_imports() -> None:
    from src.atd_metric import compute_single_atd, validate_atd_metric  # noqa: F401


def test_reference_builder_imports() -> None:
    from src.reference_builder import build_word_reference  # noqa: F401


def test_ssl_extractor_imports() -> None:
    from src.ssl_extractor import SSLExtractor, extract_dataset_embeddings  # noqa: F401


def test_data_pipeline_imports() -> None:
    from src.data_pipeline import parse_torgo_filename  # noqa: F401


def test_uaspeech_parser_imports() -> None:
    from src.uaspeech_parser import parse_uaspeech_filename  # noqa: F401


def test_baselines_imports() -> None:
    from src.baselines import extract_mfcc_features  # noqa: F401


def test_visualization_imports() -> None:
    import src.visualization  # noqa: F401


def test_paper_numbers_json_is_consistent() -> None:
    """Smoke check: paper_numbers.json exists and headline ρ matches Readme claim."""
    import json

    p = PROJECT_ROOT / "results" / "paper_numbers.json"
    if not p.exists():
        pytest.skip("results/paper_numbers.json not generated yet")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    # Headline speaker-level Spearman rho ≈ 0.756 per Readme.
    candidates = []
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if "rho" in k.lower() or "spearman" in k.lower():
                    if isinstance(v, (int, float)):
                        candidates.append((k, v))
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)

    _walk(data)
    assert candidates, "No rho/spearman entries found in paper_numbers.json"
    near_headline = [v for _, v in candidates if 0.74 < v < 0.78]
    assert near_headline, f"Expected ρ near 0.756 in paper_numbers.json; got {candidates}"


def test_combined_csv_speaker_ids_disambiguated() -> None:
    """Speaker IDs in combined CSV must be unique once dataset prefix is applied."""
    import pandas as pd

    p = PROJECT_ROOT / "results" / "combined_speaker_atd.csv"
    if not p.exists():
        pytest.skip("combined_speaker_atd.csv not generated yet")

    df = pd.read_csv(p)
    assert "dataset_speaker_id" in df.columns, (
        "Expected dataset_speaker_id column to disambiguate TORGO_M01 vs UA_M01."
    )
    assert df["dataset_speaker_id"].is_unique
