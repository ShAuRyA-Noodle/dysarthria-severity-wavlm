"""Smoke tests for Track D cross-corpus modules."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_corpus_registry_imports() -> None:
    from src.cross_corpus.corpus_registry import REGISTRY, get, list_available  # noqa: F401

    assert {"torgo", "uaspeech", "ipvs", "pcgita", "easycall"} <= set(REGISTRY)


def test_get_known_corpus_returns_dataclass() -> None:
    from src.cross_corpus.corpus_registry import get

    c = get("ipvs")
    assert c.name == "ipvs"
    assert c.language == "it"
    assert c.sample_rate == 16000
    assert "sustained_vowels" in c.tasks


def test_get_unknown_corpus_raises() -> None:
    from src.cross_corpus.corpus_registry import get

    with pytest.raises(KeyError):
        get("klingon")


def test_ipvs_parser_classify_task() -> None:
    from src.cross_corpus.ipvs_parser import _classify_task

    assert _classify_task("VA1LBULCAAS94M1") == "sustained_vowel_a"
    assert _classify_task("VE2") == "sustained_vowel_e"
    assert _classify_task("PR1") == "read_passage"
    assert _classify_task("FB2") == "read_phrase"
    assert _classify_task("B1A") == "ddk_baba"
    assert _classify_task("D2U") == "ddk_dada"
    assert _classify_task("XXXX") == "unknown"


def test_ipvs_parser_classify_group() -> None:
    from src.cross_corpus.ipvs_parser import _classify_group

    assert _classify_group("15 Young Healthy Control") == "HC_Young"
    assert _classify_group("22 Elderly Healthy Control") == "HC_Old"
    assert _classify_group("28 People with Parkinson's disease") == "PD"
    assert _classify_group("99 Random Junk") == "UNKNOWN"


def test_ipvs_parser_speaker_hash_stable() -> None:
    from src.cross_corpus.ipvs_parser import _speaker_hash

    h1 = _speaker_hash("Alberto R", "HC_Young")
    h2 = _speaker_hash("Alberto R", "HC_Young")
    h3 = _speaker_hash("Different Name", "HC_Young")
    assert h1 == h2
    assert h1 != h3
    assert h1.startswith("IPVS_HC_Young_")
    assert len(h1.split("_")[-1]) == 8


def test_ipvs_parser_pd_4level_path(tmp_path: Path) -> None:
    """Verify the 4-level PD path correctly resolves to the speaker name."""
    from src.cross_corpus.ipvs_parser import parse_ipvs_filename

    fake = tmp_path / "data/raw/ipvs/italian_parkinson/28 People with Parkinson's disease/1-5/Domenico C/VA1LBULCAAS.wav"
    fake.parent.mkdir(parents=True)
    fake.touch()
    row = parse_ipvs_filename(fake)
    assert row is not None
    assert row["group"] == "PD"
    assert row["speaker_name_raw"] == "Domenico C"
    assert row["speaker_id"].startswith("IPVS_PD_")
    assert row["task"] == "sustained_vowel_a"


def test_ipvs_parser_hc_3level_path(tmp_path: Path) -> None:
    """HC layout: 3 path components — speaker is rel[1]."""
    from src.cross_corpus.ipvs_parser import parse_ipvs_filename

    fake = tmp_path / "data/raw/ipvs/italian_parkinson/22 Elderly Healthy Control/AGNESE P/PR1LBULCAAS.wav"
    fake.parent.mkdir(parents=True)
    fake.touch()
    row = parse_ipvs_filename(fake)
    assert row is not None
    assert row["group"] == "HC_Old"
    assert row["speaker_name_raw"] == "AGNESE P"
    assert row["task"] == "read_passage"


def test_shared_pca_basis_transform_round_trip() -> None:
    from src.cross_corpus.normalization import (
        fit_corpus_stats, fit_shared_pca,
    )

    rng = np.random.default_rng(2026)
    donor = rng.standard_normal((500, 1024)).astype(np.float32)
    basis = fit_shared_pca(donor, "torgo", n_components=64)

    target = rng.standard_normal((200, 1024)).astype(np.float32) + 0.5
    target_stats = fit_corpus_stats(target, "uaspeech", "wavlm_large", 9)
    out = basis.transform(target, target_stats)
    assert out.shape == (200, 64)
    assert np.isfinite(out).all()
