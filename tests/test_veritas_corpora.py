"""Loader tests for the VERITAS cross-corpus severity harness.

One test per on-disk corpus asserts REAL records load (real speakers, real
severity ranks, resolvable wav paths). The not-yet-ready corpora are asserted to
SKIP gracefully with a non-empty activation hint (never fabricating rows).

These tests touch only metadata + the filesystem — no WavLM / GPU — so they run
fast in CI.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.veritas.corpora import ipvs, registry, torgo, uaspeech  # noqa: E402
from src.veritas.corpora.base import UtteranceRecord, remap_stale_path  # noqa: E402


# --------------------------------------------------------------------------- #
# on-disk corpora: must load REAL records
# --------------------------------------------------------------------------- #
def test_torgo_loads_real_graded_records() -> None:
    st = torgo.status()
    assert st.available, st.reason
    recs = torgo.load()
    assert len(recs) > 0
    assert all(isinstance(r, UtteranceRecord) for r in recs)
    speakers = {r.speaker_id for r in recs}
    assert len(speakers) == 15  # 8 dysarthric + 7 control
    sev = {r.severity_rank for r in recs}
    assert sev == {0.0, 1.0, 2.0}  # graded
    # controls are exactly the severity-0 speakers
    assert all(r.is_control == (r.severity_rank == 0.0) for r in recs)
    assert {r.etiology for r in recs} == {"control", "dysarthria"}
    assert all(r.language == "en" for r in recs)


def test_ipvs_loads_real_binary_records() -> None:
    st = ipvs.status()
    assert st.available, st.reason
    assert st.label_kind == "binary_detection"
    recs = ipvs.load()
    assert len(recs) > 0
    speakers = {r.speaker_id for r in recs}
    assert len(speakers) >= 40  # 24 PD + 37 HC in the mirror
    assert {r.severity_rank for r in recs} == {0.0, 1.0}  # binary
    assert {r.etiology for r in recs} == {"PD", "control"}
    assert all(r.language == "it" for r in recs)
    # every wav path actually resolves on disk (real audio)
    assert all(Path(r.wav_path).exists() for r in recs[:25])


def test_ipvs_elderly_only_drops_young_controls() -> None:
    full = ipvs.load()
    eld = ipvs.load(elderly_only=True)
    assert 0 < len(eld) <= len(full)
    # PD speakers retained; young HC removed
    assert any(r.etiology == "PD" for r in eld)


# --------------------------------------------------------------------------- #
# not-yet-ready corpora: skip gracefully, never fabricate
# --------------------------------------------------------------------------- #
def test_uaspeech_skips_gracefully_without_audio() -> None:
    st = uaspeech.status()
    if st.available:
        recs = uaspeech.load()
        assert len(recs) > 0  # if audio ever lands, it must load real rows
    else:
        assert uaspeech.load() == []
        assert st.activation  # tells us exactly what to provide
        assert st.n_speakers_hint == 28


def test_registry_reports_all_and_loads_only_real() -> None:
    statuses = {s.name: s for s in registry.all_status()}
    assert {"torgo", "uaspeech", "ipvs", "easycall", "mdvr_kcl", "vocals", "svd"} <= set(statuses)
    # every skipped corpus must carry an activation hint
    for s in statuses.values():
        if not s.available:
            assert s.activation, f"{s.name} skipped without activation hint"
    data = registry.load_available(severity_targets_only=True)
    assert "torgo" in data and "ipvs" in data
    for recs in data.values():
        assert len(recs) > 0  # never register an empty corpus


def test_svd_is_not_a_severity_target() -> None:
    svd_status = next(s for s in registry.all_status() if s.name == "svd")
    assert svd_status.is_severity_target is False


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def test_remap_stale_path_rewrites_old_checkout() -> None:
    stale = r"C:\Users\Dell\Desktop\Projects\Dysar\data\raw\ipvs\x.wav"
    out = remap_stale_path(stale)
    assert "Projects" not in out
    assert out.endswith(r"data\raw\ipvs\x.wav")
    # unknown prefixes pass through unchanged
    assert remap_stale_path("/some/other/path.wav") == "/some/other/path.wav"
