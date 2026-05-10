"""Shared constants for ATD project — single source of truth.

Replaces 5+ duplicate definitions across atd_metric.py, baselines.py,
data_pipeline.py, reference_builder.py, ssl_extractor.py.
"""

from __future__ import annotations

# ── Reproducibility ────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── TORGO speaker severity (0=Control, 1=Mild-Moderate, 2=Severe) ──────
TORGO_SEVERITY: dict[str, int] = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}

TORGO_GENDER: dict[str, str] = {
    "F01": "F", "F03": "F", "F04": "F",
    "M01": "M", "M02": "M", "M03": "M", "M04": "M", "M05": "M",
    "FC01": "F", "FC02": "F", "FC03": "F",
    "MC01": "M", "MC02": "M", "MC03": "M", "MC04": "M",
}

# Speakers eligible to build control reference trajectories.
TORGO_CONTROL_SPEAKERS: frozenset[str] = frozenset(
    spk for spk, sev in TORGO_SEVERITY.items() if sev == 0
)

TORGO_DYSARTHRIC_SPEAKERS: frozenset[str] = frozenset(
    spk for spk, sev in TORGO_SEVERITY.items() if sev > 0
)

# ── WavLM-Large layer sampling ─────────────────────────────────────────
# 9 representative layers across the 24-layer transformer.
TARGET_LAYERS: list[int] = [1, 3, 6, 9, 12, 15, 18, 21, 23]

# Map: actual transformer layer number → position in stacked (n_layers, T, 1024) arrays.
LAYER_POSITION_MAP: dict[int, int] = {
    layer: pos for pos, layer in enumerate(TARGET_LAYERS)
}

# Backwards-compat alias (some files use LAYER_IDX_MAP).
LAYER_IDX_MAP: dict[int, int] = LAYER_POSITION_MAP

# ── PCA / DTW defaults ─────────────────────────────────────────────────
PCA_N_COMPONENTS: int = 64
FASTDTW_RADIUS: int = 30  # higher than default 1; near-exact DTW
DTW_DISTANCE: str = "euclidean"

# ── SSL extractor defaults ─────────────────────────────────────────────
DEFAULT_SSL_MODEL: str = "wavlm_large"
DEFAULT_LAYER: int = 9  # primary analysis layer

# ── Audio preprocessing ────────────────────────────────────────────────
TARGET_SAMPLE_RATE: int = 16000
TARGET_DBFS: float = -20.0


def is_control_speaker(speaker_id: str) -> bool:
    """Return True iff speaker_id is one of the 7 TORGO controls."""
    return speaker_id in TORGO_CONTROL_SPEAKERS


def get_severity(speaker_id: str, default: int = -1) -> int:
    """Look up severity for a speaker_id; -1 if unknown."""
    return TORGO_SEVERITY.get(speaker_id, default)
