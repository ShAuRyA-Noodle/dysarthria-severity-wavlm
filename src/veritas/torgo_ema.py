"""TORGO AG500 EMA loader + sensor->articulator resolver (single responsibility).

Reads the real local TORGO `.pos` electromagnetic-articulography files and maps
their 12 physical coils onto SPARC's 12-channel articulatory order so that
SPARC's audio-driven inversion can be compared against *measured* articulator
positions (experiment H1).

`.pos` format (verified, docs/veritas/DATASETS.md §1.2):
  - little-endian float32, no header, row-major frames;
  - 84 cols/frame = 12 sensors x 7 values [x, y, z, phi, theta, RMS, extra];
  - x,y,z in mm (AG500 volume), 200 Hz; n_frames = filesize / 336;
  - `pos/` = head-corrected positions (used here).

Coordinate axes (verified empirically across the 7 control speakers):
  - x = anterior(+)/posterior(-);  y = LATERAL (left/right);  z = vertical.
  SPARC EMA is midsagittal 2-D (x=front-back, y=up-down), so SPARC's (X, Y) map
  to TORGO's (x, z) — TORGO's y (lateral) is NOT a SPARC axis. The experiment
  fits a per-articulator affine that recovers the exact 3-D->2-D projection, so
  this axis note is only the initial guess for the simpler z-score baseline.

Sensor -> articulator mapping is NOT uniform across speakers (docs §1.3: coils
fail / are replaced per speaker), so we do NOT hard-code a global map. Instead
`resolve_channels` derives, per speaker, which coils are dead / reference /
mouth-corner / articulator from pooled position statistics, then labels the
tongue coils by anatomy (the tongue coils are the highest-motion oral coils and
sit posterior to the lips -> lowest anterior-posterior x). Dead coils are the
ones the recording notes flag (e.g. FC01 "Coil S10 failed badly") and show up as
degenerate trajectories (huge vertical jitter or off-volume mean).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import resample_poly

# 12 sensors x 7 float32 values x 4 bytes
POS_ROW_BYTES: int = 12 * 7 * 4  # 336
EMA_FS: int = 200
SPARC_FS: int = 50
_DECIMATION: int = EMA_FS // SPARC_FS  # 4

# SPARC EMA channel order (tongue dorsum, body, tip; lower incisor/jaw; upper &
# lower lip), each as (X, Y). Authoritative for pairing with SPARC `code['ema']`.
SPARC_EMA_ORDER: list[str] = [
    "TDX", "TDY", "TBX", "TBY", "TTX", "TTY",
    "LIX", "LIY", "ULX", "ULY", "LLX", "LLY",
]
# SPARC articulator -> its (X, Y) column indices in code['ema'].
SPARC_ARTIC_COLS: dict[str, tuple[int, int]] = {
    "TD": (0, 1), "TB": (2, 3), "TT": (4, 5),
    "LI": (6, 7), "UL": (8, 9), "LL": (10, 11),
}
TONGUE_ARTICULATORS: tuple[str, ...] = ("TD", "TB", "TT")


def load_pos(path: str, n_sensors: int = 12, n_cols: int = 7) -> np.ndarray:
    """Load a head-corrected `.pos` file -> (n_frames, 12, 7) float32 array (mm).

    Raises ``ValueError`` if the file size is not an exact multiple of 336 bytes
    (i.e. not a valid 12x7 float32 frame stream)."""
    raw = np.fromfile(path, dtype="<f4")
    if raw.size % (n_sensors * n_cols) != 0:
        raise ValueError(
            f"{path}: {raw.size} float32 values not divisible by "
            f"{n_sensors * n_cols} (corrupt or wrong format)"
        )
    return raw.reshape(-1, n_sensors, n_cols).astype(np.float32)


def n_frames(path: str) -> int:
    """Frame count without fully decoding: filesize / 336 (verified integer)."""
    import os

    size = os.path.getsize(path)
    if size % POS_ROW_BYTES != 0:
        raise ValueError(f"{path}: size {size} not a multiple of {POS_ROW_BYTES}")
    return size // POS_ROW_BYTES


def pos_xyz(path_or_arr) -> np.ndarray:
    """(n_frames, 12, 3) Cartesian x,y,z in mm (the first three of the 7 cols)."""
    arr = path_or_arr if isinstance(path_or_arr, np.ndarray) else load_pos(path_or_arr)
    return arr[..., 0:3]


def pos_rms(path_or_arr) -> np.ndarray:
    """(n_frames, 12) per-sensor fit residual (col 5); lower = better quality."""
    arr = path_or_arr if isinstance(path_or_arr, np.ndarray) else load_pos(path_or_arr)
    return arr[..., 5]


def downsample_to_50fps(arr: np.ndarray, axis: int = 0) -> np.ndarray:
    """200 Hz -> 50 fps with an anti-aliasing polyphase filter (factor 4).

    Matches SPARC's 50 fps EMA rate so frames can be paired 1:1 (frame j of the
    down-sampled EMA <-> SPARC frame j, both at audio time j/50 s)."""
    return resample_poly(arr, up=1, down=_DECIMATION, axis=axis)


@dataclass
class ChannelMap:
    """Per-speaker resolution of the 12 TORGO coils.

    dead/ref/lateral/articulator are 0-based coil indices; ``tongue`` and
    ``lips_jaw`` map SPARC articulator labels to the resolved coil index.
    """

    dead: list[int]
    ref: list[int]
    lateral: list[int]
    articulators: list[int]
    tongue: dict[str, int]
    lips_jaw: dict[str, int] = field(default_factory=dict)

    @property
    def articulator_map(self) -> dict[str, int]:
        """All resolved SPARC-articulator -> coil-index assignments."""
        return {**self.tongue, **self.lips_jaw}


def resolve_channels(
    pooled_xyz: np.ndarray,
    dead_z_jitter_mm: float = 8.0,
    off_volume_mm: float = 150.0,
    flatline_mm: float = 0.05,
    ref_motion_mm: float = 1.3,
    lateral_offset_mm: float = 25.0,
) -> ChannelMap:
    """Classify the 12 coils from pooled (N, 12, 3) mm positions of one speaker.

    - **dead**: degenerate coils — vertical jitter > 8 mm (a coil that "failed
      badly"), mean position off the measurement volume (|mean| > 150 mm), or a
      flat-lined trajectory (< 0.05 mm total motion).
    - **reference**: near-stationary coils (total std < 1.3 mm) used for head
      correction (nasion / incisor / mastoids) — they define the frame.
    - **lateral**: mouth-corner coils, offset in y (lateral) from the oral
      midline by > 25 mm — not SPARC articulators.
    - **articulators**: the rest. The 3 with the smallest anterior-posterior x
      are the tongue coils (tongue sits posterior to the lips); ordered by x they
      are dorsum (most posterior) -> body -> tip (most anterior). Remaining
      (anterior) coils are labelled lips/jaw best-effort by height z (upper lip
      highest, lower lip lowest, jaw/lower-incisor between).
    """
    xyz = np.asarray(pooled_xyz, dtype=np.float64)
    mean = np.nanmean(xyz, axis=0)          # (12, 3)
    std = np.nanstd(xyz, axis=0)            # (12, 3)
    total_motion = np.linalg.norm(std, axis=1)

    dead = [
        c for c in range(12)
        if std[c, 2] > dead_z_jitter_mm
        or np.abs(mean[c]).max() > off_volume_mm
        or total_motion[c] < flatline_mm
    ]
    ref = [c for c in range(12) if c not in dead and total_motion[c] < ref_motion_mm]
    rest = [c for c in range(12) if c not in dead and c not in ref]
    y_mid = float(np.median([mean[c, 1] for c in rest])) if rest else 0.0
    lateral = [c for c in rest if abs(mean[c, 1] - y_mid) > lateral_offset_mm]
    artic = [c for c in rest if c not in lateral]

    by_x = sorted(artic, key=lambda c: mean[c, 0])  # ascending x (posterior->anterior)
    tongue_idx = by_x[:3]                            # 3 most posterior = tongue
    tongue_sorted = sorted(tongue_idx, key=lambda c: mean[c, 0])
    tongue: dict[str, int] = {}
    labels = ["TD", "TB", "TT"]  # dorsum(post) -> body -> tip(ant)
    for lab, c in zip(labels, tongue_sorted):
        tongue[lab] = c

    front = [c for c in artic if c not in tongue_idx]  # anterior coils = lips/jaw
    lips_jaw: dict[str, int] = {}
    if front:
        by_z = sorted(front, key=lambda c: -mean[c, 2])  # highest z first
        lips_jaw["UL"] = by_z[0]
        lips_jaw["LL"] = by_z[-1]
        middle = [c for c in front if c not in (by_z[0], by_z[-1])]
        if middle:
            lips_jaw["LI"] = middle[0]

    return ChannelMap(
        dead=dead, ref=ref, lateral=lateral, articulators=artic,
        tongue=tongue, lips_jaw=lips_jaw,
    )


def is_clean_for_tongue(pooled_xyz: np.ndarray, cmap: ChannelMap,
                        max_dead: int = 2, min_tongue_motion_mm: float = 1.0) -> tuple[bool, str]:
    """Objective data-validity gate for H1: enough live tongue coils on a
    non-degenerate frame. Returns (ok, reason). NOT a SPARC-performance gate.

    Rejects speakers where the AG500 session is unusable — too many dead coils
    (e.g. MC02 with 9 dead), or the resolved "tongue" coils don't actually move
    vertically (a mis-resolved reference coil)."""
    if len(cmap.dead) > max_dead:
        return False, f"{len(cmap.dead)} dead coils > {max_dead} (degenerate session)"
    if len(cmap.tongue) != 3:
        return False, f"resolved {len(cmap.tongue)} tongue coils, need 3"
    std = np.nanstd(np.asarray(pooled_xyz, dtype=np.float64), axis=0)  # (12,3)
    for lab, c in cmap.tongue.items():
        if std[c, 2] < min_tongue_motion_mm:
            return False, f"tongue coil {lab}=S{c + 1} vertical motion {std[c,2]:.2f}mm too low"
    return True, "ok"
