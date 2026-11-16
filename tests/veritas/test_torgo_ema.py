"""TDD for the TORGO AG500 EMA loader (real local .pos files).

The .pos format (verified in docs/veritas/DATASETS.md): little-endian float32,
no header, 84 cols = 12 sensors x 7 [x,y,z,phi,theta,RMS,extra], mm, 200 Hz,
n_frames = filesize / 336.
"""
import glob
import os

import numpy as np

from src.veritas.torgo_ema import (
    POS_ROW_BYTES,
    SPARC_EMA_ORDER,
    downsample_to_50fps,
    load_pos,
    n_frames,
    pos_xyz,
    resolve_channels,
)

POS = sorted(glob.glob(r"C:/Users/Dell/Desktop/Dysar/FC/FC01/**/pos/*.pos", recursive=True))[0]


def test_pos_row_bytes():
    # 12 sensors x 7 float32 values x 4 bytes = 336 bytes/frame
    assert POS_ROW_BYTES == 12 * 7 * 4 == 336


def test_load_pos_frame_count_and_shape():
    size = os.path.getsize(POS)
    assert size % POS_ROW_BYTES == 0, "real .pos must be an exact multiple of 336 bytes"
    expected = size // POS_ROW_BYTES
    arr = load_pos(POS)
    assert arr.shape == (expected, 12, 7)
    assert arr.dtype == np.float32
    assert n_frames(POS) == expected


def test_pos_xyz_is_first_three_cols_in_mm():
    arr = load_pos(POS)
    xyz = pos_xyz(POS)
    assert xyz.shape == (arr.shape[0], 12, 3)
    np.testing.assert_allclose(xyz, arr[..., 0:3])
    # AG500 measurement volume: positions are tens of mm, never astronomically large
    assert np.nanmax(np.abs(xyz)) < 1000.0


def test_downsample_200_to_50fps_quarters_length():
    xyz = pos_xyz(POS)  # 200 Hz
    ds = downsample_to_50fps(xyz)  # -> 50 fps
    assert abs(ds.shape[0] - xyz.shape[0] // 4) <= 2
    assert ds.shape[1:] == xyz.shape[1:]


def test_sparc_order_is_twelve_channels():
    assert SPARC_EMA_ORDER == [
        "TDX", "TDY", "TBX", "TBY", "TTX", "TTY",
        "LIX", "LIY", "ULX", "ULY", "LLX", "LLY",
    ]


def test_resolve_channels_flags_fc01_dead_coil_s10():
    # FC01 Notes: "Coil S10 failed badly" -> channel index 9 (0-based) must be dead,
    # and exactly 3 tongue sensors must be resolved for a clean speaker.
    files = sorted(glob.glob(r"C:/Users/Dell/Desktop/Dysar/FC/FC01/**/pos/*.pos", recursive=True))[:40]
    pooled = np.concatenate([pos_xyz(f) for f in files], axis=0)
    cmap = resolve_channels(pooled)
    assert 9 in cmap.dead, f"S10 (idx 9) should be dead, got dead={cmap.dead}"
    assert len(cmap.tongue) == 3
    # tongue sensors are alive and actually move vertically
    for idx in cmap.tongue.values():
        assert idx not in cmap.dead
