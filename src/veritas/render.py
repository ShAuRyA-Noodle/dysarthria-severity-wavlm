"""Midsagittal articulator renderer — turn SPARC EMA into a vocal-tract movie.

Given SPARC's 12-channel EMA code ``(T, 12)`` at 50 fps (columns
``TDX TDY TBX TBY TTX TTY LIX LIY ULX ULY LLX LLY`` = tongue dorsum / body /
tip, lower incisor/jaw, upper lip, lower lip) this draws an animated
midsagittal ("side view") of the vocal tract: the six articulator points on a
static hand-drawn head + hard-palate outline, with a smooth cubic-spline tongue
contour through the three tongue points. One video frame per EMA frame, so at
50 fps the movie plays back in real time, synced to the input audio.

The EMA channels are z-normalised (per channel), so each articulator is drawn at
an anatomically placed *home* position and moved by its scaled EMA deviation.
Larger X = more anterior (toward the lips); larger Y = more superior (toward the
palate) — the convention recovered from SPARC's own output (lips carry the
largest X, the tongue tip sits anterior to the dorsum, the upper lip above the
lower). The absolute cross-articulator layout is set by the fixed homes; the EMA
drives the *motion*. This is a faithful visualisation of the inferred
kinematics, not a calibrated anatomical measurement.

No network, no ground-truth articulography. MP4 encoding uses the ffmpeg binary
bundled with ``imageio-ffmpeg`` (offline; also present on a HF Space).
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless / server-safe
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# EMA channel layout (SPARC 12-dim tract variables)
# ---------------------------------------------------------------------------
CHANNELS = ["TDX", "TDY", "TBX", "TBY", "TTX", "TTY", "LIX", "LIY", "ULX", "ULY", "LLX", "LLY"]
# (x_index, y_index) per articulator
IDX = {
    "TD": (0, 1),   # tongue dorsum (back)
    "TB": (2, 3),   # tongue body (mid)
    "TT": (4, 5),   # tongue tip (front)
    "LI": (6, 7),   # lower incisor / jaw
    "UL": (8, 9),   # upper lip
    "LL": (10, 11),  # lower lip
}

# Anatomical "home" positions in display units (face profile pointing LEFT:
# lips at small x / front, pharynx at large x / back; larger y = higher).
HOME = {
    "UL": (1.75, 5.45),
    "LL": (1.95, 4.70),
    "LI": (2.55, 4.15),
    "TT": (3.25, 4.55),
    "TB": (4.60, 4.50),
    "TD": (5.90, 4.55),
}
# Fixed tongue anchors (not EMA-driven) that close the tongue contour/body.
TONGUE_FRONT_ANCHOR = (2.80, 4.15)   # tissue just below/behind the tip
TONGUE_ROOT_ANCHOR = (6.35, 3.80)    # tongue root at the back
TONGUE_BASE = [(2.95, 3.95), (4.45, 3.85), (5.95, 3.80)]  # underside base line

# EMA -> display mapping. +X anterior => screen moves LEFT (face points left);
# +Y superior => screen moves UP.
_SCALE = 0.42
_CLIP = 2.6
_SGN_X = -1.0   # +X (anterior) -> -screen_x (toward the lips, left)
_SGN_Y = +1.0   # +Y (superior) -> +screen_y (up)

# Colours (readable in light backgrounds)
C_TONGUE = "#d1495b"    # tongue points + contour
C_LIP = "#2e6fb5"       # upper/lower lip
C_JAW = "#3f8f5b"       # jaw / lower incisor
C_OUTLINE = "#b9b3ad"   # head profile
C_PALATE = "#7a7069"    # palate / rigid structures
C_FILL = "#e8a0aa"      # tongue body fill


def _articulator_xy(ema_row: np.ndarray, name: str) -> tuple[float, float]:
    xi, yi = IDX[name]
    hx, hy = HOME[name]
    ex = float(np.clip(ema_row[xi], -_CLIP, _CLIP))
    ey = float(np.clip(ema_row[yi], -_CLIP, _CLIP))
    return hx + _SGN_X * _SCALE * ex, hy + _SGN_Y * _SCALE * ey


def _tongue_contour(pts: list[tuple[float, float]], n: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Smooth curve through the ordered tongue points (front->back)."""
    pts = np.asarray(pts, dtype=float)
    try:
        from scipy.interpolate import splprep, splev

        # de-duplicate near-identical points (splprep dislikes them)
        keep = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i] - pts[keep[-1]])) > 1e-3:
                keep.append(i)
        p = pts[keep]
        if len(p) < 4:
            raise ValueError("too few unique points for cubic spline")
        tck, _ = splprep([p[:, 0], p[:, 1]], s=0.0, k=3)
        u = np.linspace(0, 1, n)
        x, y = splev(u, tck)
        return np.asarray(x), np.asarray(y)
    except Exception:
        # linear fallback — never crash the render on a spline hiccup
        return pts[:, 0], pts[:, 1]


def _draw_static_head(ax: "plt.Axes") -> None:
    """Hand-drawn midsagittal reference: head profile, palate, teeth, pharynx."""
    # Outer head profile (face pointing left), smoothed + filled (skin tone).
    # Ordered clockwise from the forehead, around the nose/lips/chin/throat,
    # up the back of the head and over the crown. A small notch at the left is
    # the mouth opening where the lip points sit.
    profile = [
        (4.0, 9.05), (2.9, 8.85), (2.15, 8.15), (1.75, 7.45),   # forehead, brow, bridge
        (0.85, 6.35), (1.7, 5.95),                              # nose tip, subnasale
        (1.55, 5.55),                                           # upper lip ridge
        (1.5, 5.15),                                            # mouth opening
        (1.75, 4.7),                                            # lower lip ridge
        (2.05, 4.05), (2.6, 3.4),                               # chin, under-chin
        (3.7, 2.95), (5.0, 2.82), (6.3, 3.0),                   # throat floor
        (7.15, 3.75), (7.5, 5.0), (7.4, 6.35),                  # lower/mid/upper back
        (6.9, 7.65), (5.6, 8.65),                               # back of head, crown
    ]
    sx, sy = _tongue_contour(profile + [profile[0]], n=240)
    ax.fill(sx, sy, color="#f6ede3", alpha=0.9, zorder=0, lw=0)
    ax.plot(sx, sy, "-", color=C_OUTLINE, lw=2.0, alpha=0.95, zorder=1,
            solid_capstyle="round")

    # Oral + pharyngeal cavity (white) so the tongue/palate read clearly.
    cavity = [
        (1.6, 5.15), (2.3, 5.35), (3.1, 5.95), (4.0, 6.1), (4.9, 5.85),
        (5.6, 5.4), (6.05, 4.85), (6.3, 4.2), (6.4, 3.55), (6.2, 3.25),
        (5.2, 3.35), (4.0, 3.45), (3.1, 3.7), (2.45, 4.15), (2.0, 4.7),
    ]
    qx, qy = _tongue_contour(cavity + [cavity[0]], n=200)
    ax.fill(qx, qy, color="#ffffff", alpha=1.0, zorder=1, lw=0)

    # Palate (continuous): alveolar ridge -> hard palate -> soft palate/velum.
    palate = [
        (1.95, 5.35), (2.4, 5.5), (3.1, 6.05), (4.0, 6.2), (4.9, 5.95),
        (5.6, 5.45), (6.05, 4.9), (6.2, 4.35),
    ]
    hx, hy = _tongue_contour(palate, n=80)
    ax.plot(hx, hy, "-", color=C_PALATE, lw=3.0, alpha=0.95, zorder=3,
            solid_capstyle="round")

    # Pharyngeal back wall (near-vertical, behind the tongue root); meets velum.
    phar = [(6.2, 4.35), (6.55, 4.15), (6.75, 3.7), (6.6, 3.25)]
    wx, wy = _tongue_contour(phar, n=40)
    ax.plot(wx, wy, "-", color=C_PALATE, lw=2.6, alpha=0.9, zorder=3,
            solid_capstyle="round")

    # Upper teeth (small triangle just behind the upper lip / alveolar ridge).
    ax.fill([1.9, 2.15, 2.12], [5.35, 5.35, 4.95], color="#f2f0ec",
            edgecolor=C_PALATE, lw=1.2, zorder=4)
    # Nostril hint
    ax.plot([0.95, 1.3], [6.0, 5.92], "-", color=C_OUTLINE, lw=1.4, alpha=0.6, zorder=2)


def render_articulators(
    ema: np.ndarray,
    out_path: str,
    fps: int = 50,
    audio: np.ndarray | None = None,
    sr: int = 16000,
    faithfulness: np.ndarray | None = None,
    title: str | None = None,
    max_seconds: float | None = 15.0,
    dpi: int = 100,
) -> str:
    """Render SPARC EMA ``(T, 12)`` to a midsagittal vocal-tract MP4.

    Parameters
    ----------
    ema : (T, 12) array of SPARC tract variables at ``fps`` (50).
    out_path : destination ``.mp4`` path.
    audio : optional waveform to mux in (so the video plays with sound). If
        given, it is trimmed to match ``max_seconds`` alongside the animation.
    sr : sample rate of ``audio``.
    faithfulness : optional per-frame reconstruction-confidence in [0,1]; drawn
        as a synced strip at the bottom, labelled EXPERIMENTAL.
    title : optional headline drawn on the figure.
    max_seconds : cap the movie length (keeps very long clips real-time and
        fast to render); ``None`` = no cap.

    Returns the path actually written.
    """
    import imageio.v2 as imageio
    import imageio_ffmpeg

    ema = np.asarray(ema, dtype=np.float32)
    if ema.ndim != 2 or ema.shape[1] != 12:
        raise ValueError(f"ema must be (T,12); got {ema.shape}")
    T = ema.shape[0]
    if T < 2:
        raise ValueError("need at least 2 EMA frames to animate")

    # length cap (keeps it real-time)
    if max_seconds is not None:
        max_frames = int(round(max_seconds * fps))
        if T > max_frames:
            ema = ema[:max_frames]
            T = ema.shape[0]

    faith = None
    if faithfulness is not None:
        faith = np.asarray(faithfulness, dtype=np.float32).ravel()

    out_path = str(out_path)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    has_strip = faith is not None and len(faith) >= 2
    fig_h = 6.6 if has_strip else 5.6
    fig = plt.figure(figsize=(6.4, fig_h), dpi=dpi)
    if has_strip:
        ax = fig.add_axes([0.02, 0.26, 0.96, 0.70])
        axf = fig.add_axes([0.10, 0.08, 0.82, 0.13])
    else:
        ax = fig.add_axes([0.02, 0.04, 0.96, 0.92])
        axf = None

    ax.set_xlim(0.0, 8.2)
    ax.set_ylim(1.6, 9.6)
    ax.set_aspect("equal")
    ax.axis("off")
    _draw_static_head(ax)
    if title:
        ax.text(0.5, 0.99, title, transform=ax.transAxes, ha="center", va="top",
                fontsize=12.5, fontweight="bold", color="#2b2b2b")
    ax.text(0.5, 0.015, "midsagittal side view · lips at left",
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
            color="#8a8a8a", style="italic")

    # dynamic artists -------------------------------------------------------
    (tongue_line,) = ax.plot([], [], "-", color=C_TONGUE, lw=3.2, zorder=5,
                             solid_capstyle="round")
    tongue_fill = ax.fill([], [], color=C_FILL, alpha=0.45, zorder=4, lw=0)[0]
    tongue_pts = ax.scatter([], [], s=55, color=C_TONGUE, zorder=6,
                            edgecolors="white", linewidths=1.0)
    lip_pts = ax.scatter([], [], s=70, color=C_LIP, zorder=6,
                         edgecolors="white", linewidths=1.0)
    jaw_pts = ax.scatter([], [], s=70, color=C_JAW, zorder=6,
                         edgecolors="white", linewidths=1.0)
    time_txt = ax.text(0.985, 0.02, "", transform=ax.transAxes, ha="right",
                       va="bottom", fontsize=9, color="#555555",
                       family="monospace")

    # simple legend
    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_TONGUE,
               markersize=8, label="tongue (tip/body/dorsum)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_LIP,
               markersize=8, label="lips (upper/lower)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_JAW,
               markersize=8, label="jaw (lower incisor)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=8,
              framealpha=0.85, borderpad=0.6)

    # faithfulness strip ----------------------------------------------------
    fneedle = None
    if has_strip:
        tt = np.linspace(0, T / fps, num=len(faith))
        axf.plot(tt, faith, "-", color="#6a5acd", lw=1.6)
        axf.fill_between(tt, 0, faith, color="#6a5acd", alpha=0.15)
        axf.set_ylim(0, 1)
        axf.set_xlim(tt[0], tt[-1])
        axf.set_yticks([0, 0.5, 1])
        axf.tick_params(labelsize=7)
        axf.set_ylabel("recon.\nconfidence", fontsize=7)
        axf.set_xlabel("time (s)", fontsize=7)
        axf.grid(alpha=0.25)
        axf.text(0.5, 1.02,
                 "EXPERIMENTAL reconstruction-confidence — not clinically validated",
                 transform=axf.transAxes, ha="center", va="bottom", fontsize=7.5,
                 color="#6a5acd", style="italic")
        (fneedle,) = axf.plot([tt[0], tt[0]], [0, 1], "-", color="#c0392b", lw=1.4)

    canvas = fig.canvas

    def _frame(t: int) -> np.ndarray:
        row = ema[t]
        pos = {name: _articulator_xy(row, name) for name in IDX}
        # tongue contour front->back with fixed anchors
        contour_pts = [TONGUE_FRONT_ANCHOR, pos["TT"], pos["TB"], pos["TD"],
                       TONGUE_ROOT_ANCHOR]
        cx, cy = _tongue_contour(contour_pts, n=60)
        tongue_line.set_data(cx, cy)
        # filled tongue body: top contour + base underside back-to-front
        base = list(reversed(TONGUE_BASE))
        poly_x = np.concatenate([cx, [TONGUE_ROOT_ANCHOR[0]], [b[0] for b in base],
                                 [TONGUE_FRONT_ANCHOR[0]]])
        poly_y = np.concatenate([cy, [TONGUE_ROOT_ANCHOR[1]], [b[1] for b in base],
                                 [TONGUE_FRONT_ANCHOR[1]]])
        tongue_fill.set_xy(np.column_stack([poly_x, poly_y]))
        tongue_pts.set_offsets([pos["TT"], pos["TB"], pos["TD"]])
        lip_pts.set_offsets([pos["UL"], pos["LL"]])
        jaw_pts.set_offsets([pos["LI"]])
        time_txt.set_text(f"{t / fps:5.2f}s / {T / fps:4.1f}s")
        if fneedle is not None:
            ct = t / fps
            fneedle.set_data([ct, ct], [0, 1])
        canvas.draw()
        buf = np.asarray(canvas.buffer_rgba())[:, :, :3]
        # even dimensions required by libx264
        h, w = buf.shape[:2]
        return buf[: h - (h % 2), : w - (w % 2)]

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    silent = out_path if audio is None else str(
        Path(tempfile.gettempdir()) / f"_veritas_silent_{Path(out_path).stem}.mp4"
    )
    writer = imageio.get_writer(
        silent, fps=fps, codec="libx264", quality=8,
        ffmpeg_log_level="error", macro_block_size=1,
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    try:
        for t in range(T):
            writer.append_data(_frame(t))
    finally:
        writer.close()
        plt.close(fig)

    if audio is None:
        return silent

    # mux the (trimmed) audio so the movie plays with sound, in sync -------
    import soundfile as sf

    dur = T / fps
    aud = np.asarray(audio, dtype=np.float32).ravel()
    n = int(round(dur * sr))
    aud = aud[:n] if len(aud) > n else np.pad(aud, (0, max(0, n - len(aud))))
    tmp_wav = str(Path(tempfile.gettempdir()) / f"_veritas_audio_{Path(out_path).stem}.wav")
    sf.write(tmp_wav, aud, sr)
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-i", silent, "-i", tmp_wav,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except Exception:
        # muxing failed -> return the silent video rather than nothing
        return silent
    finally:
        try:
            Path(tmp_wav).unlink()
        except OSError:
            pass
        try:
            if silent != out_path:
                Path(silent).unlink()
        except OSError:
            pass
    return out_path
