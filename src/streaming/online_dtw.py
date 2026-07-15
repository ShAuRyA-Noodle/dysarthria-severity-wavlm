"""Online (streaming) DTW for streaming ATD (Track A).

Algorithm: classic Dynamic Time Warping with a CLOSED ``(0, 0)`` start and the
warping path constrained to END at ``(T_query - 1, T_ref - 1)`` — i.e. the SAME
quantity computed offline by :func:`src.atd_metric.compute_single_atd`
(``fastdtw(utt, ref, dist=euclidean)`` then ``/ max(T_utt, T_ref)``). It is
processed INCREMENTALLY one query frame at a time, maintaining a single DP
column over the reference rows, so it is genuinely online/streaming with
``O(|R|)`` memory and ``O(|R|)`` work per query frame (independent of how long
the user has been speaking).

History
-------
The original prototype implemented *subsequence* DTW with a FREE initial column
(``D[i, 0] = local_cost(0, i)`` for every reference row ``i``), which lets the
alignment start anywhere in the reference. That is a different quantity than the
offline ATD (audit measured ~28% divergence; published parity Pearson r ≈ 0.81
with 23.5% of utterances off by >2%). This module now computes the exact offline
DTW: with the default (unbanded) path, the final accumulated cost divided by
``max(T_query, T_ref)`` equals the offline ATD to floating-point precision when
compared against an exact full-DP DTW reference.

Run as a script for the smoke test (requires a real TORGO wav + dysarthria env).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INF = np.float64(np.inf)


class OnlineSubseqDTW:
    """Streaming classic DTW against a fixed reference trajectory.

    Despite the historical class name (kept for API compatibility with
    ``src.streaming.session`` / ``server`` and existing tests), this computes
    *classic* DTW — closed ``(0, 0)`` start, path ending at
    ``(T_query - 1, |R| - 1)`` — NOT subsequence DTW. The default (unbanded)
    final cost equals the exact offline DTW used by the offline ATD metric.

    Parameters
    ----------
    reference : np.ndarray
        ``(R, D)`` reference trajectory in PCA-64 space (e.g. from
        ``src.reference_builder.build_word_reference``).
    radius : int
        Sakoe-Chiba band half-width. **NOTE:** the band is an *approximation*.
        When a finite band is applied it can only *increase* the accumulated
        cost (it forbids otherwise-optimal off-diagonal cells), so a banded run
        does NOT in general equal the exact offline DTW. To match the offline
        metric exactly, pass ``radius=None`` (or a radius ``>= max(R, T_query)``,
        which makes the band inactive). The default ``radius=None`` is exact.

    Notes
    -----
    Classic DTW recurrence with the standard 3 predecessors
    (``D[i-1, j]``, ``D[i, j-1]``, ``D[i-1, j-1]``) and unit step weights, so the
    accumulated cost matches ``fastdtw(..., dist=euclidean)`` (which uses the
    same recurrence) up to fastdtw's radius approximation. Compared against an
    exact full-matrix DP DTW the two agree to ~1e-9.
    """

    def __init__(self, reference: np.ndarray, radius: int | None = None) -> None:
        if reference.ndim != 2:
            raise ValueError(f"reference must be (R, D); got {reference.shape}")
        self.R = reference.astype(np.float64, copy=False)
        self.Rn = reference.shape[0]
        # radius=None  -> exact (no band). A finite radius applies an optional
        # Sakoe-Chiba band, which *changes* (raises) the value; see class docs.
        self.radius = None if radius is None else int(radius)
        self.prev_col = np.full(self.Rn, INF, dtype=np.float64)
        self.curr_col = np.full(self.Rn, INF, dtype=np.float64)
        self.t = -1
        self.band_center = 0
        self.last_atd = float("nan")

    def extend(self, u_t: np.ndarray) -> float:
        """Advance the alignment by one query frame ``u_t`` of shape ``(D,)``.

        Returns the running ATD ``D[t, |R|-1] / max(t+1, |R|)`` — the classic
        DTW accumulated cost to the END of the reference, length-normalised
        exactly as the offline ATD metric.

        The column update is ``O(|R|)``; with the default unbanded path the
        final value equals the exact offline DTW. A finite ``radius`` restricts
        each query frame's reachable reference rows to a Sakoe-Chiba band around
        the diagonal, which lowers per-frame work but *changes the value*
        (cells outside the band are treated as unreachable), so it is only an
        approximation of the offline metric.
        """
        self.t += 1
        u_t = u_t.astype(np.float64, copy=False)
        diff = self.R - u_t
        c = np.sqrt(np.einsum("ij,ij->i", diff, diff))

        # Optional Sakoe-Chiba band: reference rows reachable at this query
        # frame. Exact (default) path uses the full reference range.
        if self.radius is None:
            lo, hi = 0, self.Rn  # [lo, hi)
        else:
            # Band centred on the diagonal mapping query index t -> ref index.
            # Use max(Tq, R) as the warping span; since Tq is unknown online we
            # widen conservatively to the running t and |R|.
            span = max(self.t + 1, self.Rn)
            center = int(round(self.t * (self.Rn - 1) / max(span - 1, 1)))
            lo = max(0, center - self.radius)
            hi = min(self.Rn, center + self.radius + 1)

        self.curr_col.fill(INF)
        if self.t == 0:
            # CLOSED (0,0) start: D[0, j] = sum_{k<=j} c[k] (cumulative along ref).
            # Restricted to the band when banded; row 0 must include col 0.
            lo0 = 0  # the closed start forces the path through (0, 0)
            self.curr_col[lo0:hi] = np.cumsum(c[lo0:hi])
            # Cells before the band start (if banded and lo>0) stay INF, which is
            # consistent: an exact path through (0,0) cannot skip reference rows.
        else:
            prev = self.prev_col
            # Column-incremental classic DTW recurrence.
            j0 = max(1, lo)
            if lo == 0:
                # j = 0: only the "left" predecessor (D[i-1, 0]) is valid.
                self.curr_col[0] = c[0] + prev[0]
            for j in range(j0, hi):
                up = self.curr_col[j - 1]      # D[i,   j-1]
                diag = prev[j - 1]             # D[i-1, j-1]
                left = prev[j]                 # D[i-1, j]
                self.curr_col[j] = c[j] + min(up, diag, left)

        finite = np.isfinite(self.curr_col)
        self.band_center = int(np.argmin(np.where(finite, self.curr_col, INF)))
        denom = max(self.t + 1, self.Rn)
        self.last_atd = float(self.curr_col[self.Rn - 1] / denom)
        # Swap buffers (rolling 2-column); next call re-fills curr_col.
        self.prev_col, self.curr_col = self.curr_col, self.prev_col
        return self.last_atd

    @property
    def accumulated_cost(self) -> float:
        """Final un-normalised DTW accumulated cost ``D[t, |R|-1]``.

        After the last query frame this equals the exact offline DTW distance
        (unbanded). ``last_atd`` is this divided by ``max(t+1, |R|)``.
        """
        # prev_col holds the most recently computed column after the buffer swap.
        return float(self.prev_col[self.Rn - 1])


def exact_dtw_full(query: np.ndarray, reference: np.ndarray) -> float:
    """Exact classic DTW via a full ``(T_query, T_ref)`` DP matrix.

    Reference implementation for correctness tests: closed ``(0, 0)`` start,
    path ending at ``(T_query - 1, T_ref - 1)``, Euclidean local cost, standard
    3-predecessor recurrence with unit weights. ``O(T_query * T_ref)`` time and
    memory — NOT streaming; used only to validate :class:`OnlineSubseqDTW`.

    Returns the accumulated cost ``D[T_query - 1, T_ref - 1]`` (un-normalised).
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    Tq, Tr = len(q), len(r)
    if Tq == 0 or Tr == 0:
        raise ValueError("query and reference must be non-empty")
    D = np.full((Tq, Tr), np.inf, dtype=np.float64)
    # Local cost matrix (Euclidean).
    cost = np.sqrt(((q[:, None, :] - r[None, :, :]) ** 2).sum(axis=2))
    D[0, 0] = cost[0, 0]
    for j in range(1, Tr):
        D[0, j] = D[0, j - 1] + cost[0, j]
    for i in range(1, Tq):
        D[i, 0] = D[i - 1, 0] + cost[i, 0]
    for i in range(1, Tq):
        Di = D[i]
        Dim1 = D[i - 1]
        ci = cost[i]
        for j in range(1, Tr):
            Di[j] = ci[j] + min(Di[j - 1], Dim1[j - 1], Dim1[j])
    return float(D[Tq - 1, Tr - 1])


def online_dtw_distance(query: np.ndarray, reference: np.ndarray,
                        radius: int | None = None) -> float:
    """Convenience: stream ``query`` through :class:`OnlineSubseqDTW`, return
    the final un-normalised accumulated cost (exact offline DTW when unbanded)."""
    dtw = OnlineSubseqDTW(reference, radius=radius)
    for frame in np.asarray(query):
        dtw.extend(frame)
    return dtw.accumulated_cost


def _smoke_test() -> None:
    """Round-trip smoke test on real TORGO audio.

    Loads a real .wav, computes WavLM features, splits into reference + query,
    streams the query through `OnlineSubseqDTW`, prints final ATD + per-frame
    microseconds. NOT a unit test (needs full env + audio); run manually.
    """
    import sys
    import time

    import soundfile as sf

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.constants import FASTDTW_RADIUS
    from src.ssl_extractor import SSLExtractor

    candidates = list(Path("data/raw/torgo").rglob("*.wav"))[:1] or list(
        Path("FC01").rglob("*.wav")
    )[:1]
    if not candidates:
        print("[skip] no real TORGO wav found under data/raw/torgo or FC01/")
        return
    wav_path = candidates[0]
    audio, sr = sf.read(wav_path)
    if sr != 16000:
        print(f"[skip] expected 16 kHz, got {sr} for {wav_path}")
        return

    extractor = SSLExtractor(model_name="wavlm_large", device="auto", use_fp16=False)
    feats_dict = extractor.extract_all_layers(audio.astype(np.float32), 16000, [9])
    feats = feats_dict[9]  # (T, 1024)
    if len(feats) < 4:
        print(f"[skip] utterance too short ({len(feats)} frames)")
        return

    ref = feats[: len(feats) // 2]
    qry = feats[len(feats) // 2 :]

    # Default (unbanded) = exact offline DTW.
    dtw = OnlineSubseqDTW(ref)
    t0 = time.perf_counter()
    scores = [dtw.extend(f) for f in qry]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"frames={len(qry)}  final_ATD={scores[-1]:.4f}  "
        f"per_frame_us={elapsed_ms * 1000.0 / max(len(qry), 1):.1f}"
    )


if __name__ == "__main__":
    _smoke_test()
