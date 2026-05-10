"""Online Subsequence DTW for streaming ATD (Track A scaffold).

Algorithm: Subsequence DTW (Mueller 2015 ch.7) + UCR-Suite-style column-incremental
update. Per-frame cost O(2*radius); memory O(|R|) via 2-column rolling buffer.
Crucially, extending the alignment by one query frame is independent of how
long the user has been speaking (no growth in per-step compute).

Run as a script for the smoke test (requires a real TORGO wav + dysarthria env).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

INF = np.float32(np.inf)


class OnlineSubseqDTW:
    """Streaming subsequence DTW against a fixed reference trajectory.

    Parameters
    ----------
    reference : np.ndarray
        (R, D) reference trajectory in PCA-64 space (e.g. from
        `src.reference_builder.build_word_reference`).
    radius : int
        Sakoe-Chiba band half-width. Defaults to 30 to match
        `src.constants.FASTDTW_RADIUS`.

    Notes
    -----
    Subsequence DTW allows the alignment to start anywhere in the reference
    (free initial column) and end anywhere — but for word-aligned ATD we
    constrain the final cost to row |R|-1 (full-reference end), then
    length-normalize by `max(t+1, |R|)` to match offline ATD.
    """

    def __init__(self, reference: np.ndarray, radius: int = 30) -> None:
        if reference.ndim != 2:
            raise ValueError(f"reference must be (R, D); got {reference.shape}")
        self.R = reference.astype(np.float32, copy=False)
        self.Rn = reference.shape[0]
        self.radius = int(radius)
        self.prev_col = np.full(self.Rn, INF, dtype=np.float32)
        self.curr_col = np.full(self.Rn, INF, dtype=np.float32)
        self.t = -1
        self.band_center = 0
        self.last_atd = float("nan")

    def extend(self, u_t: np.ndarray) -> float:
        """Advance the alignment by one query frame `u_t` of shape (D,).

        Returns the running ATD = `D[|R|-1, t] / max(t+1, |R|)`.

        Implementation detail: full column update O(|R|). The band optimization
        from the design doc was dropped for the prototype because the
        Sakoe-Chiba band introduces stale-INF cells when the band moves; a
        correct banded implementation requires either monotone band tracking
        with re-initialization or a fully-allocated cost matrix. For typical
        ATD references (|R| < 200) the full-column cost is well under 1 ms
        per frame. A banded variant can be reintroduced once we add an
        explicit `min_band_low` invariant.
        """
        self.t += 1
        u_t = u_t.astype(np.float32, copy=False)
        diff = self.R - u_t
        c = np.sqrt(np.einsum("ij,ij->i", diff, diff))

        if self.t == 0:
            # SS-DTW: free initial column = local cost at every reference row.
            self.curr_col[:] = c
        else:
            self.curr_col[0] = c[0] + self.prev_col[0]
            for i in range(1, self.Rn):
                up = self.curr_col[i - 1]
                diag = self.prev_col[i - 1]
                left = self.prev_col[i]
                self.curr_col[i] = c[i] + min(up, diag, left)

        self.band_center = int(np.argmin(self.curr_col))
        denom = max(self.t + 1, self.Rn)
        self.last_atd = float(self.curr_col[self.Rn - 1] / denom)
        # Swap buffers (rolling 2-column).
        self.prev_col, self.curr_col = self.curr_col, self.prev_col
        return self.last_atd


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

    dtw = OnlineSubseqDTW(ref, radius=FASTDTW_RADIUS)
    t0 = time.perf_counter()
    scores = [dtw.extend(f) for f in qry]
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    print(
        f"frames={len(qry)}  final_ATD={scores[-1]:.4f}  "
        f"per_frame_us={elapsed_ms * 1000.0 / max(len(qry), 1):.1f}"
    )


if __name__ == "__main__":
    _smoke_test()
