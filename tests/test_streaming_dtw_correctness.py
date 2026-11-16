"""Correctness tests for the streaming (online) DTW (Track A, DEFECT 1).

The online/streaming DTW MUST compute the SAME quantity as the offline ATD:
classic DTW with a CLOSED (0,0) start and the path ending at
(T_query-1, T_ref-1), Euclidean local cost, processed one query frame at a time.

These tests compare the streaming ``OnlineSubseqDTW`` against an EXACT
full-matrix DP DTW reference (``exact_dtw_full``) on REAL PCA-reduced control
reference trajectories loaded from ``data/processed/references/*.pkl``. They use
trajectories of DIFFERING lengths (query vs reference) and assert agreement to
1e-6.

REGRESSION GUARD: the historical bug initialised the first query column with a
FREE initial column (``D[i,0] = local_cost(0,i)`` for every reference row i),
i.e. subsequence DTW that may start anywhere in the reference. That makes the
final accumulated cost SMALLER than the closed-start classic DTW. The
``test_*_rejects_free_initial_column`` test reconstructs that buggy variant and
asserts it does NOT match the exact DTW, so reintroducing the bug fails CI.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.streaming.online_dtw import (  # noqa: E402
    OnlineSubseqDTW,
    exact_dtw_full,
    online_dtw_distance,
)

REF_DIR = PROJECT_ROOT / "data" / "processed" / "references"


def _load_real_reference_trajectories(max_refs: int = 4) -> list[tuple[str, np.ndarray]]:
    """Load a few REAL PCA-reduced control reference trajectories (T_ref, 64).

    These are the exact ``reference`` arrays the offline ATD metric warps
    against (built from control speakers, projected into PCA-64 space).
    """
    pkls = sorted(REF_DIR.glob("*_layer9.pkl"))
    out: list[tuple[str, np.ndarray]] = []
    for p in pkls:
        if len(out) >= max_refs:
            break
        try:
            with open(p, "rb") as f:
                d = pickle.load(f)
        except Exception:
            continue
        ref = np.asarray(d["reference"], dtype=np.float64)
        if ref.ndim == 2 and ref.shape[0] >= 8 and ref.shape[1] == 64:
            out.append((p.stem, ref))
    return out


def _make_query_from_reference(reference: np.ndarray, n_query: int, seed: int) -> np.ndarray:
    """Derive a real-valued query of a DIFFERENT length from a real reference.

    We resample the reference trajectory to ``n_query`` rows (linear
    interpolation along the time axis) and add a small deterministic
    perturbation. This yields a genuine PCA-space trajectory of a length that
    differs from the reference, exercising the off-diagonal warping that the
    free-initial-column bug got wrong.
    """
    rng = np.random.default_rng(seed)
    t_ref = reference.shape[0]
    src_idx = np.linspace(0.0, t_ref - 1.0, n_query)
    lo = np.floor(src_idx).astype(int)
    hi = np.minimum(lo + 1, t_ref - 1)
    frac = (src_idx - lo)[:, None]
    query = (1.0 - frac) * reference[lo] + frac * reference[hi]
    query = query + 0.05 * rng.standard_normal(query.shape) * reference.std()
    return query.astype(np.float64)


def test_real_reference_data_present() -> None:
    """Guard: the test exercises REAL data, not a synthetic fallback."""
    refs = _load_real_reference_trajectories()
    assert len(refs) >= 2, (
        "expected real PCA-reduced reference trajectories under "
        f"{REF_DIR}; found {len(refs)}"
    )
    for _, ref in refs:
        assert ref.shape[1] == 64
        assert np.isfinite(ref).all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_online_dtw_equals_exact_full_dp_on_real_trajectories(seed: int) -> None:
    """Online streaming DTW final cost == exact full-DP DTW within 1e-6.

    Runs on REAL reference trajectories with queries of DIFFERING lengths
    (shorter AND longer than the reference).
    """
    refs = _load_real_reference_trajectories()
    assert refs, "no real reference trajectories found"

    checked = 0
    max_err = 0.0
    for name, ref in refs:
        t_ref = ref.shape[0]
        # Query lengths that differ from the reference, both shorter and longer.
        for n_query in (max(4, t_ref // 2), t_ref - 3, t_ref + 7):
            if n_query < 2:
                continue
            query = _make_query_from_reference(ref, n_query, seed=seed + n_query)

            exact = exact_dtw_full(query, ref)

            dtw = OnlineSubseqDTW(ref)  # default = unbanded = exact
            for frame in query:
                dtw.extend(frame)
            online_acc = dtw.accumulated_cost

            err = abs(online_acc - exact)
            max_err = max(max_err, err)
            assert err < 1e-6, (
                f"{name}: online={online_acc!r} exact={exact!r} "
                f"|err|={err:.3e} (T_query={n_query}, T_ref={t_ref})"
            )

            # Length-normalised ATD must equal exact/ max(Tq, Tr) too.
            denom = max(n_query, t_ref)
            assert dtw.last_atd == pytest.approx(exact / denom, rel=1e-9, abs=1e-9)
            checked += 1

    assert checked >= 4, f"too few comparisons exercised ({checked})"
    # Real, observed max error for the deliverable (printed when -s is passed).
    print(f"[online-vs-exact] checked={checked} max_abs_err={max_err:.3e}")


def test_online_dtw_helper_matches_exact() -> None:
    """The ``online_dtw_distance`` convenience equals the exact full-DP DTW."""
    refs = _load_real_reference_trajectories(max_refs=2)
    assert refs
    name, ref = refs[0]
    query = _make_query_from_reference(ref, ref.shape[0] + 5, seed=99)
    assert online_dtw_distance(query, ref) == pytest.approx(
        exact_dtw_full(query, ref), rel=1e-9, abs=1e-9
    )


def _buggy_free_initial_column_dtw(query: np.ndarray, reference: np.ndarray) -> float:
    """Reconstruct the ORIGINAL bug: subsequence DTW with a FREE initial column.

    First query column = local cost at every reference row (alignment may start
    anywhere in the reference). Returns the accumulated cost at the reference
    end after the last query frame. Used only to PROVE the corrected algorithm
    differs from the bug.
    """
    q = np.asarray(query, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    Tq, Tr = len(q), len(r)
    prev = np.full(Tr, np.inf)
    curr = np.full(Tr, np.inf)
    for i in range(Tq):
        c = np.sqrt(((r - q[i]) ** 2).sum(axis=1))
        if i == 0:
            curr = c.copy()  # FREE initial column (the bug)
        else:
            curr = np.full(Tr, np.inf)
            curr[0] = c[0] + prev[0]
            for j in range(1, Tr):
                curr[j] = c[j] + min(curr[j - 1], prev[j - 1], prev[j])
        prev = curr.copy()
    return float(curr[Tr - 1])


def test_corrected_dtw_rejects_free_initial_column_bug() -> None:
    """REGRESSION GUARD: the corrected DTW must NOT equal the free-column bug.

    On real trajectories the free-initial-column (subsequence) variant produces
    a strictly smaller accumulated cost. This test fails if someone reintroduces
    that initialisation into ``OnlineSubseqDTW``.
    """
    refs = _load_real_reference_trajectories()
    assert refs

    saw_difference = False
    for name, ref in refs:
        t_ref = ref.shape[0]
        # Phase-shifted query: the SECOND HALF of the real reference trajectory.
        # Its first frame is closest to the reference MIDPOINT, so the
        # free-initial-column (subsequence) bug can start the alignment there and
        # pay ~0 cost, whereas the correct closed-(0,0) classic DTW must traverse
        # the whole reference. This makes the two genuinely diverge.
        query = ref[t_ref // 2:]

        exact = exact_dtw_full(query, ref)
        buggy = _buggy_free_initial_column_dtw(query, ref)

        # The corrected online DTW matches exact ...
        assert online_dtw_distance(query, ref) == pytest.approx(exact, rel=1e-9, abs=1e-9)
        # ... and the buggy free-column variant is genuinely different (smaller).
        if abs(buggy - exact) > 1e-6:
            saw_difference = True
            assert buggy < exact - 1e-6, (
                f"{name}: free-column cost {buggy} should be strictly smaller "
                f"than classic {exact}"
            )

    assert saw_difference, (
        "free-initial-column bug produced identical results on all real "
        "trajectories — the regression guard is not actually discriminating"
    )
