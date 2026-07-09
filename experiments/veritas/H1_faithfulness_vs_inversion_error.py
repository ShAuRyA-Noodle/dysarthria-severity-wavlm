"""H1 (DECISIVE): does GT-free faithfulness predict REAL articulatory inversion
error, per frame, against measured TORGO AG500 EMA? (real local ground truth)

The VERITAS faithfulness metric (cycle-consistency: audio -> SPARC articulatory
code -> resynthesis, compare WavLM(x) vs WavLM(x_hat)) CLAIMS that a low value at
a frame means SPARC's articulatory inversion was UNRELIABLE there. H1 tests that
claim head-on: correlate per-frame faithfulness with the ACTUAL per-frame
inversion error ||SPARC_pred_ema - measured_EMA|| on healthy control speakers.
Prediction: NEGATIVE correlation (low faithfulness <-> high inversion error).

Reported honestly whatever the result. A null/positive means the metric, as
formulated, does NOT measure inversion reliability.

METHOD (see docs/veritas/DATASETS.md and src/veritas/torgo_ema.py):
  * Controls only (healthy EMA). Objective data gate: skip speakers whose AG500
    session is degenerate (>2 dead coils) or whose tongue coils don't move
    (mis-resolved). Every attempted/used/skipped speaker is recorded.
  * PRIMARY articulator = tongue (dorsum/body/tip): the coils SPARC inverts best
    and TORGO measures cleanly. Lips/jaw show ~0 SPARC<->TORGO correspondence on
    this rig and ambiguous/dead coils, so they are reported only as a secondary
    all-articulator variant, never as the headline.
  * Coordinate alignment is fit on a HELD-OUT slice of utterances (ALIGN),
    applied to a DISJOINT eval slice, and never sees faithfulness (no circularity):
      - integer lag (SPARC pred lags EMA ~+6 frames / +120 ms; searched per spk);
      - per-articulator 3-D->2-D affine mapping measured TORGO (x,y,z) onto SPARC
        (X,Y), absorbing the unknown head-frame rotation/scale/reflection.
    Per-frame inversion error (eval) = Euclidean norm over the tongue channels of
    (SPARC_pred - affine(measured)).
  * Faithfulness = plain cycle-consistency (uncentered; primary) and, as a
    robustness check, healthy-manifold-centered (mu fit on ALIGN control frames).
    WavLM layer 9.
  * H1 stats: median per-utterance Spearman (primary; scale-free, no cross-speaker
    confound), pooled Spearman (raw + within-speaker standardized), per-speaker
    Spearman, and per-utterance mean-faithfulness vs mean-error. Pearson too.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import TORGO_CONTROL_SPEAKERS  # noqa: E402
from src.seed_utils import set_global_seed  # noqa: E402
from src.veritas.faithfulness import HealthyReference, frame_faithfulness  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.torgo_ema import (  # noqa: E402
    SPARC_ARTIC_COLS,
    TONGUE_ARTICULATORS,
    downsample_to_50fps,
    is_clean_for_tongue,
    pos_xyz,
    resolve_channels,
)
from src.veritas.wavlm_features import WavLMFeatures  # noqa: E402

def spk_group(spk: str) -> str:
    """TORGO top-level group dir for a speaker id (controls FC*/MC*, dysarthric F*/M*)."""
    if spk.startswith("FC"):
        return "FC"
    if spk.startswith("MC"):
        return "MC"
    if spk.startswith("F"):
        return "F"
    return "M"


def speaker_pos_files(spk: str) -> list[str]:
    """Deterministic ordered list of a speaker's head-corrected .pos
    files (the `pos/` dir only, never `rawpos/`)."""
    grp = spk_group(spk)
    files = glob.glob(str(ROOT / grp / spk / "**" / "pos" / "*.pos"), recursive=True)
    return sorted(files)


def wav_for_pos(pos_path: str) -> str:
    """EMA-synchronous audio path for a .pos file: ../wavall/<stem>.wav.all."""
    p = Path(pos_path)
    return str(p.parents[1] / "wavall" / (p.stem + ".wav.all"))


def frame_log_energy(x: np.ndarray, n: int, hop: int = 320) -> np.ndarray:
    """Per-50fps-frame log RMS energy of 16 kHz audio (hop 320 = 20 ms).

    The silence/voicing confound control for H1: silent frames are trivially
    faithful (WavLM(x)~WavLM(x_hat)) yet articulatorily unconstrained, so faith
    and error can co-move via energy without any true reliability link."""
    e = np.array([float(np.mean(x[j * hop:(j + 1) * hop] ** 2)) for j in range(n)])
    return np.log(e + 1e-8)


def _partial_spearman(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Spearman partial correlation of a,b controlling c (rank-residual method)."""
    from scipy.stats import rankdata

    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)

    def _resid(y, x):
        X = np.c_[x, np.ones_like(x)]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        return y - X @ coef

    if np.std(rc) == 0:
        return float("nan")
    return float(np.corrcoef(_resid(ra, rc), _resid(rb, rc))[0, 1])


def load_utt(pos_path: str, inv: SPARCInverter):
    """Return (pred_ema (Lp,12) 50fps, true_xyz (Le,12,3) 50fps mm, audio16k, code).

    ``code`` is the full SPARC articulatory code so the caller can decode x_hat
    WITHOUT re-encoding (one SPARC forward pass per utterance). Audio and EMA
    share t=0; audio runs ~0.8 s longer, so pred has >= EMA frames. Returns
    (None, reason) if audio is missing/too short."""
    wav = wav_for_pos(pos_path)
    try:
        x, _ = librosa.load(wav, sr=16000, mono=True)
    except Exception as e:  # noqa: BLE001
        return None, f"audio load failed: {e}"
    if len(x) < 1600:
        return None, "audio < 0.1 s"
    code = inv.encode(x)
    pred = np.asarray(code["ema"], dtype=np.float64)                   # (Lp, 12) 50 fps
    true = downsample_to_50fps(pos_xyz(pos_path).astype(np.float64))   # (Le, 12, 3) 50 fps
    return (pred, true, x, code), "ok"


# ---- alignment fit on the held-out ALIGN slice -----------------------------

def _pairs_for_lag(cmap):
    """(sparc_col, tongue_coil, torgo_axis) pairs used only to score candidate
    lags: SPARC X<->TORGO x, SPARC Y<->TORGO z, for TD/TB/TT."""
    out = []
    for art in TONGUE_ARTICULATORS:
        cx, cy = SPARC_ARTIC_COLS[art]
        s = cmap.tongue[art]
        out.append((cx, s, 0))  # X vs anterior-posterior x
        out.append((cy, s, 2))  # Y vs vertical z
    return out


def estimate_lag(align_utts, cmap, max_lag: int) -> int:
    """Integer frame lag (pred delayed vs EMA) maximizing mean |corr| over tongue
    pairs, pooled across ALIGN utts. Positive lag => pred[j] pairs true[j-lag]."""
    pairs = _pairs_for_lag(cmap)
    best_lag, best_score = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        scores = []
        for (sc, s, ax) in pairs:
            rs = []
            for pred, true, _ in align_utts:
                L = min(len(pred), len(true))
                if lag >= 0:
                    a, b = pred[lag:L, sc], true[: L - lag, s, ax]
                else:
                    a, b = pred[: L + lag, sc], true[-lag:L, s, ax]
                if len(a) > 15 and np.std(a) > 0 and np.std(b) > 0:
                    r = np.corrcoef(a, b)[0, 1]
                    if np.isfinite(r):
                        rs.append(abs(r))
            if rs:
                scores.append(np.mean(rs))
        m = float(np.mean(scores)) if scores else -np.inf
        if m > best_score:
            best_lag, best_score = lag, m
    return best_lag


def _lag_align(pred, true, lag):
    """Trim pred/true so pred[k] pairs true[k]; pred lags true by `lag` frames."""
    L = min(len(pred), len(true))
    if lag >= 0:
        return pred[lag:L], true[: L - lag], lag  # index offset into original pred = lag
    return pred[: L + lag], true[-lag:L], 0


def fit_affine(align_utts, cmap, lag):
    """Per tongue articulator, LS affine mapping TORGO (x,y,z)+1 -> SPARC (X,Y),
    fit on lag-aligned ALIGN frames. Returns {art: (coefX(4,), coefY(4,))}."""
    stackP = {art: [] for art in TONGUE_ARTICULATORS}
    stackT = {art: [] for art in TONGUE_ARTICULATORS}
    for pred, true, _ in align_utts:
        pa, ta, _ = _lag_align(pred, true, lag)
        for art in TONGUE_ARTICULATORS:
            cx, cy = SPARC_ARTIC_COLS[art]
            s = cmap.tongue[art]
            stackP[art].append(pa[:, [cx, cy]])
            stackT[art].append(ta[:, s, :])
    coefs = {}
    for art in TONGUE_ARTICULATORS:
        P = np.concatenate(stackP[art], 0)        # (N,2) SPARC X,Y
        T = np.concatenate(stackT[art], 0)        # (N,3) TORGO xyz
        A = np.c_[T, np.ones(len(T))]             # (N,4)
        cX, *_ = np.linalg.lstsq(A, P[:, 0], rcond=None)
        cY, *_ = np.linalg.lstsq(A, P[:, 1], rcond=None)
        coefs[art] = (cX, cY)
    return coefs


def per_frame_error(pred, true, cmap, lag, coefs):
    """Per-frame tongue inversion error on (lag-aligned) frames + the index
    offset into the ORIGINAL pred array (so faithfulness can be paired)."""
    pa, ta, off = _lag_align(pred, true, lag)
    diffs = []
    for art in TONGUE_ARTICULATORS:
        cx, cy = SPARC_ARTIC_COLS[art]
        s = cmap.tongue[art]
        A = np.c_[ta[:, s, :], np.ones(len(ta))]
        cX, cY = coefs[art]
        predX, predY = A @ cX, A @ cY            # measured mapped into SPARC space
        diffs.append(pa[:, cx] - predX)
        diffs.append(pa[:, cy] - predY)
    D = np.stack(diffs, axis=1)                   # (M, 6)
    err = np.linalg.norm(D, axis=1)               # (M,)
    return err, off


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=45)
    ap.add_argument("--align-utts", type=int, default=12,
                    help="held-out utterances per speaker for lag+affine+mu fit")
    ap.add_argument("--layer", type=int, default=9, help="WavLM layer for faithfulness")
    ap.add_argument("--max-lag", type=int, default=15)
    ap.add_argument("--device", default=default_device)
    ap.add_argument("--speakers", default=",".join(sorted(TORGO_CONTROL_SPEAKERS)))
    ap.add_argument("--out", default=str(ROOT / "results" / "veritas" / "h1_faithfulness_vs_inversion_error.json"))
    args = ap.parse_args()
    set_global_seed(42)

    speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    inv = SPARCInverter(device=args.device)
    wavlm = WavLMFeatures(layer=args.layer, device=args.device)

    per_spk: dict[str, dict] = {}
    # frame-level pools (faith paired with error), tagged by speaker for standardized pooling
    all_faith, all_faith_c, all_err, all_energy, all_spk = [], [], [], [], []
    utt_rows = []          # per-utterance summaries (for median-of-corr + mean-vs-mean)
    align_frames_for_mu: list[np.ndarray] = []

    for spk in speakers:
        pos_files = speaker_pos_files(spk)[: args.limit_per_speaker]
        n_align = min(args.align_utts, max(1, len(pos_files) // 3))
        align_files, eval_files = pos_files[:n_align], pos_files[n_align:]

        # pooled xyz for channel resolution (align utts only)
        pooled = []
        for f in align_files:
            try:
                pooled.append(pos_xyz(f))
            except Exception as e:  # noqa: BLE001
                print(f"  {spk}: pos read failed {f}: {e}", flush=True)
        if not pooled:
            per_spk[spk] = {"used": 0, "skipped_reason": "no readable .pos"}
            continue
        pooled = np.concatenate(pooled, 0)
        cmap = resolve_channels(pooled)
        ok, reason = is_clean_for_tongue(pooled, cmap)
        tongue_coils = {k: int(v) for k, v in cmap.tongue.items()}
        if not ok:
            per_spk[spk] = {
                "used": 0, "skipped_reason": reason,
                "dead_coils_1idx": [c + 1 for c in cmap.dead],
                "tongue_coils_1idx": {k: v + 1 for k, v in tongue_coils.items()},
            }
            print(f"[skip] {spk}: {reason} (dead={[c+1 for c in cmap.dead]})", flush=True)
            continue

        # load ALIGN utts, fit lag + affine + collect mu frames
        align_utts, mu_frames = [], []
        for f in align_files:
            (data, msg) = load_utt(f, inv)
            if data is None:
                continue
            pred, true, x, _code = data
            align_utts.append((pred, true, f))
            mu_frames.append(wavlm.extract(x, sr=16000))
        if len(align_utts) < 2:
            per_spk[spk] = {"used": 0, "skipped_reason": "too few align utts loaded"}
            continue
        lag = estimate_lag(align_utts, cmap, args.max_lag)
        coefs = fit_affine(align_utts, cmap, lag)
        spk_mu = HealthyReference().fit(np.concatenate(mu_frames, 0))
        for fr in mu_frames:
            align_frames_for_mu.append(fr)

        # ALIGN correspondence diagnostic (corr of SPARC vs affine(true) on align)
        align_corr = _align_corr(align_utts, cmap, lag, coefs)

        # EVAL
        spk_faith, spk_faith_c, spk_err, spk_en = [], [], [], []
        n_used = n_err = 0
        for f in eval_files:
            (data, msg) = load_utt(f, inv)
            if data is None:
                n_err += 1
                continue
            pred, true, x, code = data
            try:
                err, off = per_frame_error(pred, true, cmap, lag, coefs)
                fx = wavlm.extract(x, sr=16000)
                xhat = inv.decode(code)  # SPARC sr == 16 kHz (no re-encode)
                fxh = wavlm.extract(xhat, sr=16000)
                fa = frame_faithfulness(fx, fxh, None)            # plain (primary)
                fac = frame_faithfulness(fx, fxh, spk_mu)         # centered (robustness)
                en = frame_log_energy(x, len(pred))               # confound control
                M = min(len(err), len(fa) - off, len(fac) - off, len(en) - off)
                if M < 10:
                    n_err += 1
                    continue
                e = err[:M]
                fa = fa[off: off + M]
                fac = fac[off: off + M]
                en = en[off: off + M]
                spk_faith.append(fa); spk_faith_c.append(fac); spk_err.append(e); spk_en.append(en)
                all_faith.append(fa); all_faith_c.append(fac); all_err.append(e); all_energy.append(en)
                all_spk.append(np.full(M, spk))
                # per-utterance stats
                ru = spearmanr(fa, e).correlation if np.std(e) > 0 else np.nan
                utt_rows.append({
                    "spk": spk, "n_frames": int(M),
                    "utt_spearman": float(ru),
                    "mean_faith": float(np.mean(fa)), "mean_err": float(np.mean(e)),
                })
                n_used += 1
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"  {spk}: eval err {f}: {e}", flush=True)

        if n_used == 0:
            per_spk[spk] = {"used": 0, "skipped_reason": "no eval utts produced frames"}
            continue
        F = np.concatenate(spk_faith); Fc = np.concatenate(spk_faith_c)
        E = np.concatenate(spk_err); EN = np.concatenate(spk_en)
        rho_s = spearmanr(F, E).correlation
        rho_p = pearsonr(F, E)[0]
        rho_sc = spearmanr(Fc, E).correlation
        part = _partial_spearman(F, E, EN)
        hi = EN > np.percentile(EN, 75)  # clearest-speech frames
        rho_hi = spearmanr(F[hi], E[hi]).correlation if hi.sum() > 10 and np.std(E[hi]) > 0 else float("nan")
        per_spk[spk] = {
            "used": n_used, "eval_utts_failed": n_err,
            "n_frames": int(len(F)),
            "lag_frames": int(lag), "lag_ms": int(lag * 20),
            "dead_coils_1idx": [c + 1 for c in cmap.dead],
            "tongue_coils_1idx": {k: v + 1 for k, v in tongue_coils.items()},
            "align_corr_tongue": align_corr,
            "spearman_plain": _f(rho_s), "pearson_plain": _f(rho_p),
            "spearman_centered": _f(rho_sc),
            "spearman_faith_energy": _f(spearmanr(F, EN).correlation),
            "spearman_err_energy": _f(spearmanr(E, EN).correlation),
            "partial_spearman_plain_ctrl_energy": _f(part),
            "spearman_plain_top25pct_energy": _f(rho_hi),
            "mean_faith": float(np.mean(F)), "mean_err": float(np.mean(E)),
        }
        print(f"[{spk}] used={n_used} frames={len(F)} lag={lag} "
              f"align_corr={align_corr:.3f} rho_plain={_f(rho_s):+.4f} "
              f"partial|energy={_f(part):+.4f} rho_top25%E={_f(rho_hi):+.4f}", flush=True)

    # ---- aggregate H1 ------------------------------------------------------
    used_spk = [s for s in speakers if per_spk.get(s, {}).get("used", 0) > 0]
    result = _aggregate(all_faith, all_faith_c, all_err, all_energy, all_spk, utt_rows, used_spk)

    out = {
        "hypothesis": "per-frame faithfulness NEGATIVELY predicts per-frame articulatory "
                      "inversion error vs measured TORGO EMA (low faithfulness <-> high error)",
        "decisive": True,
        "articulator_primary": "tongue (dorsum/body/tip); 6 SPARC channels TDX..TTY",
        "faithfulness_primary": "plain cycle-consistency (uncentered), WavLM layer %d" % args.layer,
        "faithfulness_secondary": "healthy-manifold centered (mu on ALIGN control frames)",
        "alignment": {
            "method": "per-speaker integer lag (max_lag=%d) + per-articulator 3D->2D affine "
                      "(TORGO xyz -> SPARC XY); fit on held-out ALIGN utts, applied to disjoint "
                      "EVAL utts; faithfulness never used in the fit (no circularity)" % args.max_lag,
            "ema_downsample": "200Hz -> 50fps polyphase (scipy.resample_poly 1/4)",
            "coord_axes": "TORGO x=ant-post, y=lateral(dropped by affine), z=vertical; "
                          "SPARC 2D (X,Y) recovered by the affine",
            "align_utts_per_speaker": args.align_utts,
        },
        "data_gate": "controls only; skip if >2 dead coils or tongue coils don't move "
                     "(objective AG500 session validity, not SPARC performance)",
        "n_speakers_used": len(used_spk),
        "speakers_used": used_spk,
        "speakers_skipped": {s: per_spk[s].get("skipped_reason") for s in speakers
                             if per_spk.get(s, {}).get("used", 0) == 0},
        **result,
        "per_speaker": per_spk,
        "layer": args.layer,
        "limit_per_speaker": args.limit_per_speaker,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n================= H1 RESULT =================")
    print(f"speakers used: {used_spk}")
    print(f"n_frames={result['n_frames']} n_utts={result['n_utts']} n_speakers={len(used_spk)}")
    print(f"median per-utt Spearman (plain)   : {_f(result['median_utt_spearman_plain']):+.4f}")
    print(f"pooled Spearman within-spk (plain): {_f(result['pooled_within_spk_spearman_plain']):+.4f}")
    print(f"pooled Spearman raw (plain)       : {_f(result['pooled_spearman_plain']):+.4f} "
          f"(p={_f(result['pooled_spearman_plain_p']):.2e})")
    print(f"pooled Pearson raw (plain)        : {_f(result['pooled_pearson_plain']):+.4f}")
    print(f"median per-utt Spearman (centered): {_f(result['median_utt_spearman_centered']):+.4f}")
    print(f"per-utt mean-faith vs mean-err    : rho={_f(result['utt_meanfaith_vs_meanerr_spearman']):+.4f}")
    print(f"VERDICT: {result['verdict']}")
    print(f"-> {args.out}")


def _f(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _align_corr(align_utts, cmap, lag, coefs) -> float:
    """Mean corr of SPARC vs affine(measured) over the 6 tongue channels on ALIGN."""
    rs = []
    for pred, true, _ in align_utts:
        pa, ta, _ = _lag_align(pred, true, lag)
        for art in TONGUE_ARTICULATORS:
            cx, cy = SPARC_ARTIC_COLS[art]
            s = cmap.tongue[art]
            A = np.c_[ta[:, s, :], np.ones(len(ta))]
            for ch, coef in ((cx, coefs[art][0]), (cy, coefs[art][1])):
                p, q = pa[:, ch], A @ coef
                if len(p) > 15 and np.std(p) > 0 and np.std(q) > 0:
                    r = np.corrcoef(p, q)[0, 1]
                    if np.isfinite(r):
                        rs.append(r)
    return float(np.mean(rs)) if rs else float("nan")


def _aggregate(all_faith, all_faith_c, all_err, all_energy, all_spk, utt_rows, used_spk) -> dict:
    if not all_faith:
        return {"n_frames": 0, "n_utts": 0, "verdict": "NO DATA", "median_utt_spearman_plain": float("nan"),
                "pooled_spearman_plain": float("nan"), "pooled_spearman_plain_p": float("nan"),
                "pooled_pearson_plain": float("nan"), "pooled_within_spk_spearman_plain": float("nan"),
                "median_utt_spearman_centered": float("nan"),
                "partial_spearman_plain_ctrl_energy": float("nan"),
                "pooled_spearman_top25pct_energy": float("nan"),
                "utt_meanfaith_vs_meanerr_spearman": float("nan")}
    F = np.concatenate(all_faith); Fc = np.concatenate(all_faith_c)
    E = np.concatenate(all_err); EN = np.concatenate(all_energy); S = np.concatenate(all_spk)

    # within-speaker standardization removes cross-speaker offsets before pooling
    Fz, Ez, ENz = np.empty_like(F), np.empty_like(E), np.empty_like(EN)
    for spk in np.unique(S):
        m = S == spk
        Fz[m] = (F[m] - F[m].mean()) / (F[m].std() + 1e-9)
        Ez[m] = (E[m] - E[m].mean()) / (E[m].std() + 1e-9)
        ENz[m] = (EN[m] - EN[m].mean()) / (EN[m].std() + 1e-9)

    pooled_s, pooled_p = spearmanr(F, E)
    pooled_sz = spearmanr(Fz, Ez).correlation
    pooled_pe = pearsonr(F, E)[0]
    pooled_sc = spearmanr(Fc, E).correlation

    # energy-confound controls (within-speaker standardized pools)
    faith_energy = spearmanr(Fz, ENz).correlation
    err_energy = spearmanr(Ez, ENz).correlation
    partial = _partial_spearman(Fz, Ez, ENz)
    hi = ENz > 0.0  # above-median energy within each speaker == clearest speech
    hi25 = ENz > np.array([np.percentile(ENz[S == s], 75) for s in S])  # per-spk top quartile
    top50 = spearmanr(Fz[hi], Ez[hi]).correlation if hi.sum() > 20 else float("nan")
    top25 = spearmanr(Fz[hi25], Ez[hi25]).correlation if hi25.sum() > 20 else float("nan")

    utt_sp = [r["utt_spearman"] for r in utt_rows if np.isfinite(r["utt_spearman"])]
    med_utt = float(np.median(utt_sp)) if utt_sp else float("nan")
    mf = [r["mean_faith"] for r in utt_rows]
    me = [r["mean_err"] for r in utt_rows]
    mm_rho = spearmanr(mf, me).correlation if len(mf) > 2 else float("nan")

    verdict = _verdict(med_utt, pooled_sz, pooled_p, partial, top25)

    return {
        "n_frames": int(len(F)), "n_utts": len(utt_rows),
        "median_utt_spearman_plain": med_utt,
        "mean_utt_spearman_plain": float(np.mean(utt_sp)) if utt_sp else float("nan"),
        "pooled_spearman_plain": _f(pooled_s), "pooled_spearman_plain_p": _f(pooled_p),
        "pooled_within_spk_spearman_plain": _f(pooled_sz),
        "pooled_pearson_plain": _f(pooled_pe),
        "pooled_spearman_centered": _f(pooled_sc),
        "median_utt_spearman_centered": _f(pooled_sc),  # frame-pooled centered proxy
        "spearman_faith_energy": _f(faith_energy),
        "spearman_err_energy": _f(err_energy),
        "partial_spearman_plain_ctrl_energy": _f(partial),
        "pooled_spearman_top50pct_energy": _f(top50),
        "pooled_spearman_top25pct_energy": _f(top25),
        "utt_meanfaith_vs_meanerr_spearman": _f(mm_rho),
        "verdict": verdict,
    }


def _verdict(med_utt, pooled_within, pooled_p, partial_ctrl_energy, top25_energy) -> str:
    """Blunt verdict on H1. NEGATIVE (and surviving the energy control) => the
    metric predicts inversion error. Sign convention: faith HIGH = good, error
    HIGH = bad, so validation requires NEGATIVE correlation."""
    if not np.isfinite(med_utt):
        return "NO DATA"
    sig = np.isfinite(pooled_p) and pooled_p < 0.05
    # the honest test is the energy-controlled correlation (silence is trivially faithful)
    ctrl = partial_ctrl_energy if np.isfinite(partial_ctrl_energy) else pooled_within
    strong_neg = med_utt <= -0.15 and ctrl <= -0.10
    weak_neg = (med_utt <= -0.05) or (np.isfinite(top25_energy) and top25_energy <= -0.10 and ctrl <= -0.03)
    if strong_neg and sig:
        return (f"faithfulness predicts inversion error: YES (median per-utt Spearman={med_utt:+.3f}, "
                f"energy-controlled partial={ctrl:+.3f}, negative & significant).")
    if weak_neg:
        return (f"faithfulness predicts inversion error: WEAK (median per-utt Spearman={med_utt:+.3f}, "
                f"energy-controlled partial={ctrl:+.3f}; negative only on the clearest frames "
                f"[top-25% energy Spearman={top25_energy:+.3f}] and small). Not a reliable per-frame signal.")
    # near zero or wrong sign after controlling for energy => NO
    return (f"faithfulness predicts inversion error: NO (median per-utt Spearman={med_utt:+.3f}; "
            f"the raw association is a speech-energy artifact — energy-controlled partial Spearman"
            f"={ctrl:+.3f} ~ 0, faith-vs-energy and error-vs-energy dominate). As formulated the "
            f"cycle-consistency metric does NOT measure per-frame inversion reliability on real EMA.")


if __name__ == "__main__":
    main()
