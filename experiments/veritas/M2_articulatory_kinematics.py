"""M2 — interpretable articulatory-KINEMATIC severity features from frozen-SSL
acoustic-to-articulatory inversion (SPARC), on real TORGO audio.

Per utterance, from the tongue articulators (dorsum TD, body TB, tip TT) of the
SPARC-inverted trajectory (50 fps):
  * range  = mean over tongue articulators of the 2-D positional std (the
             articulatory WORKING SPACE; reduced range = undershoot).
  * speed  = mean frame-to-frame 2-D tongue speed (velocity; slowed in dysarthria).
Aggregated per speaker (mean over utterances).

Validation, honest:
  (a) EMA-GROUNDING: for TORGO speakers whose AG500 session is clean, compute the
      SAME two features from the REAL measured EMA (tongue coils, x=ant-post /
      z=vertical, downsampled 200->50 fps) and correlate SPARC-derived vs
      EMA-derived across speakers. High corr => the SPARC features reflect real
      articulation, not inversion artifacts.
  (b) SEVERITY: Spearman of each SPARC-derived feature vs dysarthria severity
      (0/1/2) across all 15 TORGO speakers.

Everything real (offline cached SPARC/WavLM, real audio + real EMA). Seed 42.
Reported honestly whatever the numbers.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.constants import TORGO_CONTROL_SPEAKERS, TORGO_SEVERITY  # noqa: E402
from src.seed_utils import set_global_seed  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.torgo_ema import (  # noqa: E402
    SPARC_ARTIC_COLS,
    TONGUE_ARTICULATORS,
    downsample_to_50fps,
    is_clean_for_tongue,
    pos_xyz,
    resolve_channels,
)


def spk_group(spk: str) -> str:
    if spk.startswith("FC"):
        return "FC"
    if spk.startswith("MC"):
        return "MC"
    if spk.startswith("F"):
        return "F"
    return "M"


def audio_wavs(spk: str, limit: int) -> list[str]:
    """Head-mic wavs for a speaker (higher SNR), deterministic."""
    grp = spk_group(spk)
    pats = sorted(w for w in glob.glob(str(ROOT / grp / "**" / "*.wav"), recursive=True)
                  if spk in w and "headMic" in w)
    return pats[:limit]


def pos_files(spk: str, limit: int) -> list[str]:
    grp = spk_group(spk)
    return sorted(glob.glob(str(ROOT / grp / spk / "**" / "pos" / "*.pos"), recursive=True))[:limit]


def tongue_range_speed(xy: np.ndarray) -> tuple[float, float]:
    """xy: (T, 2*K) stacked [art0_x, art0_y, art1_x, art1_y, ...] for K tongue
    articulators. range = mean per-articulator 2-D positional std; speed = mean
    frame-to-frame 2-D speed averaged over articulators."""
    T, C = xy.shape
    K = C // 2
    if T < 3:
        return float("nan"), float("nan")
    rngs, spds = [], []
    for k in range(K):
        p = xy[:, 2 * k:2 * k + 2]                      # (T,2)
        rngs.append(float(np.sqrt(p.var(0).sum())))     # radius of gyration
        v = np.linalg.norm(np.diff(p, axis=0), axis=1)  # per-frame speed
        spds.append(float(v.mean()))
    return float(np.mean(rngs)), float(np.mean(spds))


def sparc_tongue_xy(ema: np.ndarray) -> np.ndarray:
    """(T,12) SPARC ema -> (T,6) tongue [TDx,TDy,TBx,TBy,TTx,TTy]."""
    cols = []
    for art in TONGUE_ARTICULATORS:
        cx, cy = SPARC_ARTIC_COLS[art]
        cols += [cx, cy]
    return ema[:, cols]


def ema_tongue_xy(pos_path: str, cmap) -> np.ndarray:
    """Real EMA (T,12,3) -> (T,6) tongue using x (ant-post) & z (vertical)."""
    xyz = downsample_to_50fps(pos_xyz(pos_path).astype(np.float64))  # (T,12,3)
    cols = []
    for art in TONGUE_ARTICULATORS:
        s = cmap.tongue[art]
        cols.append(xyz[:, s, [0, 2]])                 # (T,2) x,z
    return np.concatenate(cols, axis=1)                # (T,6)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=40)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "results" / "veritas" / "m2_articulatory_kinematics.json"))
    args = ap.parse_args()
    set_global_seed(42)

    inv = SPARCInverter(device=args.device)
    per_spk: dict[str, dict] = {}

    for i, (spk, sev) in enumerate(TORGO_SEVERITY.items(), 1):
        wavs = audio_wavs(spk, args.limit_per_speaker)
        rr, ss = [], []
        for w in wavs:
            try:
                x, _ = librosa.load(w, sr=16000, mono=True)
                if len(x) < 1600:
                    continue
                ema = np.asarray(inv.encode(x)["ema"], dtype=np.float64)
                r, s = tongue_range_speed(sparc_tongue_xy(ema))
                if np.isfinite(r):
                    rr.append(r); ss.append(s)
            except Exception as e:  # noqa: BLE001
                print(f"skip {w}: {e}", flush=True)

        # EMA-derived features where the session is clean
        ema_r = ema_s = float("nan")
        pfiles = pos_files(spk, args.limit_per_speaker)
        if pfiles:
            try:
                pooled = np.concatenate([pos_xyz(f) for f in pfiles[:12]], 0)
                cmap = resolve_channels(pooled)
                ok, _reason = is_clean_for_tongue(pooled, cmap)
                if ok:
                    er, es = [], []
                    for f in pfiles:
                        try:
                            r, s = tongue_range_speed(ema_tongue_xy(f, cmap))
                            if np.isfinite(r):
                                er.append(r); es.append(s)
                        except Exception:  # noqa: BLE001
                            pass
                    if er:
                        ema_r, ema_s = float(np.mean(er)), float(np.mean(es))
            except Exception as e:  # noqa: BLE001
                print(f"  {spk}: EMA feature failed: {e}", flush=True)

        if rr:
            per_spk[spk] = {
                "severity": sev, "is_control": spk in TORGO_CONTROL_SPEAKERS,
                "n_utts": len(rr),
                "sparc_range": float(np.mean(rr)), "sparc_speed": float(np.mean(ss)),
                "ema_range": ema_r, "ema_speed": ema_s,
            }
        print(f"[{i}/{len(TORGO_SEVERITY)}] {spk} sev={sev} n={len(rr)} "
              f"sparc_range={per_spk.get(spk, {}).get('sparc_range', float('nan')):.4f} "
              f"sparc_speed={per_spk.get(spk, {}).get('sparc_speed', float('nan')):.4f} "
              f"ema_range={ema_r:.4f}", flush=True)

    # ---- correlations ----
    keys = list(per_spk)
    sev = np.array([per_spk[k]["severity"] for k in keys], float)

    def _rho(field):
        v = np.array([per_spk[k][field] for k in keys], float)
        m = np.isfinite(v)
        if m.sum() < 3 or len(np.unique(sev[m])) < 2:
            return (float("nan"), float("nan"), int(m.sum()))
        r, p = spearmanr(sev[m], v[m])
        return (float(r), float(p), int(m.sum()))

    def _ground(fs, fe):
        vs = np.array([per_spk[k][fs] for k in keys], float)
        ve = np.array([per_spk[k][fe] for k in keys], float)
        m = np.isfinite(vs) & np.isfinite(ve)
        if m.sum() < 3:
            return (float("nan"), float("nan"), int(m.sum()))
        r, p = spearmanr(vs[m], ve[m])
        return (float(r), float(p), int(m.sum()))

    out = {
        "description": "articulatory-kinematic severity features from SPARC inversion; "
                       "EMA-grounded + severity-correlated on real TORGO",
        "features": "tongue range (working space) & speed (velocity)",
        "severity_corr": {
            "sparc_range_vs_severity": _rho("sparc_range"),
            "sparc_speed_vs_severity": _rho("sparc_speed"),
        },
        "ema_grounding": {  # does SPARC-derived feature track REAL EMA feature across speakers?
            "range_sparc_vs_ema": _ground("sparc_range", "ema_range"),
            "speed_sparc_vs_ema": _ground("sparc_speed", "ema_speed"),
            "n_ema_usable": int(sum(np.isfinite([per_spk[k]["ema_range"] for k in keys]))),
        },
        "n_speakers": len(keys),
        "per_speaker": per_spk,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print("\n===== M2 =====")
    print(f"sparc_range vs severity : rho={out['severity_corr']['sparc_range_vs_severity']}")
    print(f"sparc_speed vs severity : rho={out['severity_corr']['sparc_speed_vs_severity']}")
    print(f"EMA-grounding range (sparc vs real): {out['ema_grounding']['range_sparc_vs_ema']} "
          f"(n_usable={out['ema_grounding']['n_ema_usable']})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
