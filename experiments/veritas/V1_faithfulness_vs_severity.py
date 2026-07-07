"""H2: does GT-free faithfulness DROP with dysarthria severity? (real TORGO audio)

For each of the 15 TORGO speakers we compute mean cycle-consistency
faithfulness over a sample of their utterances, then correlate per-speaker mean
faithfulness with severity (0=control, 1=mild-moderate, 2=severe). H2 predicts
a NEGATIVE Spearman rho (faithfulness falls as severity rises). The result is
reported honestly whatever its sign — a null/positive is an informative finding
about the metric, not a failure to be massaged.

Mic choice: prefer head-mic recordings (higher SNR) over array-mic when a
speaker has both; every TORGO speaker has head-mic audio, so the comparison is
mic-consistent across severity groups.
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
from src.veritas.faithfulness import cycle_faithfulness  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.wavlm_features import WavLMFeatures  # noqa: E402

SPK_DIR = {  # speaker_id -> top-level group dir (TORGO layout on disk)
    **{s: ("FC" if s.startswith("FC") else "F") for s in TORGO_SEVERITY if s.startswith("F")},
    **{s: ("MC" if s.startswith("MC") else "M") for s in TORGO_SEVERITY if s.startswith("M")},
}


def speaker_wavs(spk: str, limit: int) -> list[str]:
    """Deterministic list of up to ``limit`` wavs for a speaker.

    Prefer head-mic recordings (higher SNR); fall back to any mic if a speaker
    has no head-mic audio.
    """
    grp = SPK_DIR[spk]
    pats = glob.glob(str(ROOT / grp / "**" / f"*{spk}*" / "**" / "*.wav"), recursive=True) or glob.glob(
        str(ROOT / grp / "**" / "*.wav"), recursive=True
    )
    pats = sorted(w for w in pats if spk in w)
    head = [w for w in pats if "headMic" in w]
    chosen = head or pats
    return chosen[:limit]


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=40)
    ap.add_argument("--device", default=default_device, help="cpu or cuda:0")
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "veritas" / "faithfulness_severity.json"),
    )
    args = ap.parse_args()
    set_global_seed(42)

    inv, fx = SPARCInverter(device=args.device), WavLMFeatures(device=args.device)
    per_spk: dict[str, dict] = {}
    for i, (spk, sev) in enumerate(TORGO_SEVERITY.items(), 1):
        wavs = speaker_wavs(spk, args.limit_per_speaker)
        vals = []
        for w in wavs:
            try:
                x, sr = librosa.load(w, sr=16000)
                if len(x) < 1600:  # skip <0.1 s clips
                    continue
                vals.append(float(cycle_faithfulness(x, sr, inv, fx).mean()))
            except Exception as e:  # noqa: BLE001 — real audio: skip + record, don't crash run
                print(f"skip {w}: {e}")
        if vals:
            per_spk[spk] = {
                "severity": sev,
                "mean_faithfulness": float(np.mean(vals)),
                "std_faithfulness": float(np.std(vals)),
                "n": len(vals),
                "is_control": spk in TORGO_CONTROL_SPEAKERS,
            }
        print(
            f"[{i}/{len(TORGO_SEVERITY)}] {spk} sev={sev} "
            f"n={len(vals)} mean_faith="
            f"{per_spk.get(spk, {}).get('mean_faithfulness', float('nan')):.4f}",
            flush=True,
        )

    sev = [v["severity"] for v in per_spk.values()]
    fth = [v["mean_faithfulness"] for v in per_spk.values()]
    rho, p = spearmanr(sev, fth) if len(per_spk) > 2 else (float("nan"), float("nan"))
    out = {
        "hypothesis": "faithfulness DROPS with severity (expect negative rho)",
        "metric": "cycle_faithfulness mean per utterance, averaged per speaker",
        "mic": "head-mic preferred (higher SNR); fallback any",
        "limit_per_speaker": args.limit_per_speaker,
        "spearman_rho": float(rho),
        "p_value": float(p),
        "n_speakers": len(per_spk),
        "per_speaker": per_spk,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"rho={rho:.4f} p={p:.4f} n={len(per_spk)} -> {args.out}")


if __name__ == "__main__":
    main()
