"""H2: does GT-free faithfulness DROP with dysarthria severity? (real TORGO audio)

For each of the 15 TORGO speakers we compute mean cycle-consistency
faithfulness over a sample of their utterances, then correlate per-speaker mean
faithfulness with severity (0=control, 1=mild-moderate, 2=severe). H2 predicts
a NEGATIVE Spearman rho (faithfulness falls as severity rises). The result is
reported honestly whatever its sign — a null/positive is an informative finding
about the metric, not a failure to be massaged.

Faithfulness is the CENTERED cycle-consistency metric (FIX 1): features are
centered against the healthy manifold (mean of control-speaker WavLM frames)
before the cosine, which recovers dynamic range. The healthy reference is fit on
control head-mic wavs DISJOINT from each control's evaluation slice, so a
control's own eval frames never define its centroid.

Faithfulness is cycle-consistency ALONE — no input-OOD penalty is fused in
(FIX 2), which would have made the severity correlation true by construction.

P2 (shared-information confound): SPARC inverts from WavLM layer 9, so
cycle-consistency measured AT layer 9 could be inflated by shared information.
We therefore report faithfulness at layer 9 AND layer 15 (one extra layer is
essentially free — both come from a single forward pass).

Accounting (FIX 3): every speaker's attempted/used/skipped counts are recorded
and the run FAILS LOUD if any TORGO speaker produced 0 usable utterances (the
prior version could silently drop speakers).

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
from src.veritas.faithfulness import HealthyReference, frame_faithfulness  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.wavlm_features import WavLMFeatures  # noqa: E402

SPK_DIR = {  # speaker_id -> top-level group dir (TORGO layout on disk)
    **{s: ("FC" if s.startswith("FC") else "F") for s in TORGO_SEVERITY if s.startswith("F")},
    **{s: ("MC" if s.startswith("MC") else "M") for s in TORGO_SEVERITY if s.startswith("M")},
}


def speaker_wavs(spk: str) -> list[str]:
    """Deterministic ordered list of a speaker's wavs (head-mic preferred).

    Returns the FULL ordered list; callers slice it — eval uses ``[:limit]`` and
    the healthy-reference fit uses the disjoint slice ``[limit:limit+k]``.
    """
    grp = SPK_DIR[spk]
    pats = glob.glob(str(ROOT / grp / "**" / f"*{spk}*" / "**" / "*.wav"), recursive=True) or glob.glob(
        str(ROOT / grp / "**" / "*.wav"), recursive=True
    )
    pats = sorted(w for w in pats if spk in w)
    head = [w for w in pats if "headMic" in w]
    return head or pats


def fit_healthy_references(
    wavlm: WavLMFeatures, layers: list[int], eval_limit: int, mu_per_spk: int
) -> tuple[dict[int, HealthyReference], int]:
    """Fit one healthy-manifold centroid PER layer, pooled across all control
    speakers, on head-mic wavs DISJOINT from each control's eval slice."""
    frames: dict[int, list[np.ndarray]] = {L: [] for L in layers}
    n_wavs = 0
    for spk in sorted(TORGO_CONTROL_SPEAKERS):
        fit_wavs = speaker_wavs(spk)[eval_limit : eval_limit + mu_per_spk]
        for w in fit_wavs:
            try:
                x, _ = librosa.load(w, sr=16000)
                if len(x) < 1600:
                    continue
                feats = wavlm.extract_layers(x, sr=16000, layers=tuple(layers))
                for L in layers:
                    frames[L].append(feats[L])
                n_wavs += 1
            except Exception as e:  # noqa: BLE001
                print(f"skip (mu-fit) {w}: {e}", flush=True)
    refs = {L: HealthyReference().fit(np.concatenate(frames[L], 0)) for L in layers}
    return refs, n_wavs


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-speaker", type=int, default=40)
    ap.add_argument("--mu-per-speaker", type=int, default=8,
                    help="control head-mic wavs per speaker used to fit the healthy centroid")
    ap.add_argument("--layers", default="9,15",
                    help="comma-separated WavLM layers; the first is the primary/headline layer")
    ap.add_argument("--device", default=default_device, help="cpu or cuda:0")
    ap.add_argument(
        "--out",
        default=str(ROOT / "results" / "veritas" / "faithfulness_severity.json"),
    )
    args = ap.parse_args()
    set_global_seed(42)

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    primary = layers[0]

    inv = SPARCInverter(device=args.device)
    fx = WavLMFeatures(device=args.device)

    refs, n_fit_wavs = fit_healthy_references(
        fx, layers, args.limit_per_speaker, args.mu_per_speaker
    )
    print(f"healthy reference fit on {n_fit_wavs} control wavs, layers={layers}", flush=True)

    per_spk: dict[str, dict] = {}
    tot_attempted = tot_used = tot_short = tot_error = 0
    for i, (spk, sev) in enumerate(TORGO_SEVERITY.items(), 1):
        wavs = speaker_wavs(spk)[: args.limit_per_speaker]
        attempted = len(wavs)
        vals: dict[int, list[float]] = {L: [] for L in layers}
        n_short = n_error = 0
        for w in wavs:
            try:
                x, sr = librosa.load(w, sr=16000)
                if len(x) < 1600:  # skip <0.1 s clips
                    n_short += 1
                    continue
                xhat = inv.resynth(x)
                fx_x = fx.extract_layers(x, sr=16000, layers=tuple(layers))
                fx_h = fx.extract_layers(xhat, sr=16000, layers=tuple(layers))
                for L in layers:
                    vals[L].append(float(frame_faithfulness(fx_x[L], fx_h[L], refs[L]).mean()))
            except Exception as e:  # noqa: BLE001 — real audio: skip + record, don't crash run
                n_error += 1
                print(f"skip {w}: {e}", flush=True)
        used = len(vals[primary])
        skipped = n_short + n_error
        by_layer = {
            str(L): {
                "mean": float(np.mean(vals[L])) if used else float("nan"),
                "std": float(np.std(vals[L])) if used else float("nan"),
            }
            for L in layers
        }
        per_spk[spk] = {
            "severity": sev,
            "is_control": spk in TORGO_CONTROL_SPEAKERS,
            "attempted": attempted,
            "used": used,
            "skipped": skipped,
            "skipped_short": n_short,
            "skipped_error": n_error,
            "mean_faithfulness": by_layer[str(primary)]["mean"],
            "std_faithfulness": by_layer[str(primary)]["std"],
            "by_layer": by_layer,
        }
        tot_attempted += attempted
        tot_used += used
        tot_short += n_short
        tot_error += n_error
        print(
            f"[{i}/{len(TORGO_SEVERITY)}] {spk} sev={sev} "
            f"attempted={attempted} used={used} skipped={skipped} "
            f"mean_faith(L{primary})={per_spk[spk]['mean_faithfulness']:.4f}",
            flush=True,
        )

    # per-layer Spearman over speakers with >=1 usable utterance
    usable = [s for s in TORGO_SEVERITY if per_spk[s]["used"] > 0]
    sev_vec = [per_spk[s]["severity"] for s in usable]
    per_layer = {}
    for L in layers:
        fth = [per_spk[s]["by_layer"][str(L)]["mean"] for s in usable]
        rho, p = spearmanr(sev_vec, fth) if len(usable) > 2 else (float("nan"), float("nan"))
        per_layer[str(L)] = {"spearman_rho": float(rho), "p_value": float(p), "n": len(usable)}

    missing = [s for s in TORGO_SEVERITY if per_spk[s]["used"] == 0]
    total_skipped = tot_short + tot_error

    out = {
        "hypothesis": "faithfulness DROPS with severity (expect negative rho)",
        "metric": "centered cycle-consistency faithfulness (healthy-manifold centered), "
                  "mean per utterance, averaged per speaker",
        "centered": True,
        "healthy_reference": "mean of control-speaker WavLM frames; fit on head-mic wavs "
                             "disjoint from each control's eval slice",
        "input_ood_fused": False,
        "mic": "head-mic preferred (higher SNR); fallback any",
        "limit_per_speaker": args.limit_per_speaker,
        "mu_per_speaker": args.mu_per_speaker,
        "n_healthy_fit_wavs": n_fit_wavs,
        "layers": layers,
        "primary_layer": primary,
        "spearman_rho": per_layer[str(primary)]["spearman_rho"],
        "p_value": per_layer[str(primary)]["p_value"],
        "n_speakers": len(usable),
        "per_layer": per_layer,
        "accounting": {
            "total_attempted": tot_attempted,
            "total_used": tot_used,
            "total_skipped": total_skipped,
            "total_skipped_short": tot_short,
            "total_skipped_error": tot_error,
            "speakers_with_zero_utterances": missing,
            "all_speakers_have_utterances": len(missing) == 0,
        },
        "per_speaker": per_spk,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(
        f"rho(L{primary})={out['spearman_rho']:.4f} p={out['p_value']:.4f} "
        f"n={len(usable)} total_skipped={total_skipped} -> {args.out}",
        flush=True,
    )
    for L in layers:
        pl = per_layer[str(L)]
        print(f"  layer {L}: rho={pl['spearman_rho']:.4f} p={pl['p_value']:.4f}", flush=True)

    # FIX 3: fail loud if any TORGO speaker produced zero usable utterances
    if missing:
        raise RuntimeError(
            f"FAIL-LOUD: {len(missing)} TORGO speaker(s) produced 0 usable utterances: "
            f"{missing} (JSON with accounting written to {args.out})"
        )


if __name__ == "__main__":
    main()
