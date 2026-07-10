"""P2 (HONEST): does articulatory-bottleneck clarification make dysarthric speech
more TYPICAL of healthy speech? Objective, offline, real TORGO audio.

WHAT WE CAN AND CANNOT MEASURE (read this).
    The honest intelligibility measure is ASR word/character error rate against the
    TORGO prompt transcripts. This machine has NO ASR model in its offline HF cache
    (only WavLM-Large + SPARC + text-embedding models — checked at build time), and
    the network is unavailable, so WER/CER CANNOT be computed here. We therefore use
    a cached-model PROXY and label it as such:

      PRIMARY PROXY = healthy-manifold TYPICALITY. Fit a WavLM-Large layer-9
      diagonal-Mahalanobis out-of-distribution (OOD) model on healthy TORGO CONTROL
      frames (src.veritas.faithfulness.OODScorer). Lower OOD distance = closer to the
      healthy speech manifold = more typical. We report mean OOD for ORIGINAL vs a
      pure-resynthesis control (strength 0) vs CLARIFIED (strength>0), per severity.

    Typicality is NOT intelligibility. Moving toward the healthy manifold is what the
    clarifier is DESIGNED to do, so this proxy is partially ALIGNED with the method's
    objective — treat a positive result as "the resynthesis is acoustically more
    healthy-typical", not as proven intelligibility gain. Two honesty controls guard
    the claim:
      (1) RESYNTH control (strength 0, no projection) isolates the SPARC/HiFi-GAN
          bottleneck's own effect from the projection's effect; the projection must
          beat plain resynthesis to matter.
      (2) CONTENT preservation (per-frame WavLM cosine, original vs synthesized)
          guards against trivially lowering OOD by morphing everything toward a
          generic healthy centroid — that would collapse content cosine.
    A robustness OOD is also reported at a second WavLM layer (12).

    The result is reported honestly whatever it is (HELPS / NEUTRAL / HURTS), per
    severity, with numbers -> results/veritas/p2_clarify_eval.json.

Speaker-disjoint by construction: the healthy manifold (EMA stats + OOD) is fit on
the 7 TORGO CONTROL speakers; evaluation is on the 8 DYSARTHRIC speakers.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
from pathlib import Path

# Set BEFORE torch initialises CUDA: expandable segments avoids fragmentation OOM
# on smaller cards (this box is a 12 GB RTX 4080), and the cuBLAS workspace config
# both bounds deterministic-cuBLAS memory and makes it truly reproducible.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import src.veritas  # noqa: E402,F401  (offline bootstrap FIRST)
from src.seed_utils import set_global_seed  # noqa: E402
from src.veritas.clarify import ArticulatoryClarifier, HealthyEMAStats  # noqa: E402
from src.veritas.corpora.torgo import load as load_torgo  # noqa: E402
from src.veritas.faithfulness import OODScorer  # noqa: E402
from src.veritas.inversion import SPARCInverter  # noqa: E402
from src.veritas.wavlm_features import WavLMFeatures  # noqa: E402

SEV_NAME = {1.0: "mild_moderate", 2.0: "severe"}
PRIMARY_LAYER = 9
ROBUST_LAYER = 12
STATS_CACHE = ROOT / "data" / "processed" / "veritas" / "healthy_ema_stats.npz"
OOD_CACHE = ROOT / "data" / "processed" / "veritas" / "healthy_ood_stats.npz"


def _read16k(path: str, min_s: float, max_s: float) -> np.ndarray | None:
    try:
        w, sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:  # noqa: BLE001
        return None
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        import librosa

        w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    w = np.nan_to_num(np.asarray(w, dtype=np.float32))
    dur = len(w) / 16000.0
    if dur < min_s or dur > max_s:
        return None
    peak = float(np.max(np.abs(w)))
    if peak < 1e-4:
        return None
    return np.ascontiguousarray(w / peak * 0.95, dtype=np.float32)


def _sample(records, n: int, seed: int):
    rng = np.random.default_rng(seed)
    recs = sorted(records, key=lambda r: r.utterance_id)  # deterministic base order
    idx = rng.permutation(len(recs))[:n]
    return [recs[i] for i in sorted(idx)]


def _cos_mean(a: np.ndarray, b: np.ndarray) -> float:
    T = min(len(a), len(b))
    if T == 0:
        return float("nan")
    a, b = a[:T], b[:T]
    num = (a * b).sum(1)
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    return float(np.mean(num / den))


def build_healthy_models(inv, wavlm12, wavlm9, args):
    """Fit (or load) healthy EMA stats + per-layer OOD scorers on control speakers."""
    ctrl = [r for r in load_torgo() if r.is_control]
    by_spk: dict[str, list] = {}
    for r in ctrl:
        by_spk.setdefault(r.speaker_id, []).append(r)

    if STATS_CACHE.exists() and OOD_CACHE.exists() and not args.refit:
        stats = HealthyEMAStats.load(str(STATS_CACHE))
        d = np.load(OOD_CACHE)
        ood9 = OODScorer(); ood9.mu, ood9.var = d["mu9"], d["var9"]
        ood12 = OODScorer(); ood12.mu, ood12.var = d["mu12"], d["var12"]
        print(f"[fit] loaded cached healthy stats ({STATS_CACHE.name}, {OOD_CACHE.name})",
              flush=True)
        return stats, ood9, ood12, {"cached": True, "n_control_speakers": len(by_spk)}

    ema_frames, f9, f12 = [], [], []
    n_used = 0
    for spk in sorted(by_spk):
        for r in _sample(by_spk[spk], args.fit_utts_per_speaker, args.seed):
            w = _read16k(r.wav_path, args.min_seconds, args.max_seconds)
            if w is None:
                continue
            try:
                code = inv.encode(w)
                layers = wavlm9.extract_layers(w, sr=16000, layers=(PRIMARY_LAYER, ROBUST_LAYER))
            except Exception as e:  # noqa: BLE001
                print(f"  [fit] {spk} encode failed: {e}", flush=True)
                continue
            ema_frames.append(np.asarray(code["ema"], dtype=np.float32))
            f9.append(layers[PRIMARY_LAYER]); f12.append(layers[ROBUST_LAYER])
            n_used += 1
    if n_used < 10:
        raise RuntimeError(f"too few control utterances fit ({n_used}); cannot build manifold")

    stats = HealthyEMAStats.fit(np.concatenate(ema_frames, 0))
    ood9 = OODScorer().fit(np.concatenate(f9, 0))
    ood12 = OODScorer().fit(np.concatenate(f12, 0))
    STATS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    stats.save(str(STATS_CACHE))
    np.savez(OOD_CACHE, mu9=ood9.mu, var9=ood9.var, mu12=ood12.mu, var12=ood12.var)
    print(f"[fit] healthy manifold from {n_used} control utts over {len(by_spk)} speakers "
          f"-> {STATS_CACHE.name}, {OOD_CACHE.name}", flush=True)
    return stats, ood9, ood12, {"cached": False, "n_control_speakers": len(by_spk),
                                "n_control_utts": n_used}


def eval_utt(clar, wavlm, ood9, ood12, wav, strengths):
    """Return per-utterance metrics for original / resynth / clarified@strengths."""
    code = clar.encode(wav)
    # original acoustics
    l0 = wavlm.extract_layers(wav, sr=16000, layers=(PRIMARY_LAYER, ROBUST_LAYER))
    fx9 = l0[PRIMARY_LAYER]
    out = {"orig": {"ood9": float(np.mean(ood9.score(fx9))),
                    "ood12": float(np.mean(ood12.score(l0[ROBUST_LAYER])))}}
    # conditions: strength 0 (pure resynth) + each requested strength
    for s in [0.0] + list(strengths):
        code_s = clar.clarify_code(code, s)
        w = np.asarray(clar.inverter.decode(code_s), dtype=np.float32).squeeze()
        ls = wavlm.extract_layers(w, sr=16000, layers=(PRIMARY_LAYER, ROBUST_LAYER))
        key = "resynth" if s == 0.0 else f"clar{s:g}"
        out[key] = {
            "ood9": float(np.mean(ood9.score(ls[PRIMARY_LAYER]))),
            "ood12": float(np.mean(ood12.score(ls[ROBUST_LAYER]))),
            "content_cos9": _cos_mean(fx9, ls[PRIMARY_LAYER]),  # vs ORIGINAL
        }
    return out


def _summ(vals: list[float]) -> dict:
    a = np.asarray([v for v in vals if np.isfinite(v)], dtype=np.float64)
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "n": int(len(a))}


def aggregate(rows: list[dict], strengths, cond_keys) -> dict:
    """Per-severity + overall condition summaries + honest verdicts."""
    groups: dict[str, list] = {"overall": rows}
    for sev, name in SEV_NAME.items():
        groups[name] = [r for r in rows if r["severity"] == sev]

    top_strength = f"clar{max(strengths):g}"
    report: dict = {}
    for gname, grp in groups.items():
        if not grp:
            continue
        block: dict = {"n_utts": len(grp),
                       "speakers": sorted({r["speaker"] for r in grp})}
        for ck in cond_keys:
            block[ck] = {
                "ood9": _summ([r["cond"][ck]["ood9"] for r in grp]),
                "ood12": _summ([r["cond"][ck]["ood12"] for r in grp]),
            }
            if "content_cos9" in grp[0]["cond"][ck]:
                block[ck]["content_cos9"] = _summ(
                    [r["cond"][ck]["content_cos9"] for r in grp])
        # paired deltas (positive = more typical than the reference)
        o = np.array([r["cond"]["orig"]["ood9"] for r in grp])
        rs = np.array([r["cond"]["resynth"]["ood9"] for r in grp])
        cl = np.array([r["cond"][top_strength]["ood9"] for r in grp])
        block["delta_ood9_clar_vs_orig"] = _summ((o - cl).tolist())
        block["delta_ood9_clar_vs_resynth"] = _summ((rs - cl).tolist())
        block["frac_clar_more_typical_than_orig"] = float(np.mean(cl < o))
        block["frac_clar_more_typical_than_resynth"] = float(np.mean(cl < rs))
        block["verdict"] = _verdict(
            proj_delta=float(np.mean(rs - cl)),
            proj_delta_std=float(np.std(rs - cl)),
            orig_delta=float(np.mean(o - cl)),
            content=float(np.mean([r["cond"][top_strength]["content_cos9"] for r in grp])),
            content_resynth=float(np.mean([r["cond"]["resynth"]["content_cos9"] for r in grp])),
            top_strength=max(strengths),
        )
        report[gname] = block
    return report


def _verdict(proj_delta, proj_delta_std, orig_delta, content, content_resynth,
             top_strength) -> str:
    """Honest verdict on the TYPICALITY proxy at the strongest clarification.

    ``proj_delta`` = mean(OOD_resynth) - mean(OOD_clar): the projection's effect
    ABOVE plain resynthesis (positive = more typical). Guarded by content cosine.
    ``rel`` is a Cohen's-d-style effect size (mean delta / its std).
    """
    rel = proj_delta / (abs(proj_delta_std) + 1e-9)
    content_drop = content_resynth - content  # extra content lost by projecting
    tag = (f"[strength={top_strength:g}] projection-vs-resynth OOD delta="
           f"{proj_delta:+.3f} (std {proj_delta_std:.3f}); clarified content cosine="
           f"{content:.3f} (resynth {content_resynth:.3f})")
    if not np.isfinite(proj_delta):
        return "NO DATA. " + tag
    if content < 0.5:
        return ("HURTS / INVALID: clarified audio no longer preserves input content "
                f"(cosine {content:.3f} < 0.5) -- typicality change is not meaningful. " + tag)
    if rel > 0.15 and proj_delta > 0:
        return ("PROXY: HELPS typicality -- projection moves articulation toward the "
                "healthy manifold beyond plain resynthesis (NOT proven intelligibility; "
                "typicality proxy, no offline ASR). " + tag)
    if rel < -0.15 and proj_delta < 0:
        return ("PROXY: HURTS typicality -- projection moves AWAY from the healthy "
                "manifold relative to plain resynthesis. " + tag)
    return ("PROXY: NEUTRAL -- projection does not change healthy-manifold typicality "
            "beyond plain SPARC resynthesis. " + tag)


def main() -> None:
    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit-per-speaker", type=int, default=25,
                    help="dysarthric utterances evaluated per speaker")
    ap.add_argument("--fit-utts-per-speaker", type=int, default=40,
                    help="control utterances per speaker for the healthy manifold")
    ap.add_argument("--strengths", default="0.5,1.0",
                    help="clarification strengths to evaluate (comma list)")
    ap.add_argument("--mode", default="per_tract", choices=["per_tract", "coral"])
    ap.add_argument("--min-seconds", type=float, default=0.4)
    ap.add_argument("--max-seconds", type=float, default=8.0)
    ap.add_argument("--device", default=default_device)
    ap.add_argument("--refit", action="store_true", help="ignore cached healthy stats")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "results" / "veritas" / "p2_clarify_eval.json"))
    args = ap.parse_args()
    set_global_seed(args.seed)
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]

    t0 = time.perf_counter()
    print(f"[P2] device={args.device} strengths={strengths} mode={args.mode}", flush=True)
    inv = SPARCInverter(device=args.device)
    wavlm = WavLMFeatures(layer=PRIMARY_LAYER, device=args.device)

    stats, ood9, ood12, fit_info = build_healthy_models(inv, wavlm, wavlm, args)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # release any fit-phase fragmentation before eval
    clar = ArticulatoryClarifier(inv, healthy_stats=stats, mode=args.mode, seed=args.seed)

    # dysarthric eval set (speaker-disjoint from the control fit)
    dys = [r for r in load_torgo() if not r.is_control]
    by_spk: dict[str, list] = {}
    for r in dys:
        by_spk.setdefault(r.speaker_id, []).append(r)

    cond_keys = ["orig", "resynth"] + [f"clar{s:g}" for s in strengths]
    rows: list[dict] = []
    for spk in sorted(by_spk):
        sev = by_spk[spk][0].severity_rank
        picked = _sample(by_spk[spk], args.limit_per_speaker, args.seed)
        n_ok = 0
        for r in picked:
            w = _read16k(r.wav_path, args.min_seconds, args.max_seconds)
            if w is None:
                continue
            try:
                cond = eval_utt(clar, wavlm, ood9, ood12, w, strengths)
            except Exception as e:  # noqa: BLE001
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # recover from a transient OOM
                print(f"  [eval] {spk}/{r.utterance_id} failed: {e}", flush=True)
                continue
            rows.append({"speaker": spk, "severity": float(sev),
                         "utt": r.utterance_id, "cond": cond})
            n_ok += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[{spk}] sev={sev:g} evaluated {n_ok} utts", flush=True)

    report = aggregate(rows, strengths, cond_keys)
    out = {
        "title": "P2 articulatory-bottleneck clarifier -- honest typicality eval (TORGO)",
        "proxy": "healthy-manifold typicality: WavLM-Large L9 diagonal-Mahalanobis OOD "
                 "distance to TORGO control frames (lower = more typical). Robustness "
                 "layer 12. Content preservation = per-frame WavLM-L9 cosine vs original.",
        "proxy_is_not_intelligibility": True,
        "no_asr_reason": "no ASR model in the offline HF cache (only WavLM-Large + SPARC "
                         "+ text embedders) and network unavailable; WER/CER not computable. "
                         "TORGO prompt transcripts exist (metadata 'word' column) but cannot "
                         "be scored without an ASR.",
        "circularity_disclosure": "the clarifier projects toward the healthy manifold, so "
                                  "this typicality proxy is partially aligned with its objective; "
                                  "the strength-0 RESYNTH control and content-cosine guard are "
                                  "reported so the projection's marginal effect is isolated.",
        "manifold_fit": fit_info,
        "eval": {"n_dysarthric_speakers": len(by_spk), "n_utts": len(rows),
                 "strengths": strengths, "mode": args.mode,
                 "primary_layer": PRIMARY_LAYER, "robust_layer": ROBUST_LAYER,
                 "limit_per_speaker": args.limit_per_speaker,
                 "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
                 "seed": args.seed, "device": args.device},
        "results": report,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))

    print("\n================= P2 RESULT (HONEST typicality proxy) =================")
    print("PROXY = healthy-manifold typicality (WavLM-L9 OOD). NOT intelligibility/WER "
          "(no offline ASR).")
    for gname in ["overall", "mild_moderate", "severe"]:
        b = report.get(gname)
        if not b:
            continue
        top = f"clar{max(strengths):g}"
        print(f"\n[{gname}] n={b['n_utts']} speakers={b['speakers']}")
        print(f"  OOD9  original={b['orig']['ood9']['mean']:.3f}  "
              f"resynth={b['resynth']['ood9']['mean']:.3f}  "
              f"{top}={b[top]['ood9']['mean']:.3f}")
        print(f"  content cos9 (vs orig): resynth={b['resynth']['content_cos9']['mean']:.3f}  "
              f"{top}={b[top]['content_cos9']['mean']:.3f}")
        print(f"  delta OOD9 clar-vs-resynth={b['delta_ood9_clar_vs_resynth']['mean']:+.3f}  "
              f"clar-vs-orig={b['delta_ood9_clar_vs_orig']['mean']:+.3f}")
        print(f"  VERDICT: {b['verdict']}")
    print(f"\n-> {args.out}  ({out['elapsed_s']}s)")


if __name__ == "__main__":
    main()
