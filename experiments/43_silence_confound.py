"""Silence-confound test for the headline ATD metric (Track B make-or-break).

QUESTION
--------
The headline ATD = speaker-level Spearman rho = 0.756 (layer 9) between mean ATD
and severity. ATD = length-normalised DTW distance between a speaker's WavLM-Large
(layer 9, PCA-64) trajectory and a per-word control reference. A phoneme
decomposition showed ~70-76% of the DTW path cost falls in SILENCE / unattributed
regions (leading/trailing silence + MFA sp/sil), NOT in phones. Does the severity
signal survive when silence is removed?

WHAT THIS SCRIPT DOES
---------------------
Re-runs the *identical* headline pipeline (WavLM layer 9, PCA-64, fastdtw radius=1,
euclidean, normalise by max(T_utt, T_ref)) on three frame selections:

  (A) FULL       trajectory  -> must reproduce ~0.756 (sanity check, read from the
                                published per-utterance CSV; also recomputed live for
                                a subset to confirm parity).
  (B) SPEECH-ONLY trajectory -> frames inside [onset of first non-silence phone,
                                offset of last non-silence phone] from the MFA
                                TextGrid (drop leading/trailing sil + sp/sil).
  (C) SILENCE-ONLY trajectory -> the complement: all frames OUTSIDE the speech span
                                (leading + trailing silence + any internal sil/sp gap).

For (B) and (C), per-word references are rebuilt from CONTROL speakers using the
*same* trimming, using the GLOBAL control set (directly comparable to the published
0.756). A LOCO (leave-one-control-out) speech-only variant is also computed.

Frame-rate assumption: WavLM-Large = 50 fps (20 ms hop). frame = round(time*fps);
span uses f0 = floor(onset*fps), f1 = ceil(offset*fps), clipped to [0, T].

Outputs
-------
results/silence_confound.json
results/speaker_atd_speech_only.csv

Everything REAL: reuses cached embeddings + on-disk MFA TextGrids. No WavLM
re-extraction, no GPU, seed = 42. Does NOT modify paper/, src/ logic, or existing
result files.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.constants import (  # noqa: E402
    LAYER_POSITION_MAP,
    RANDOM_SEED,
    TORGO_CONTROL_SPEAKERS,
    TORGO_SEVERITY,
)
from src.phoneme.aligner import parse_textgrid  # noqa: E402

# ----------------------------------------------------------------------------
# Configuration — identical to the published headline pipeline.
# ----------------------------------------------------------------------------
LAYER = 9
LAYER_POS = LAYER_POSITION_MAP[LAYER]          # 3
MODEL = "wavlm_large"
FPS = 50                                       # WavLM-Large frame rate (20 ms hop)
PCA_N = 64
DTW_RADIUS = 1                                 # headline compute_single_atd uses
                                               # fastdtw(...) default radius == 1
N_BOOT = 10000
MIN_CONTROL_UTTS = 2                           # same as reference builder
MIN_SPEECH_FRAMES = 2                          # need >=2 frames to PCA/DTW
np.random.seed(RANDOM_SEED)

TEXTGRID_DIR = ROOT / "data" / "processed" / "textgrids"
EMB_DIR = ROOT / "data" / "embeddings" / MODEL
PUBLISHED_UTT_CSV = ROOT / "results" / "atd_scores_layer9.csv"


# ----------------------------------------------------------------------------
# Frame-span helpers
# ----------------------------------------------------------------------------
def speech_span_frames(textgrid_path: Path, T: int, fps: int = FPS):
    """Return (f0, f1) frame indices of the SPEECH span = [onset of first
    non-silence phone, offset of last non-silence phone]. None if no phones."""
    phones = parse_textgrid(textgrid_path)     # already drops sil/sp/spn/empty
    if not phones:
        return None
    onset = min(p[1] for p in phones)
    offset = max(p[2] for p in phones)
    f0 = int(np.floor(onset * fps))
    f1 = int(np.ceil(offset * fps))
    f0 = max(0, min(f0, T))
    f1 = max(0, min(f1, T))
    if f1 <= f0:
        return None
    return f0, f1


def load_layer(utt_id: str, speaker: str) -> np.ndarray | None:
    p = EMB_DIR / speaker / f"{utt_id}.npy"
    if not p.exists():
        return None
    return np.load(str(p))[LAYER_POS]          # (T, 1024)


def slice_traj(utt_id: str, speaker: str, mode: str):
    """Return the requested frame selection (T_sel, 1024) + bookkeeping.

    mode: 'full' | 'speech' | 'silence'.
    Returns (frames, n_total, n_speech) or None if unavailable / too short.
    """
    full = load_layer(utt_id, speaker)
    if full is None:
        return None
    T = full.shape[0]
    if mode == "full":
        if T < MIN_SPEECH_FRAMES:
            return None
        return full, T, None

    tg = TEXTGRID_DIR / speaker / f"{utt_id}.TextGrid"
    if not tg.exists():
        return None
    span = speech_span_frames(tg, T)
    if span is None:
        return None
    f0, f1 = span
    n_speech = f1 - f0
    if mode == "speech":
        if n_speech < MIN_SPEECH_FRAMES:
            return None
        return full[f0:f1], T, n_speech
    if mode == "silence":
        sil = np.concatenate([full[:f0], full[f1:]], axis=0)  # leading+trailing+internal-edge
        if sil.shape[0] < MIN_SPEECH_FRAMES:
            return None
        return sil, T, n_speech
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# Reference construction on a chosen frame selection
# (mirrors src.reference_builder.build_word_reference: concat-fit PCA-64,
#  median-length template, DTW-warp others to template, average).
# ----------------------------------------------------------------------------
def _warp_to_template(traj, path, tlen):
    D = traj.shape[1]
    warped = np.zeros((tlen, D))
    counts = np.zeros(tlen)
    for ti, si in path:
        if ti < tlen and si < len(traj):
            warped[ti] += traj[si]
            counts[ti] += 1
    counts = np.maximum(counts, 1)
    warped /= counts[:, None]
    return warped


def build_reference(trajectories: list[np.ndarray]):
    """Build a reference dict {reference, pca_model} from raw (T,1024) trajectories.

    Identical algorithm to build_word_reference but operating on whatever frame
    selection the caller passes in (full / speech / silence). Returns None if <2
    usable trajectories.
    """
    trajs = [t for t in trajectories if t is not None and t.shape[0] >= 1]
    if len(trajs) < MIN_CONTROL_UTTS:
        return None
    all_frames = np.concatenate(trajs, axis=0)
    n_comp = min(PCA_N, all_frames.shape[0], all_frames.shape[1])
    if n_comp < 1:
        return None
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED)
    pca.fit(all_frames)
    reduced = [pca.transform(t) for t in trajs]
    lengths = [len(t) for t in reduced]
    median_len = int(np.median(lengths))
    ci = min(range(len(lengths)), key=lambda i: abs(lengths[i] - median_len))
    template = reduced[ci]
    aligned = [template]
    for i, tr in enumerate(reduced):
        if i == ci:
            continue
        _, path = fastdtw(template, tr, radius=DTW_RADIUS, dist=euclidean)
        aligned.append(_warp_to_template(tr, path, len(template)))
    reference = np.mean(aligned, axis=0)
    return {"reference": reference, "pca_model": pca, "n_comp": n_comp,
            "n_utts": len(trajs)}


def atd_single(query_frames: np.ndarray, ref: dict) -> float | None:
    """Length-normalised DTW distance, identical to compute_single_atd."""
    try:
        q = ref["pca_model"].transform(query_frames)
        dist, _ = fastdtw(q, ref["reference"], radius=DTW_RADIUS, dist=euclidean)
        return float(dist) / max(len(q), len(ref["reference"]))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Bootstrap CI for a speaker-level Spearman rho (resample the 15 speakers).
# ----------------------------------------------------------------------------
def bootstrap_rho_ci(sev: np.ndarray, val: np.ndarray, n_boot=N_BOOT, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)
    n = len(sev)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s, v = sev[idx], val[idx]
        if len(np.unique(s)) < 2 or len(np.unique(v)) < 2:
            continue
        r, _ = spearmanr(s, v)
        if not np.isnan(r):
            rhos.append(r)
    rhos = np.array(rhos)
    if len(rhos) == 0:
        return (float("nan"), float("nan"), 0)
    return (float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5)), len(rhos))


# ----------------------------------------------------------------------------
# Main driver for one mode.
# ----------------------------------------------------------------------------
def run_mode(mode: str, meta: pd.DataFrame, candidate_words, loco: bool = False):
    """Compute per-utterance ATD for `mode`, aggregate to speaker, return summary.

    mode: 'speech' | 'silence'. For full ATD we read the published CSV instead.
    loco=True builds per-held-out-speaker references (leave-one-control-out).
    """
    t0 = time.time()
    is_control = meta["speaker_id"].isin(TORGO_CONTROL_SPEAKERS)

    # Pre-slice every utterance's trajectory once (cache).
    slices: dict[str, tuple] = {}
    for _, row in meta.iterrows():
        sl = slice_traj(row["utterance_id"], row["speaker_id"], mode)
        if sl is not None:
            slices[row["utterance_id"]] = sl

    # Build per-word control reference cache (GLOBAL: all controls).
    ctrl_meta = meta[is_control]
    word_to_ctrl_utts: dict[str, list[tuple[str, str]]] = {}
    for _, row in ctrl_meta.iterrows():
        word_to_ctrl_utts.setdefault(row["word"], []).append(
            (row["utterance_id"], row["speaker_id"]))

    def build_word_ref(word, exclude_speaker=None):
        items = word_to_ctrl_utts.get(word, [])
        trajs = []
        used = set()
        for utt_id, spk in items:
            if exclude_speaker is not None and spk == exclude_speaker:
                continue
            if utt_id in slices:
                trajs.append(slices[utt_id][0])
                used.add(spk)
        if len(trajs) < MIN_CONTROL_UTTS:
            return None
        return build_reference(trajs)

    global_ref_cache: dict[str, dict | None] = {}
    if not loco:
        for word in candidate_words:
            global_ref_cache[word] = build_word_ref(word)

    rows = []
    n_used = 0
    n_skipped = 0
    for _, row in meta.iterrows():
        utt_id = row["utterance_id"]
        spk = row["speaker_id"]
        word = row["word"]
        if utt_id not in slices:
            n_skipped += 1
            continue
        if loco:
            ref = build_word_ref(word, exclude_speaker=spk)  # leave-one-control-out
        else:
            ref = global_ref_cache.get(word)
        if ref is None:
            n_skipped += 1
            continue
        atd = atd_single(slices[utt_id][0], ref)
        if atd is None:
            n_skipped += 1
            continue
        rows.append({
            "utterance_id": utt_id, "speaker_id": spk, "word": word,
            "severity_score": int(TORGO_SEVERITY.get(spk, -1)),
            "atd_score": atd,
            "n_frames_total": slices[utt_id][1],
            "n_frames_speech": slices[utt_id][2],
        })
        n_used += 1

    udf = pd.DataFrame(rows)
    spk_df = (udf.groupby("speaker_id")
              .agg(mean_atd=("atd_score", "mean"),
                   std_atd=("atd_score", "std"),
                   median_atd=("atd_score", "median"),
                   n_utterances=("atd_score", "count"),
                   severity_score=("severity_score", "first"))
              .reset_index())
    spk_df["is_control"] = spk_df["severity_score"] == 0
    rho, p = spearmanr(spk_df["severity_score"], spk_df["mean_atd"])
    lo, hi, nb = bootstrap_rho_ci(spk_df["severity_score"].values,
                                  spk_df["mean_atd"].values)
    elapsed = time.time() - t0
    print(f"[{mode}{' LOCO' if loco else ''}] rho={rho:.4f} p={p:.5f} "
          f"CI=[{lo:.3f},{hi:.3f}] n_utts={n_used} n_skip={n_skipped} "
          f"n_spk={len(spk_df)} ({elapsed:.0f}s)")
    return {
        "rho": float(rho), "p": float(p),
        "ci95": [lo, hi], "n_boot_valid": nb,
        "n_utts": int(n_used), "n_skipped": int(n_skipped),
        "n_speakers": int(len(spk_df)),
    }, udf, spk_df


def full_atd_from_published(meta: pd.DataFrame):
    """Read the published per-utterance ATD and recompute speaker rho (sanity)."""
    df = pd.read_csv(PUBLISHED_UTT_CSV).dropna(subset=["atd_score"])
    spk = (df.groupby("speaker_id")
           .agg(mean_atd=("atd_score", "mean"),
                n_utterances=("atd_score", "count"),
                severity_score=("severity_score", "first")).reset_index())
    rho, p = spearmanr(spk["severity_score"], spk["mean_atd"])
    lo, hi, nb = bootstrap_rho_ci(spk["severity_score"].values, spk["mean_atd"].values)
    print(f"[full/published] rho={rho:.4f} p={p:.5f} CI=[{lo:.3f},{hi:.3f}] "
          f"n_utts={len(df)} n_spk={len(spk)}")
    return {"rho": float(rho), "p": float(p), "ci95": [lo, hi],
            "n_boot_valid": nb, "n_utts": int(len(df)), "n_skipped": 0,
            "n_speakers": int(len(spk)), "source": str(PUBLISHED_UTT_CSV.name)}, spk


def main():
    meta = pd.read_csv(ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv")
    meta = meta[meta["dataset"] == "torgo"].copy()
    meta["speaker_id"] = meta["speaker_id"].astype(str)
    meta["word"] = meta["word"].astype(str)

    # candidate words = words with >= MIN_CONTROL_UTTS control utterances
    ctrl = meta[meta["speaker_id"].isin(TORGO_CONTROL_SPEAKERS)]
    wc = ctrl.groupby("word").size()
    candidate_words = wc[wc >= MIN_CONTROL_UTTS].index.tolist()
    print(f"TORGO utts={len(meta)}  candidate words={len(candidate_words)}")

    # (A) FULL — published headline (sanity reproduction)
    full_summary, full_spk = full_atd_from_published(meta)

    # (B) SPEECH-ONLY — global control references
    speech_summary, speech_udf, speech_spk = run_mode("speech", meta, candidate_words)

    # (B') SPEECH-ONLY LOCO (cheap-ish; leave-one-control-out)
    speech_loco_summary, _, _ = run_mode("speech", meta, candidate_words, loco=True)

    # (C) SILENCE-ONLY — global control references on the complement frames
    silence_summary, _, silence_spk = run_mode("silence", meta, candidate_words)

    out = {
        "description": "Silence-confound test: does the ATD severity signal survive "
                       "removing leading/trailing + sp/sil silence?",
        "frame_rate_fps": FPS,
        "frame_rate_assumption": "WavLM-Large 50 fps (20 ms hop); "
                                 "f0=floor(onset*fps), f1=ceil(offset*fps), clip[0,T]",
        "layer": LAYER, "model": MODEL, "pca_n": PCA_N, "dtw_radius": DTW_RADIUS,
        "n_bootstrap": N_BOOT, "seed": RANDOM_SEED,
        "speech_span": "[onset of first non-silence MFA phone, offset of last non-silence phone]",
        "full_atd_rho": full_summary,
        "speech_only_rho": speech_summary,
        "speech_only_loco_rho": speech_loco_summary,
        "silence_only_rho": silence_summary,
        "headline_published_rho": 0.7560864148142503,
    }
    (ROOT / "results" / "silence_confound.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")

    # speaker-level speech-only CSV (+ merge full & silence for side-by-side)
    merged = speech_spk.rename(columns={
        "mean_atd": "mean_atd_speech_only", "std_atd": "std_atd_speech_only",
        "median_atd": "median_atd_speech_only", "n_utterances": "n_utts_speech_only"})
    merged = merged.merge(
        full_spk[["speaker_id", "mean_atd"]].rename(columns={"mean_atd": "mean_atd_full"}),
        on="speaker_id", how="left")
    merged = merged.merge(
        silence_spk[["speaker_id", "mean_atd"]].rename(
            columns={"mean_atd": "mean_atd_silence_only"}),
        on="speaker_id", how="left")
    cols = ["speaker_id", "severity_score", "is_control", "mean_atd_full",
            "mean_atd_speech_only", "mean_atd_silence_only", "std_atd_speech_only",
            "median_atd_speech_only", "n_utts_speech_only"]
    merged[cols].sort_values("severity_score").to_csv(
        ROOT / "results" / "speaker_atd_speech_only.csv", index=False)

    print("\n=== SUMMARY ===")
    print(f"full-ATD rho     : {full_summary['rho']:.4f}  (published 0.7561)")
    print(f"speech-only rho  : {speech_summary['rho']:.4f}  CI{speech_summary['ci95']}")
    print(f"speech-only LOCO : {speech_loco_summary['rho']:.4f}")
    print(f"silence-only rho : {silence_summary['rho']:.4f}  CI{silence_summary['ci95']}")
    print("Wrote results/silence_confound.json + results/speaker_atd_speech_only.csv")


if __name__ == "__main__":
    main()
