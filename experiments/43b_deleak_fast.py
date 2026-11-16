"""
Experiment 43b — FAST de-leaked headline ATD correlation (radius=1 only).

Same isolation logic as 43_deleak_headline.py but radius=1 ONLY (the value the
published headline actually used: src/atd_metric.compute_single_atd calls
fastdtw without a radius kwarg => fastdtw default radius=1). radius=1 is the
directly-comparable setting to the published rho=0.756, and avoids the ~hours
runtime of radius=30.

For control utterances:
  * LEAKED    = scored vs the GLOBAL full-control reference + PCA basis
                (reproduces published behaviour; control's own speaker is in
                 the reference).
  * DE-LEAKED = scored vs the LEAVE-ONE-CONTROL-OUT (LOCO) reference + PCA
                basis built from the OTHER controls only.
Dysarthric utterances never enter any reference; scored vs GLOBAL ref in both
conditions, so only the control side differs.

Outputs:
  results/deleak_report_fast.json
  results/speaker_atd_deleaked_fast.csv
Seed = 42.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr

PROJECT_ROOT = Path(r"c:\Users\Dell\Desktop\Dysar")
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import (
    LAYER_POSITION_MAP,
    RANDOM_SEED,
    TORGO_CONTROL_SPEAKERS,
    TORGO_DYSARTHRIC_SPEAKERS,
    TORGO_SEVERITY,
)

LAYERS = [9, 1]
RADIUS = 1
N_BOOTSTRAP = 10000

META_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
GLOBAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"
LOCO_ROOT = PROJECT_ROOT / "data" / "processed" / "references_loco"
RESULTS_DIR = PROJECT_ROOT / "results"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_global_ref_cache(manifest: dict, layer: int) -> dict:
    cache = {}
    for key, info in manifest.get("word_layer_ref", {}).items():
        w, lyr = key.rsplit("_", 1)
        if int(lyr) != layer:
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as f:
                cache[w] = pickle.load(f)
    return cache


def load_loco_ref_cache(held_out: str, layer: int) -> dict:
    man_path = LOCO_ROOT / held_out / "reference_manifest.json"
    cache = {}
    with open(man_path, encoding="utf-8") as f:
        man = json.load(f)
    for key, info in man.get("word_layer_ref", {}).items():
        w, lyr = key.rsplit("_", 1)
        if int(lyr) != layer:
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as f:
                cache[w] = pickle.load(f)
    return cache


def atd_from_ref(utt_frames: np.ndarray, ref_data: dict) -> float:
    pca = ref_data["pca_model"]
    reference = ref_data["reference"]
    utt_reduced = pca.transform(utt_frames)
    distance, _ = fastdtw(utt_reduced, reference, radius=RADIUS, dist=euclidean)
    return float(distance / max(len(utt_reduced), len(reference)))


def cohens_d(lo: np.ndarray, hi: np.ndarray) -> float:
    n1, n2 = len(lo), len(hi)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = lo.var(ddof=1), hi.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    return 0.0 if pooled == 0 else float((hi.mean() - lo.mean()) / pooled)


def bootstrap_ci(sev: np.ndarray, vals: np.ndarray, seed: int = RANDOM_SEED) -> list:
    rng = np.random.default_rng(seed)
    n = len(sev)
    rhos = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        s, v = sev[idx], vals[idx]
        if np.all(s == s[0]) or np.all(v == v[0]):
            continue
        r, _ = spearmanr(s, v)
        if not np.isnan(r):
            rhos.append(r)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    t0 = time.time()
    log("loading metadata + ssl index + global manifest ...")
    meta = pd.read_csv(META_PATH)
    with open(INDEX_PATH) as f:
        ssl_index = json.load(f)
    with open(GLOBAL_MANIFEST) as f:
        global_manifest = json.load(f)

    meta = meta[meta["speaker_id"].isin(TORGO_SEVERITY.keys())].copy()
    meta["severity"] = meta["speaker_id"].map(TORGO_SEVERITY)
    meta["is_ctrl"] = meta["speaker_id"].isin(TORGO_CONTROL_SPEAKERS)
    log(f"TORGO utts: {len(meta)} (ctrl={meta['is_ctrl'].sum()}, dys={(~meta['is_ctrl']).sum()})")

    report = {"meta": {"radius": RADIUS, "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED,
                       "severity_map": TORGO_SEVERITY}, "by_layer": {}}

    for layer in LAYERS:
        layer_pos = LAYER_POSITION_MAP[layer]
        log(f"===== LAYER {layer} (pos {layer_pos}) =====")
        global_cache = load_global_ref_cache(global_manifest, layer)
        log(f"  global refs: {len(global_cache)} words")
        loco_caches = {spk: load_loco_ref_cache(spk, layer) for spk in sorted(TORGO_CONTROL_SPEAKERS)}

        rows = []
        n_ctrl = n_dys = n_skip = 0
        for _, r in meta.iterrows():
            utt_id, spk, word = r["utterance_id"], r["speaker_id"], r["word"]
            is_ctrl, sev = bool(r["is_ctrl"]), int(r["severity"])
            entry = ssl_index.get(utt_id)
            if entry is None or entry.get("wavlm_large") is None:
                n_skip += 1; continue
            emb_path = PROJECT_ROOT / entry["wavlm_large"]
            if not emb_path.exists():
                n_skip += 1; continue
            global_ref = global_cache.get(word)
            if global_ref is None:
                n_skip += 1; continue
            if is_ctrl:
                loco_ref = loco_caches[spk].get(word)
                if loco_ref is None:
                    n_skip += 1; continue
            utt_frames = np.load(str(emb_path))[layer_pos]
            atd_leaked = atd_from_ref(utt_frames, global_ref)
            atd_deleaked = atd_from_ref(utt_frames, loco_ref) if is_ctrl else atd_leaked
            rows.append({"speaker_id": spk, "severity_score": sev, "is_control": is_ctrl,
                         "atd_leaked": atd_leaked, "atd_deleaked": atd_deleaked})
            if is_ctrl: n_ctrl += 1
            else: n_dys += 1
            if (n_ctrl + n_dys) % 500 == 0:
                log(f"  scored {n_ctrl + n_dys} (ctrl={n_ctrl}, dys={n_dys}) ...")

        df = pd.DataFrame(rows)
        spk_df = df.groupby("speaker_id").agg(
            mean_atd_leaked=("atd_leaked", "mean"),
            mean_atd_deleaked=("atd_deleaked", "mean"),
            n_utterances=("atd_leaked", "count"),
            severity_score=("severity_score", "first"),
            is_control=("is_control", "first")).reset_index()

        sev_arr = spk_df["severity_score"].to_numpy(float)
        leaked = spk_df["mean_atd_leaked"].to_numpy(float)
        deleaked = spk_df["mean_atd_deleaked"].to_numpy(float)
        rho_l, p_l = spearmanr(sev_arr, leaked)
        rho_d, p_d = spearmanr(sev_arr, deleaked)
        dys = spk_df[spk_df["severity_score"] > 0]
        rho_dys_d, p_dys_d = spearmanr(dys["severity_score"].to_numpy(float),
                                       dys["mean_atd_deleaked"].to_numpy(float))
        c0l = spk_df[spk_df.severity_score == 0]["mean_atd_leaked"].to_numpy(float)
        c2l = spk_df[spk_df.severity_score == 2]["mean_atd_leaked"].to_numpy(float)
        c0d = spk_df[spk_df.severity_score == 0]["mean_atd_deleaked"].to_numpy(float)
        c2d = spk_df[spk_df.severity_score == 2]["mean_atd_deleaked"].to_numpy(float)

        report["by_layer"][f"layer{layer}"] = {
            "layer": layer, "n_speakers": int(len(spk_df)),
            "n_control_utts": n_ctrl, "n_dys_utts": n_dys, "n_skipped": n_skip,
            "leaked_rho": float(rho_l), "leaked_p": float(p_l),
            "leaked_bootstrap_ci": bootstrap_ci(sev_arr, leaked),
            "deleaked_rho": float(rho_d), "deleaked_p": float(p_d),
            "deleaked_bootstrap_ci": bootstrap_ci(sev_arr, deleaked),
            "delta_rho_leaked_minus_deleaked": float(rho_l - rho_d),
            "dysarthric_only_deleaked_rho": float(rho_dys_d),
            "dysarthric_only_deleaked_p": float(p_dys_d),
            "cohens_d_leaked_ctrl_vs_severe": cohens_d(c0l, c2l),
            "cohens_d_deleaked_ctrl_vs_severe": cohens_d(c0d, c2d),
            "control_mean_leaked": float(c0l.mean()), "control_mean_deleaked": float(c0d.mean()),
            "severe_mean": float(c2l.mean()),
        }
        log(f"  L{layer}: LEAKED rho={rho_l:.4f} p={p_l:.6f} | "
            f"DE-LEAKED rho={rho_d:.4f} p={p_d:.6f} | delta={rho_l - rho_d:+.4f}")
        if layer == 9:
            spk_df.sort_values("severity_score").to_csv(
                RESULTS_DIR / "speaker_atd_deleaked_fast.csv", index=False)

    with open(RESULTS_DIR / "deleak_report_fast.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"wrote results/deleak_report_fast.json  | TOTAL {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
