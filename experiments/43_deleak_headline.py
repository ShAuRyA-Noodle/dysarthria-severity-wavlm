"""
Experiment 43 — De-leaked headline ATD correlation.

Isolates the CONTROL SELF-INCLUSION LEAKAGE in the published headline
(speaker-level Spearman rho=0.756, n=15, layer9/radius1).

For control utterances:
  * LEAKED    = scored vs the GLOBAL full-control reference + PCA basis
                (reproduces current published behavior; control's own
                 speaker is in the reference).
  * DE-LEAKED = scored vs the LEAVE-ONE-CONTROL-OUT (LOCO) reference + PCA
                basis built from the OTHER controls only.
Dysarthric utterances are leakage-free by construction (never enter any
reference); they are scored vs the GLOBAL reference in BOTH conditions,
so only the control side differs between leaked and de-leaked.

DTW radius is held FIXED across leaked/de-leaked within a comparison.
We run radius=30 (rigorous, matches constants.FASTDTW_RADIUS) and
radius=1 (the value the published headline actually used, since
atd_metric.compute_single_atd calls fastdtw without a radius kwarg).

Layers: 1 (paper's current "primary") and 9 (historical headline).

Outputs:
  results/atd_scores_deleaked_layer{1,9}.csv   (utterance-level, both conditions)
  results/speaker_atd_deleaked_layer{1,9}.csv  (speaker-level, both conditions)
  results/deleak_report.json                   (full stats per layer/radius)

Seed = 42 (PCA models already fit with seed 42; bootstrap uses seed 42).
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
from src.seed_utils import set_global_seed  # noqa: E402

from src.constants import (
    LAYER_POSITION_MAP,
    RANDOM_SEED,
    TORGO_CONTROL_SPEAKERS,
    TORGO_DYSARTHRIC_SPEAKERS,
    TORGO_SEVERITY,
)

LAYERS = [1, 9]
RADII = [30, 1]
N_BOOTSTRAP = 10000

META_PATH = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
INDEX_PATH = PROJECT_ROOT / "data" / "embeddings" / "ssl_index.json"
GLOBAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "references" / "reference_manifest.json"
LOCO_ROOT = PROJECT_ROOT / "data" / "processed" / "references_loco"
RESULTS_DIR = PROJECT_ROOT / "results"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_global_ref_cache(manifest: dict, layer: int) -> dict:
    """Load all GLOBAL references for a layer. Key = word (verbatim)."""
    cache = {}
    for key, info in manifest.get("word_layer_ref", {}).items():
        # manifest key is f"{word}_{layer}" — split off the trailing _<layer>
        w, lyr = key.rsplit("_", 1)
        if int(lyr) != layer:
            continue
        p = PROJECT_ROOT / info["path"]
        if p.exists():
            with open(p, "rb") as f:
                cache[w] = pickle.load(f)
    return cache


def load_loco_ref_cache(held_out: str, layer: int) -> dict:
    """Load all LOCO references for a held-out speaker + layer. Key = word."""
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


def atd_from_ref(utt_frames: np.ndarray, ref_data: dict, radius: int) -> float:
    """Project utt frames into ref PCA basis, DTW vs ref, length-normalize."""
    pca = ref_data["pca_model"]
    reference = ref_data["reference"]
    utt_reduced = pca.transform(utt_frames)
    distance, _ = fastdtw(utt_reduced, reference, radius=radius, dist=euclidean)
    return float(distance / max(len(utt_reduced), len(reference)))


def cohens_d(group_lo: np.ndarray, group_hi: np.ndarray) -> float:
    """Speaker-level Cohen's d, (hi - lo) / pooled_sd. Positive => hi larger."""
    n1, n2 = len(group_lo), len(group_hi)
    if n1 < 2 or n2 < 2:
        return float("nan")
    v1, v2 = group_lo.var(ddof=1), group_hi.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((group_hi.mean() - group_lo.mean()) / pooled)


def bootstrap_ci_spearman(sev: np.ndarray, vals: np.ndarray, seed: int = RANDOM_SEED,
                          n_boot: int = N_BOOTSTRAP) -> list:
    """Seed-42 percentile bootstrap 95% CI for Spearman rho (speaker-level)."""
    rng = np.random.default_rng(seed)
    n = len(sev)
    rhos = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s, v = sev[idx], vals[idx]
        # spearmanr returns nan if a resample has zero variance; skip those
        if np.all(s == s[0]) or np.all(v == v[0]):
            continue
        r, _ = spearmanr(s, v)
        if not np.isnan(r):
            rhos.append(r)
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return [float(lo), float(hi)]


def main() -> None:
    set_global_seed(42)
    t0 = time.time()
    log("loading metadata + ssl index + global manifest ...")
    meta = pd.read_csv(META_PATH)
    with open(INDEX_PATH) as f:
        ssl_index = json.load(f)
    with open(GLOBAL_MANIFEST) as f:
        global_manifest = json.load(f)

    # TORGO only, severity known.
    meta = meta[meta["speaker_id"].isin(TORGO_SEVERITY.keys())].copy()
    meta["severity"] = meta["speaker_id"].map(TORGO_SEVERITY)
    meta["is_ctrl"] = meta["speaker_id"].isin(TORGO_CONTROL_SPEAKERS)
    log(f"TORGO utterances in metadata: {len(meta)}  "
        f"(controls={meta['is_ctrl'].sum()}, dysarthric={(~meta['is_ctrl']).sum()})")

    report = {"meta": {
        "n_bootstrap": N_BOOTSTRAP, "seed": RANDOM_SEED,
        "control_speakers": sorted(TORGO_CONTROL_SPEAKERS),
        "dysarthric_speakers": sorted(TORGO_DYSARTHRIC_SPEAKERS),
        "severity_map": TORGO_SEVERITY,
    }, "by_layer_radius": {}}

    for layer in LAYERS:
        layer_pos = LAYER_POSITION_MAP[layer]
        log(f"===== LAYER {layer} (array pos {layer_pos}) =====")

        log("  loading GLOBAL ref cache ...")
        global_cache = load_global_ref_cache(global_manifest, layer)
        log(f"    global refs: {len(global_cache)} words")

        log("  loading LOCO ref caches (7 controls) ...")
        loco_caches = {spk: load_loco_ref_cache(spk, layer) for spk in sorted(TORGO_CONTROL_SPEAKERS)}
        for spk, c in loco_caches.items():
            log(f"    LOCO[{spk}]: {len(c)} words")

        # ---- per-utterance: load embedding once, compute all radii ----
        rows = []
        n_ctrl_done = 0
        n_dys_done = 0
        n_skip_no_emb = 0
        n_skip_no_ref = 0

        for _, r in meta.iterrows():
            utt_id = r["utterance_id"]
            spk = r["speaker_id"]
            word = r["word"]
            is_ctrl = bool(r["is_ctrl"])
            sev = int(r["severity"])

            entry = ssl_index.get(utt_id)
            if entry is None or entry.get("wavlm_large") is None:
                n_skip_no_emb += 1
                continue
            emb_path = PROJECT_ROOT / entry["wavlm_large"]
            if not emb_path.exists():
                n_skip_no_emb += 1
                continue

            global_ref = global_cache.get(word)
            if global_ref is None:
                # no global reference for this word/layer -> not scoreable, skip
                n_skip_no_ref += 1
                continue

            if is_ctrl:
                loco_ref = loco_caches[spk].get(word)
                if loco_ref is None:
                    # control word has global ref but no LOCO ref for this held-out
                    # speaker (e.g. <2 OTHER controls said it). Cannot do a clean
                    # leaked-vs-deleaked comparison for this utt -> skip BOTH sides.
                    n_skip_no_ref += 1
                    continue

            emb = np.load(str(emb_path))            # (n_layers, T, 1024)
            utt_frames = emb[layer_pos]             # (T, 1024)

            row = {
                "utterance_id": utt_id, "speaker_id": spk, "word": word,
                "severity_score": sev, "is_control": is_ctrl,
            }
            for radius in RADII:
                atd_leaked = atd_from_ref(utt_frames, global_ref, radius)
                if is_ctrl:
                    atd_deleaked = atd_from_ref(utt_frames, loco_ref, radius)
                else:
                    # dysarthric: identical in both conditions (global ref)
                    atd_deleaked = atd_leaked
                row[f"atd_leaked_r{radius}"] = atd_leaked
                row[f"atd_deleaked_r{radius}"] = atd_deleaked
            rows.append(row)

            if is_ctrl:
                n_ctrl_done += 1
            else:
                n_dys_done += 1
            if (n_ctrl_done + n_dys_done) % 500 == 0:
                log(f"    scored {n_ctrl_done + n_dys_done} utts "
                    f"(ctrl={n_ctrl_done}, dys={n_dys_done}) ...")

        utt_df = pd.DataFrame(rows)
        utt_csv = RESULTS_DIR / f"atd_scores_deleaked_layer{layer}.csv"
        utt_df.to_csv(utt_csv, index=False)
        log(f"  wrote {utt_csv}  (rows={len(utt_df)}, ctrl={n_ctrl_done}, dys={n_dys_done}, "
            f"skip_no_emb={n_skip_no_emb}, skip_no_ref={n_skip_no_ref})")

        # ---- speaker-level aggregation + stats per radius ----
        spk_frames = {}
        for radius in RADII:
            lc, dc = f"atd_leaked_r{radius}", f"atd_deleaked_r{radius}"
            spk_df = utt_df.groupby("speaker_id").agg(
                mean_atd_leaked=(lc, "mean"),
                mean_atd_deleaked=(dc, "mean"),
                n_utterances=(lc, "count"),
                severity_score=("severity_score", "first"),
                is_control=("is_control", "first"),
            ).reset_index()
            spk_frames[radius] = spk_df

            sev_arr = spk_df["severity_score"].to_numpy(dtype=float)
            leaked_arr = spk_df["mean_atd_leaked"].to_numpy(dtype=float)
            deleaked_arr = spk_df["mean_atd_deleaked"].to_numpy(dtype=float)

            n_speakers = len(spk_df)
            rho_l, p_l = spearmanr(sev_arr, leaked_arr)
            rho_d, p_d = spearmanr(sev_arr, deleaked_arr)
            ci_d = bootstrap_ci_spearman(sev_arr, deleaked_arr)
            ci_l = bootstrap_ci_spearman(sev_arr, leaked_arr)

            # dysarthric-only (n=8)
            dys = spk_df[spk_df["severity_score"] > 0]
            dys_sev = dys["severity_score"].to_numpy(dtype=float)
            rho_dys_l, p_dys_l = spearmanr(dys_sev, dys["mean_atd_leaked"].to_numpy(dtype=float))
            rho_dys_d, p_dys_d = spearmanr(dys_sev, dys["mean_atd_deleaked"].to_numpy(dtype=float))

            # Cohen's d control(sev=0) vs severe(sev=2), speaker-level
            ctrl_l = spk_df[spk_df["severity_score"] == 0]["mean_atd_leaked"].to_numpy(dtype=float)
            sev2_l = spk_df[spk_df["severity_score"] == 2]["mean_atd_leaked"].to_numpy(dtype=float)
            ctrl_d = spk_df[spk_df["severity_score"] == 0]["mean_atd_deleaked"].to_numpy(dtype=float)
            sev2_d = spk_df[spk_df["severity_score"] == 2]["mean_atd_deleaked"].to_numpy(dtype=float)
            d_leaked = cohens_d(ctrl_l, sev2_l)
            d_deleaked = cohens_d(ctrl_d, sev2_d)

            key = f"layer{layer}_radius{radius}"
            report["by_layer_radius"][key] = {
                "layer": layer, "radius": radius, "n_speakers": int(n_speakers),
                "n_control_utts_rescored": int(n_ctrl_done),
                "n_dysarthric_utts": int(n_dys_done),
                "leaked_rho": float(rho_l), "leaked_p": float(p_l),
                "leaked_bootstrap_ci": ci_l,
                "deleaked_rho": float(rho_d), "deleaked_p": float(p_d),
                "deleaked_bootstrap_ci": ci_d,
                "delta_rho_leaked_minus_deleaked": float(rho_l - rho_d),
                "dysarthric_only_leaked_rho": float(rho_dys_l),
                "dysarthric_only_leaked_p": float(p_dys_l),
                "dysarthric_only_rho": float(rho_dys_d),
                "dysarthric_only_p": float(p_dys_d),
                "cohens_d_leaked_control_vs_severe": float(d_leaked),
                "cohens_d": float(d_deleaked),
                "cohens_d_deleaked_control_vs_severe": float(d_deleaked),
                "leaked_group_means": {
                    "control": float(ctrl_l.mean()), "severe": float(sev2_l.mean()),
                },
                "deleaked_group_means": {
                    "control": float(ctrl_d.mean()), "severe": float(sev2_d.mean()),
                },
            }
            log(f"  L{layer} r{radius}: leaked rho={rho_l:.4f} p={p_l:.6f} | "
                f"deleaked rho={rho_d:.4f} p={p_d:.6f} | delta={rho_l - rho_d:+.4f}")

        # write speaker-level CSV (use radius=30 as the canonical columns, but
        # include both radii so nothing is lost)
        base = spk_frames[30][["speaker_id", "severity_score", "is_control", "n_utterances"]].copy()
        for radius in RADII:
            sf = spk_frames[radius]
            base[f"mean_atd_leaked_r{radius}"] = sf["mean_atd_leaked"].values
            base[f"mean_atd_deleaked_r{radius}"] = sf["mean_atd_deleaked"].values
        spk_csv = RESULTS_DIR / f"speaker_atd_deleaked_layer{layer}.csv"
        base.sort_values("severity_score").to_csv(spk_csv, index=False)
        log(f"  wrote {spk_csv}")

    out_json = RESULTS_DIR / "deleak_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log(f"wrote {out_json}")
    log(f"TOTAL elapsed {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
