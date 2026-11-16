"""
Experiment 11 — Ablation study for ATD metric.

6 ablation experiments to validate each component of ATD.
All use global speaker-level evaluation (15 speakers).
"""
import json
import logging
import pickle
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

from src.atd_metric import TORGO_SEVERITY, LAYER_POSITION_MAP

ALL_SPEAKERS = sorted(TORGO_SEVERITY.keys())
DYSARTHRIC_SPEAKERS = sorted(s for s, sev in TORGO_SEVERITY.items() if sev > 0)
TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def load_speaker_atd_from_file(layer: int) -> dict:
    """Load pre-computed speaker ATD for a given layer."""
    path = PROJECT_ROOT / f"results/speaker_atd_layer{layer}.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return dict(zip(df.speaker_id, df.mean_atd))


def _loso_spearman(speaker_scores: dict) -> float:
    """LOSO Spearman: dysarthric speakers only.

    Controls are reference-builders, never held out. Each dysarthric speaker
    is effectively LOSO already (their utterances were never in any reference).
    Hence LOSO rho == Spearman of dysarthric-only mean scores vs severity.
    """
    y_true, y_pred = [], []
    for spk in DYSARTHRIC_SPEAKERS:
        if spk in speaker_scores:
            y_true.append(TORGO_SEVERITY[spk])
            y_pred.append(speaker_scores[spk])
    if len(y_true) < 3:
        return float("nan")
    rho, _ = spearmanr(y_true, y_pred)
    return round(float(rho), 4)


def evaluate_speaker_metric(speaker_scores: dict, metric_name: str) -> dict:
    """Evaluate a speaker-level metric against severity.

    Returns BOTH global Spearman (15 speakers) and LOSO Spearman (8 dysarthric).
    Critical: LOSO must be computed from THIS variant's per-speaker scores,
    not from cached layer-9 ATD scores (which was the bug in
    add_loso_to_ablations.py producing identical 0.3944 across rows).
    """
    y_true, y_pred = [], []
    for spk in ALL_SPEAKERS:
        if spk in speaker_scores:
            y_true.append(TORGO_SEVERITY[spk])
            y_pred.append(speaker_scores[spk])

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    rho_global = 0.0
    rmse: float | str = "N/A"
    if len(y_true_arr) >= 3:
        rho_global, _ = spearmanr(y_true_arr, y_pred_arr)
        # RMSE only meaningful if scores are on severity scale; we report it
        # mainly for ATD which is unitless. Kept for backward compat.
        rmse = round(float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))), 4)

    rho_loso = _loso_spearman(speaker_scores)

    return {
        "variant": metric_name,
        "spearman_global": round(float(rho_global), 4),
        "spearman_loso": rho_loso,
        "rmse": rmse,
        "n_speakers": len(y_true_arr),
        "n_dysarthric_in_loso": sum(1 for s in DYSARTHRIC_SPEAKERS if s in speaker_scores),
    }


def ablation_1_layer1():
    """Full ATD with Layer 1 (best layer)."""
    scores = load_speaker_atd_from_file(1)
    result = evaluate_speaker_metric(scores, "Full ATD — Layer 1")
    result["notes"] = "Best layer by Spearman"
    return result


def ablation_2_layer9():
    """Full ATD with Layer 9."""
    scores = load_speaker_atd_from_file(9)
    result = evaluate_speaker_metric(scores, "Full ATD — Layer 9")
    result["notes"] = "Mid-network articulatory layer"
    return result


def ablation_3_no_dtw():
    """No DTW: mean L2 distance from reference mean embedding."""
    print("  Running Ablation 3: No DTW (mean L2 distance)...")
    ssl_index = json.load(open(PROJECT_ROOT / "data/embeddings/ssl_index.json"))
    ref_manifest = json.load(open(PROJECT_ROOT / "data/processed/references/reference_manifest.json"))
    word_refs = ref_manifest.get("word_layer_ref", {})
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    layer_idx = 9
    layer_pos = LAYER_POSITION_MAP[layer_idx]

    speaker_scores = {}
    for _, row in metadata_df.iterrows():
        uid = row["utterance_id"]
        spk = row["speaker_id"]

        entry = ssl_index.get(uid)
        if entry is None or entry.get("wavlm_large") is None:
            continue
        word = entry.get("word", "")
        if not word:
            continue

        emb_path = PROJECT_ROOT / str(entry["wavlm_large"])
        if not emb_path.exists():
            continue
        emb = np.load(str(emb_path))
        utt_mean = emb[layer_pos].mean(axis=0)  # (1024,)

        ref_key = f"{word}_{layer_idx}"
        ref_info = word_refs.get(ref_key)
        if ref_info is None:
            continue
        ref_path = PROJECT_ROOT / ref_info["path"]
        if not ref_path.exists():
            continue
        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)
        ref_traj = ref_data["reference"]  # (T_ref, 64)
        ref_mean = ref_traj.mean(axis=0)

        pca_model = ref_data.get("pca_model")
        if pca_model is not None:
            utt_mean_pca = pca_model.transform(utt_mean.reshape(1, -1)).flatten()
        else:
            utt_mean_pca = utt_mean[:len(ref_mean)]

        dist = np.linalg.norm(utt_mean_pca - ref_mean)
        speaker_scores.setdefault(spk, []).append(dist)

    speaker_mean = {spk: np.mean(dists) for spk, dists in speaker_scores.items()}
    result = evaluate_speaker_metric(speaker_mean, "No DTW — Mean L2")
    result["notes"] = "Mean embed distance, no trajectory"
    return result


def ablation_4_no_pca():
    """No PCA: DTW on full 1024-dim (limited to 200 utterances per speaker)."""
    print("  Running Ablation 4: No PCA (full 1024-dim DTW, sampled)...")
    ssl_index = json.load(open(PROJECT_ROOT / "data/embeddings/ssl_index.json"))
    ref_manifest = json.load(open(PROJECT_ROOT / "data/processed/references/reference_manifest.json"))
    word_refs = ref_manifest.get("word_layer_ref", {})
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    layer_idx = 9
    layer_pos = LAYER_POSITION_MAP[layer_idx]

    speaker_scores = {}
    speaker_count = {}

    for _, row in metadata_df.iterrows():
        uid = row["utterance_id"]
        spk = row["speaker_id"]

        if speaker_count.get(spk, 0) >= 200:
            continue

        entry = ssl_index.get(uid)
        if entry is None or entry.get("wavlm_large") is None:
            continue
        word = entry.get("word", "")
        if not word:
            continue

        emb_path = PROJECT_ROOT / str(entry["wavlm_large"])
        if not emb_path.exists():
            continue
        emb = np.load(str(emb_path))
        utt_traj = emb[layer_pos]  # (T, 1024)

        ref_key = f"{word}_{layer_idx}"
        ref_info = word_refs.get(ref_key)
        if ref_info is None:
            continue
        ref_path = PROJECT_ROOT / ref_info["path"]
        if not ref_path.exists():
            continue
        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)

        pca = ref_data.get("pca_model")
        if pca is None:
            continue
        ref_traj_pca = ref_data["reference"]  # (T_ref, 64)
        ref_traj_1024 = pca.inverse_transform(ref_traj_pca)  # (T_ref, 1024)

        try:
            dist, _ = fastdtw(utt_traj, ref_traj_1024, radius=1, dist=euclidean)
            speaker_scores.setdefault(spk, []).append(dist)
            speaker_count[spk] = speaker_count.get(spk, 0) + 1
        except Exception:
            continue

    speaker_mean = {spk: np.mean(dists) for spk, dists in speaker_scores.items()}
    result = evaluate_speaker_metric(speaker_mean, "No PCA — 1024d DTW")
    result["notes"] = "Full dim DTW, sampled 200/spk"
    return result


def ablation_5_mean_pool():
    """Mean-pool distance: no trajectory structure."""
    print("  Running Ablation 5: Mean-pool distance...")
    ssl_index = json.load(open(PROJECT_ROOT / "data/embeddings/ssl_index.json"))
    ref_manifest = json.load(open(PROJECT_ROOT / "data/processed/references/reference_manifest.json"))
    word_refs = ref_manifest.get("word_layer_ref", {})
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    layer_idx = 9
    layer_pos = LAYER_POSITION_MAP[layer_idx]

    speaker_scores = {}
    for _, row in metadata_df.iterrows():
        uid = row["utterance_id"]
        spk = row["speaker_id"]

        entry = ssl_index.get(uid)
        if entry is None or entry.get("wavlm_large") is None:
            continue
        word = entry.get("word", "")
        if not word:
            continue

        emb_path = PROJECT_ROOT / str(entry["wavlm_large"])
        if not emb_path.exists():
            continue
        emb = np.load(str(emb_path))
        utt_pool = emb[layer_pos].mean(axis=0)  # (1024,)

        ref_key = f"{word}_{layer_idx}"
        ref_info = word_refs.get(ref_key)
        if ref_info is None:
            continue
        ref_path = PROJECT_ROOT / ref_info["path"]
        if not ref_path.exists():
            continue
        with open(ref_path, "rb") as f:
            ref_data = pickle.load(f)

        ref_traj = ref_data["reference"]  # (T_ref, 64)
        ref_pool_pca = ref_traj.mean(axis=0)
        pca = ref_data.get("pca_model")
        if pca:
            utt_pca = pca.transform(utt_pool.reshape(1, -1)).flatten()
        else:
            utt_pca = utt_pool[:64]

        dist = np.linalg.norm(utt_pca - ref_pool_pca)
        speaker_scores.setdefault(spk, []).append(dist)

    speaker_mean = {spk: np.mean(dists) for spk, dists in speaker_scores.items()}
    result = evaluate_speaker_metric(speaker_mean, "Mean-pool distance")
    result["notes"] = "No trajectory, pool then L2"
    return result


def ablation_6_mfcc_trajectory():
    """MFCC trajectory with DTW — tests if SSL representations are necessary."""
    print("  Running Ablation 6: MFCC trajectory DTW...")
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    metadata_df["filepath_processed"] = metadata_df["filepath_processed"].astype(str)

    controls = [s for s, sev in TORGO_SEVERITY.items() if sev == 0]

    # Build MFCC references from controls
    word_refs = {}
    for _, row in metadata_df.iterrows():
        spk = row["speaker_id"]
        if spk not in controls:
            continue
        word = row.get("word", "")
        if not word or word == "nan":
            continue
        fpath = row["filepath_processed"]
        if fpath == "nan" or pd.isna(fpath):
            continue
        audio_path = PROJECT_ROOT / str(fpath)
        if not audio_path.exists():
            continue
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            if len(y) < 512:
                continue
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T  # (T, 13)
            word_refs.setdefault(word, []).append(mfccs)
        except Exception:
            continue

    # Average references per word (use first as template, DTW-align others)
    word_mean_refs = {}
    for word, trajs in word_refs.items():
        if len(trajs) < 2:
            continue
        # Use the median-length trajectory as reference
        lengths = [len(t) for t in trajs]
        median_idx = np.argsort(lengths)[len(lengths) // 2]
        word_mean_refs[word] = trajs[median_idx]

    # Compute MFCC-DTW distance for all speakers
    speaker_scores = {}
    for _, row in metadata_df.iterrows():
        spk = row["speaker_id"]
        word = row.get("word", "")
        if word not in word_mean_refs:
            continue
        fpath = row["filepath_processed"]
        if fpath == "nan" or pd.isna(fpath):
            continue
        audio_path = PROJECT_ROOT / str(fpath)
        if not audio_path.exists():
            continue
        try:
            y, sr = librosa.load(str(audio_path), sr=16000)
            if len(y) < 512:
                continue
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T  # (T, 13)
            ref = word_mean_refs[word]
            dist, _ = fastdtw(mfccs, ref, radius=1, dist=euclidean)
            speaker_scores.setdefault(spk, []).append(dist)
        except Exception:
            continue

    speaker_mean = {spk: np.mean(dists) for spk, dists in speaker_scores.items()}
    result = evaluate_speaker_metric(speaker_mean, "MFCC trajectory DTW")
    result["notes"] = "Same framework, MFCC not SSL"
    return result


def main():
    set_global_seed(42)
    print("=" * 60)
    print("  Ablation Study — ATD Components")
    print("=" * 60)

    results = []

    print("\nAblation 1: Full ATD — Layer 1")
    results.append(ablation_1_layer1())

    print("Ablation 2: Full ATD — Layer 9")
    results.append(ablation_2_layer9())

    print("Ablation 3: No DTW")
    results.append(ablation_3_no_dtw())

    print("Ablation 4: No PCA")
    results.append(ablation_4_no_pca())

    print("Ablation 5: Mean-pool distance")
    results.append(ablation_5_mean_pool())

    print("Ablation 6: MFCC trajectory")
    results.append(ablation_6_mfcc_trajectory())

    # Save
    df = pd.DataFrame(results)
    # Reorder columns for readability.
    col_order = ["variant", "spearman_global", "spearman_loso", "rmse",
                 "n_speakers", "n_dysarthric_in_loso", "notes"]
    df = df[[c for c in col_order if c in df.columns]]
    out_path = PROJECT_ROOT / "results/tables/table2_ablations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\n{'='*72}")
    print("  Table 2 — Ablation Results (per-variant LOSO from each variant's scores)")
    print(f"{'='*72}")
    print(df.to_string(index=False))
    best_idx = df["spearman_global"].idxmax()
    print(
        f"\n  Best variant by global rho: {df.loc[best_idx, 'variant']} "
        f"(rho_global={df.loc[best_idx, 'spearman_global']:.4f}, "
        f"rho_loso={df.loc[best_idx, 'spearman_loso']})"
    )
    print(f"\nSaved to: {out_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
