"""Reference trajectory builder — aggregates control-speaker embeddings per word."""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Severity mapping (hardcoded, never change) ────────────────────────
TORGO_SEVERITY = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}

RANDOM_SEED = 42

# Layer index mapping: position in stacked array → actual transformer layer
# TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]
LAYER_IDX_MAP = {1: 0, 3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 18: 6, 21: 7, 23: 8}


def pca_reduce_trajectory(
    trajectory: np.ndarray,
    n_components: int = 64,
    pca_model: Optional[PCA] = None,
) -> tuple[np.ndarray, PCA]:
    """
    PCA-reduce trajectory from (T, 1024) to (T, n_components).

    If pca_model provided: use existing model (for test utterances).
    If pca_model=None: fit new PCA on this trajectory.
    This makes DTW ~16x faster with negligible quality loss.

    Args:
        trajectory: Frame-level embeddings, shape (T, 1024).
        n_components: Target dimensionality (default 64).
        pca_model: Pre-fitted PCA model, or None to fit a new one.

    Returns:
        Tuple of (reduced_trajectory shape (T, n_components), pca_model).
    """
    if pca_model is None:
        pca_model = PCA(n_components=n_components, random_state=RANDOM_SEED)
        pca_model.fit(trajectory)

    reduced = pca_model.transform(trajectory)
    return reduced, pca_model


def dtw_align_trajectories(
    trajectory_a: np.ndarray,
    trajectory_b: np.ndarray,
) -> tuple[float, list]:
    """
    DTW-align two trajectories using fastdtw.

    Both must be (T, n_dims) — PCA-reduce first.

    Args:
        trajectory_a: Reference trajectory, shape (T_a, D).
        trajectory_b: Query trajectory, shape (T_b, D).

    Returns:
        Tuple of (dtw_distance, alignment_path).
    """
    distance, path = fastdtw(trajectory_a, trajectory_b, dist=euclidean)
    return distance, path


def _warp_trajectory_to_template(
    trajectory: np.ndarray,
    path: list,
    template_length: int,
) -> np.ndarray:
    """
    Warp a trajectory to match template length using DTW path.

    For each template frame, averages all trajectory frames aligned to it.

    Args:
        trajectory: Source trajectory, shape (T_src, D).
        path: DTW alignment path (list of (template_idx, source_idx) tuples).
        template_length: Length of the template trajectory.

    Returns:
        Warped trajectory of shape (template_length, D).
    """
    n_dims = trajectory.shape[1]
    warped = np.zeros((template_length, n_dims))
    counts = np.zeros(template_length)

    for template_idx, source_idx in path:
        if template_idx < template_length and source_idx < len(trajectory):
            warped[template_idx] += trajectory[source_idx]
            counts[template_idx] += 1

    # Avoid division by zero
    counts = np.maximum(counts, 1)
    warped /= counts[:, np.newaxis]

    return warped


def build_word_reference(
    word: str,
    ssl_index: dict,
    metadata_df: pd.DataFrame,
    model_name: str = "wavlm_large",
    layer_idx: int = 3,
    n_components: int = 64,
    project_root: Optional[Path] = None,
) -> Optional[dict]:
    """
    Build reference trajectory for a single word from control speakers.

    Algorithm:
        1. Find all utterances of `word` where is_control=True
        2. Load their frame-level embeddings at layer_idx
        3. If fewer than 2 control utterances: return None (skip word)
        4. PCA-reduce all trajectories to 64 dims (fit PCA on concatenated data)
        5. Select the median-length trajectory as template
        6. DTW-align all other control trajectories to template
        7. Average aligned trajectories frame-by-frame = reference
        8. Also store: pca_model (needed to project test utterances)

    Args:
        word: The word/prompt to build a reference for.
        ssl_index: Master SSL index dict.
        metadata_df: Metadata DataFrame.
        model_name: SSL model name (key in ssl_index entries).
        layer_idx: Index into the stacked embedding array (0-8).
        n_components: PCA target dimensionality.
        project_root: Project root for resolving paths.

    Returns:
        Dict with reference trajectory, PCA model, and metadata.
        None if insufficient control data.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # 1. Find control utterances of this word
    ctrl_mask = (metadata_df["word"] == word) & (metadata_df["is_control"] == True)
    ctrl_rows = metadata_df[ctrl_mask]

    if len(ctrl_rows) < 2:
        return None

    # 2. Load frame-level embeddings
    trajectories = []
    speakers_used = []

    for _, row in ctrl_rows.iterrows():
        utt_id = row["utterance_id"]
        entry = ssl_index.get(utt_id)
        if entry is None or entry.get(model_name) is None:
            continue

        npy_path = project_root / entry[model_name]
        if not npy_path.exists():
            continue

        emb = np.load(str(npy_path))  # (n_layers, T, 1024)
        traj = emb[layer_idx]  # (T, 1024)
        trajectories.append(traj)
        speakers_used.append(row["speaker_id"])

    # 3. Check minimum utterances
    if len(trajectories) < 2:
        return None

    # 4. PCA-reduce: fit on concatenated control data
    all_frames = np.concatenate(trajectories, axis=0)  # (sum_T, 1024)
    actual_components = min(n_components, all_frames.shape[0], all_frames.shape[1])
    pca_model = PCA(n_components=actual_components, random_state=RANDOM_SEED)
    pca_model.fit(all_frames)

    reduced_trajectories = [pca_model.transform(t) for t in trajectories]

    # 5. Select median-length trajectory as template
    lengths = [len(t) for t in reduced_trajectories]
    median_len = int(np.median(lengths))
    # Find closest to median
    closest_idx = min(range(len(lengths)), key=lambda i: abs(lengths[i] - median_len))
    template = reduced_trajectories[closest_idx]

    # 6. DTW-align all trajectories to template
    aligned_trajectories = [template]  # Include template itself

    for i, traj in enumerate(reduced_trajectories):
        if i == closest_idx:
            continue
        _, path = dtw_align_trajectories(template, traj)
        warped = _warp_trajectory_to_template(traj, path, len(template))
        aligned_trajectories.append(warped)

    # 7. Average aligned trajectories
    reference = np.mean(aligned_trajectories, axis=0)  # (T_ref, n_components)

    return {
        "word": word,
        "reference": reference,
        "pca_model": pca_model,
        "n_control_speakers": len(set(speakers_used)),
        "n_control_utterances": len(trajectories),
        "layer_idx": layer_idx,
        "model_name": model_name,
        "control_speakers_used": sorted(set(speakers_used)),
        "reference_length": len(reference),
    }


def build_all_references(
    ssl_index: dict,
    metadata_df: pd.DataFrame,
    model_name: str = "wavlm_large",
    target_layers: Optional[list[int]] = None,
    output_dir: str = "data/processed/references/",
    min_control_utterances: int = 2,
    project_root: Optional[Path] = None,
) -> dict:
    """
    Build references for ALL words at ALL target layers.

    For each (word, layer) combination:
    - Build reference using control speakers
    - Save to: {output_dir}/{word}_layer{layer}.pkl (includes PCA model)
    - Track: which words had enough control data

    Args:
        ssl_index: Master SSL index dict.
        metadata_df: Metadata DataFrame.
        model_name: SSL model name.
        target_layers: Transformer layer indices (actual layer numbers).
        output_dir: Where to save .pkl files.
        min_control_utterances: Minimum control utterances needed per word.
        project_root: Project root for resolving paths.

    Returns:
        Reference manifest dict.
    """
    if target_layers is None:
        target_layers = [1, 3, 6, 9, 12, 15, 18, 21, 23]
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    out_path = project_root / output_dir
    out_path.mkdir(parents=True, exist_ok=True)

    # Get all words from control speakers
    ctrl_mask = metadata_df["is_control"] == True
    ctrl_df = metadata_df[ctrl_mask]
    word_counts = ctrl_df.groupby("word").size()
    candidate_words = word_counts[word_counts >= min_control_utterances].index.tolist()

    logger.info(f"Building references for {len(candidate_words)} words × {len(target_layers)} layers")

    manifest = {
        "word_layer_ref": {},
        "words_with_references": [],
        "words_skipped": [],
        "total_references": 0,
        "model_name": model_name,
        "target_layers": target_layers,
    }

    words_with_refs = set()
    words_skipped = set()

    total_combos = len(candidate_words) * len(target_layers)
    pbar = tqdm(total=total_combos, desc="Building references")

    for word in candidate_words:
        word_has_ref = False
        for layer in target_layers:
            pbar.update(1)
            # Map transformer layer to stacked array index
            layer_arr_idx = LAYER_IDX_MAP.get(layer)
            if layer_arr_idx is None:
                logger.warning(f"Layer {layer} not in LAYER_IDX_MAP, skipping")
                continue

            try:
                ref_data = build_word_reference(
                    word=word,
                    ssl_index=ssl_index,
                    metadata_df=metadata_df,
                    model_name=model_name,
                    layer_idx=layer_arr_idx,
                    n_components=64,
                    project_root=project_root,
                )

                if ref_data is None:
                    continue

                # Sanitize word for filename (replace special chars)
                safe_word = word.replace("/", "_").replace("\\", "_").replace(" ", "_")
                pkl_filename = f"{safe_word}_layer{layer}.pkl"
                pkl_path = out_path / pkl_filename

                with open(pkl_path, "wb") as f:
                    pickle.dump(ref_data, f)

                rel_path = str(pkl_path.relative_to(project_root)).replace("\\", "/")
                key = f"{word}_{layer}"
                manifest["word_layer_ref"][key] = {
                    "path": rel_path,
                    "n_control_speakers": ref_data["n_control_speakers"],
                    "n_control_utterances": ref_data["n_control_utterances"],
                    "reference_length": ref_data["reference_length"],
                    "control_speakers_used": ref_data["control_speakers_used"],
                }
                manifest["total_references"] += 1
                word_has_ref = True

            except Exception as e:
                logger.error(f"Failed to build reference for word='{word}', layer={layer}: {e}")
                continue

        if word_has_ref:
            words_with_refs.add(word)
        else:
            words_skipped.add(word)

    pbar.close()

    manifest["words_with_references"] = sorted(words_with_refs)
    manifest["words_skipped"] = sorted(words_skipped)

    # Save manifest
    manifest_path = out_path / "reference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Reference Building Complete")
    print(f"{'='*60}")
    print(f"  Words with references: {len(words_with_refs)}")
    print(f"  Words skipped: {len(words_skipped)}")
    print(f"  Total references (word × layer): {manifest['total_references']}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}\n")

    return manifest
