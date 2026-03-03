"""ATD metric computation — trajectory deviation scoring via DTW on SSL embeddings."""

import json
import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from tqdm import tqdm

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

# Layer index mapping: transformer layer → position in stacked array
# TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]
LAYER_POSITION_MAP = {1: 0, 3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 18: 6, 21: 7, 23: 8}

TORGO_SEVERITY = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}


def compute_single_atd(
    utterance_id: str,
    ssl_index: dict,
    reference_cache: dict,
    model_name: str = "wavlm_large",
    layer_idx: int = 9,
    layer_position: int = 3,
    project_root: Optional[Path] = None,
) -> Optional[float]:
    """
    Compute ATD for a single utterance against its word's reference.

    Steps:
        1. Load utterance embedding from ssl_index
        2. Extract the target layer from stacked array
        3. Get word from ssl_index entry
        4. Load reference for that word+layer
        5. Project utterance to PCA space using reference's pca_model
        6. DTW distance between projected utterance and reference
        7. Normalize by max(T_utt, T_ref)

    Args:
        utterance_id: Unique utterance identifier.
        ssl_index: Master SSL index dict.
        reference_cache: Pre-loaded dict of {word_layer_key: ref_data}.
        model_name: SSL model name.
        layer_idx: Transformer layer number (e.g., 9).
        layer_position: Index in stacked array (e.g., 3 for layer 9).
        project_root: Project root for resolving paths.

    Returns:
        ATD score (float) or None if computation fails.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    try:
        # 1. Get utterance entry
        entry = ssl_index.get(utterance_id)
        if entry is None or entry.get(model_name) is None:
            return None

        # 2. Load embedding
        emb_path = project_root / entry[model_name]
        if not emb_path.exists():
            return None

        emb = np.load(str(emb_path))  # (n_layers, T, 1024)
        utt_frames = emb[layer_position]  # (T_utt, 1024)

        # 3. Get word
        word = entry.get("word", "")
        if not word:
            return None

        # 4. Get reference
        ref_key = f"{word}_{layer_idx}"
        ref_data = reference_cache.get(ref_key)
        if ref_data is None:
            return None

        # 5. Project to PCA space
        pca_model = ref_data["pca_model"]
        ref_trajectory = ref_data["reference"]  # (T_ref, 64)

        utt_reduced = pca_model.transform(utt_frames)  # (T_utt, 64)

        # 6. DTW distance
        distance, _ = fastdtw(utt_reduced, ref_trajectory, dist=euclidean)

        # 7. Normalize
        t_utt = len(utt_reduced)
        t_ref = len(ref_trajectory)
        atd = distance / max(t_utt, t_ref)

        return float(atd)

    except Exception as e:
        logger.debug(f"ATD failed for {utterance_id}: {e}")
        return None


def _load_reference_cache(
    reference_manifest: dict,
    layer_idx: int,
    project_root: Path,
) -> dict:
    """
    Pre-load all reference .pkl files for a given layer into memory.

    Args:
        reference_manifest: The reference manifest dict.
        layer_idx: Transformer layer number.
        project_root: Project root for resolving paths.

    Returns:
        Dict mapping 'word_layer' key to loaded reference data.
    """
    cache = {}
    for key, info in reference_manifest.get("word_layer_ref", {}).items():
        # Only load references for the target layer
        if not key.endswith(f"_{layer_idx}"):
            continue
        pkl_path = project_root / info["path"]
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                cache[key] = pickle.load(f)
    logger.info(f"Loaded {len(cache)} references for layer {layer_idx}")
    return cache


def compute_all_atd_scores(
    ssl_index: dict,
    metadata_df: pd.DataFrame,
    reference_manifest: dict,
    model_name: str = "wavlm_large",
    layer_idx: int = 9,
    output_path: str = "results/atd_scores.csv",
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Compute ATD for every utterance in the dataset.

    Includes controls (they should have low ATD) and dysarthric speakers.
    Saves intermediate results every 500 utterances for crash safety.

    Args:
        ssl_index: Master SSL index dict.
        metadata_df: Metadata DataFrame.
        reference_manifest: Reference manifest dict.
        model_name: SSL model name.
        layer_idx: Transformer layer number (e.g., 9).
        output_path: Where to save results CSV.
        project_root: Project root for resolving paths.

    Returns:
        DataFrame with ATD scores for all utterances.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    layer_position = LAYER_POSITION_MAP.get(layer_idx, 3)

    # Pre-load all references for this layer
    print(f"Loading references for layer {layer_idx}...")
    ref_cache = _load_reference_cache(reference_manifest, layer_idx, project_root)
    print(f"  Loaded {len(ref_cache)} word references")

    results = []
    n_computed = 0
    n_skipped = 0

    out_file = project_root / output_path

    for _, row in tqdm(
        metadata_df.iterrows(), total=len(metadata_df), desc="Computing ATD"
    ):
        utt_id = row["utterance_id"]
        speaker_id = row["speaker_id"]

        atd = compute_single_atd(
            utterance_id=utt_id,
            ssl_index=ssl_index,
            reference_cache=ref_cache,
            model_name=model_name,
            layer_idx=layer_idx,
            layer_position=layer_position,
            project_root=project_root,
        )

        results.append({
            "utterance_id": utt_id,
            "speaker_id": speaker_id,
            "word": row.get("word", ""),
            "severity_score": int(TORGO_SEVERITY.get(speaker_id, -1)),
            "atd_score": atd,
            "model_name": model_name,
            "layer_idx": layer_idx,
            "is_control": bool(TORGO_SEVERITY.get(speaker_id, -1) == 0),
            "session": row.get("session", ""),
        })

        if atd is not None:
            n_computed += 1
        else:
            n_skipped += 1

        # Save intermediate every 500
        if len(results) % 500 == 0:
            pd.DataFrame(results).to_csv(out_file, index=False)

    atd_df = pd.DataFrame(results)
    atd_df.to_csv(out_file, index=False)

    print(f"\nATD computation complete:")
    print(f"  Computed: {n_computed}")
    print(f"  Skipped (no reference): {n_skipped}")
    print(f"  Saved to: {out_file}")

    return atd_df


def aggregate_speaker_atd(
    atd_df: pd.DataFrame,
    output_path: str = "results/speaker_atd.csv",
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Aggregate utterance-level ATD to speaker level.

    For each speaker, computes mean, std, median ATD and utterance count.

    Args:
        atd_df: Utterance-level ATD DataFrame.
        output_path: Where to save speaker-level CSV.
        project_root: Project root for resolving paths.

    Returns:
        Speaker-level DataFrame.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    # Filter to utterances with valid ATD scores
    valid = atd_df.dropna(subset=["atd_score"])

    speaker_stats = valid.groupby("speaker_id").agg(
        mean_atd=("atd_score", "mean"),
        std_atd=("atd_score", "std"),
        median_atd=("atd_score", "median"),
        n_utterances=("atd_score", "count"),
        severity_score=("severity_score", "first"),
        is_control=("is_control", "first"),
    ).reset_index()

    out_file = project_root / output_path
    speaker_stats.to_csv(out_file, index=False)

    print(f"\nSpeaker-level ATD:")
    print(speaker_stats.to_string(index=False))
    print(f"\nSaved to: {out_file}")

    return speaker_stats


def _cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Args:
        group1: First group of values.
        group2: Second group of values.

    Returns:
        Cohen's d value.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group2.mean() - group1.mean()) / pooled_std)


def validate_atd_metric(
    atd_df: pd.DataFrame,
    speaker_atd_df: pd.DataFrame,
    output_dir: str = "results/",
    project_root: Optional[Path] = None,
) -> dict:
    """
    THE CRITICAL VALIDATION FUNCTION.
    Tests whether ATD actually measures dysarthria severity.

    Test 1: Utterance-level Spearman correlation
    Test 2: Speaker-level Spearman correlation (PRIMARY)
    Test 3: Severity group separation (Mann-Whitney U)
    Test 4: Effect size (Cohen's d)

    Args:
        atd_df: Utterance-level ATD DataFrame.
        speaker_atd_df: Speaker-level aggregated DataFrame.
        output_dir: Where to save the report.
        project_root: Project root.

    Returns:
        Dict with validation results.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    report = {}
    lines = []

    def p(s: str) -> None:
        print(s)
        lines.append(s)

    p("\n" + "=" * 55)
    p("  ATD VALIDATION REPORT")
    p("=" * 55)

    # Filter valid scores
    valid_utt = atd_df.dropna(subset=["atd_score"])

    # ── Test 1: Utterance-level correlation ──────────────────────────
    utt_rho, utt_p = spearmanr(
        valid_utt["severity_score"], valid_utt["atd_score"]
    )
    report["utterance_spearman"] = float(utt_rho)
    report["utterance_p_value"] = float(utt_p)
    p(f"\n  Test 1 — Utterance-level Spearman rho: {utt_rho:.3f} (p={utt_p:.2e})")

    # ── Test 2: Speaker-level correlation (PRIMARY) ──────────────────
    spk_rho, spk_p = spearmanr(
        speaker_atd_df["severity_score"], speaker_atd_df["mean_atd"]
    )
    report["speaker_spearman"] = float(spk_rho)
    report["p_value"] = float(spk_p)
    p(f"  Test 2 — Speaker-level Spearman rho:   {spk_rho:.3f} (p={spk_p:.4f})")

    # Also Pearson
    spk_pearson, spk_pearson_p = pearsonr(
        speaker_atd_df["severity_score"], speaker_atd_df["mean_atd"]
    )
    report["speaker_pearson"] = float(spk_pearson)
    report["speaker_pearson_p"] = float(spk_pearson_p)
    p(f"  Test 2b— Speaker-level Pearson r:      {spk_pearson:.3f} (p={spk_pearson_p:.4f})")

    # ── Test 3: Group separation ─────────────────────────────────────
    # Between severity=1 (mild) and severity=2 (severe)
    mild_atd = valid_utt[valid_utt["severity_score"] == 1]["atd_score"].values
    severe_atd = valid_utt[valid_utt["severity_score"] == 2]["atd_score"].values
    control_atd = valid_utt[valid_utt["severity_score"] == 0]["atd_score"].values

    report["group_means"] = {
        "control": float(control_atd.mean()) if len(control_atd) > 0 else None,
        "mild": float(mild_atd.mean()) if len(mild_atd) > 0 else None,
        "severe": float(severe_atd.mean()) if len(severe_atd) > 0 else None,
    }

    p(f"\n  Group means:")
    if len(control_atd) > 0:
        p(f"    Control (sev=0): {control_atd.mean():.4f} +/- {control_atd.std():.4f}  (n={len(control_atd)})")
    p(f"    Mild    (sev=1): {mild_atd.mean():.4f} +/- {mild_atd.std():.4f}  (n={len(mild_atd)})")
    p(f"    Severe  (sev=2): {severe_atd.mean():.4f} +/- {severe_atd.std():.4f}  (n={len(severe_atd)})")

    if len(mild_atd) > 0 and len(severe_atd) > 0:
        u_stat, mw_p = mannwhitneyu(mild_atd, severe_atd, alternative="less")
        report["mann_whitney_u"] = float(u_stat)
        report["mann_whitney_p"] = float(mw_p)
        p(f"\n  Test 3 — Mann-Whitney U (mild < severe): U={u_stat:.0f}, p={mw_p:.4f}")
    else:
        report["mann_whitney_p"] = 1.0
        p(f"\n  Test 3 — Mann-Whitney U: SKIPPED (insufficient data)")

    # Also test control vs dysarthric
    if len(control_atd) > 0:
        dysarthric_atd = valid_utt[valid_utt["severity_score"] > 0]["atd_score"].values
        u_cd, p_cd = mannwhitneyu(control_atd, dysarthric_atd, alternative="less")
        report["control_vs_dysarthric_p"] = float(p_cd)
        p(f"  Test 3b— Mann-Whitney U (ctrl < dys):  U={u_cd:.0f}, p={p_cd:.2e}")

    # ── Test 4: Effect size ──────────────────────────────────────────
    if len(mild_atd) > 0 and len(severe_atd) > 0:
        d = _cohens_d(mild_atd, severe_atd)
        report["cohens_d_mild_severe"] = float(d)
        effect_label = "small" if abs(d) < 0.5 else "medium" if abs(d) < 0.8 else "large"
        p(f"  Test 4 — Cohen's d (mild vs severe):   {d:.3f} ({effect_label})")

    if len(control_atd) > 0 and len(severe_atd) > 0:
        d_cs = _cohens_d(control_atd, severe_atd)
        report["cohens_d_control_severe"] = float(d_cs)
        effect_label_cs = "small" if abs(d_cs) < 0.5 else "medium" if abs(d_cs) < 0.8 else "large"
        p(f"  Test 4b— Cohen's d (ctrl vs severe):   {d_cs:.3f} ({effect_label_cs})")

    # ── Verdict ──────────────────────────────────────────────────────
    passed = spk_rho > 0.55 and spk_p < 0.05
    report["passed"] = passed

    p(f"\n{'='*55}")
    if passed:
        p(f"  VERDICT: ATD VALIDATED (rho={spk_rho:.3f}, p={spk_p:.4f})")
        p(f"  Proceed to paper writing!")
    elif spk_rho > 0.40:
        p(f"  VERDICT: ATD WEAK (rho={spk_rho:.3f})")
        p(f"  Try other layers before giving up.")
        p(f"  Diagnosis:")
        p(f"    - Check severity mapping: Severe=2, Mild=1, Control=0")
        p(f"    - Check PCA projection consistency")
        p(f"    - Try layers 6, 12, 15 instead of 9")
    else:
        p(f"  VERDICT: ATD FAILED (rho={spk_rho:.3f})")
        p(f"  Diagnosis:")
        p(f"    - Verify severity labels match TORGO documentation")
        p(f"    - Ensure control speakers excluded from references")
        p(f"    - Check PCA model is from reference, not refitted")
        p(f"    - Try computing without normalization")
        p(f"    - Try different distance metric (cosine vs euclidean)")
    p("=" * 55)

    # Per-speaker detail
    p(f"\n  Per-speaker ATD detail:")
    p(f"  {'Speaker':<10} {'Severity':>8} {'Mean ATD':>10} {'Std ATD':>10} {'N':>6}")
    p(f"  {'-'*48}")
    for _, r in speaker_atd_df.sort_values("severity_score").iterrows():
        p(f"  {r['speaker_id']:<10} {r['severity_score']:>8} {r['mean_atd']:>10.4f} {r['std_atd']:>10.4f} {r['n_utterances']:>6}")

    # Save report
    report_path = project_root / output_dir / "atd_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    report_txt_path = project_root / output_dir / "logs" / "phase4_validation.txt"
    report_txt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Report saved: {report_path}")
    print(f"  Text report: {report_txt_path}")

    return report
