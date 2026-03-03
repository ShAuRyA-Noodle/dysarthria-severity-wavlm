"""
Experiment 08 — Run all baselines with LOSO cross-validation.

Evaluates 4 methods:
1. MFCC + SVR
2. WavLM Pool + Ridge
3. ATD-L9 + Ridge (ours simple)
4. ATD-All + Ridge (ours full)

Uses LOSO (leave-one-speaker-out) for primary evaluation,
and Split B (severity generalization) for secondary.

Usage:
    python experiments/08_baselines.py
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines import (
    TORGO_SEVERITY,
    evaluate_regression_model,
    extract_mfcc_features,
    get_speaker_atd_features,
    get_speaker_multilayer_atd_features,
    predict_atd_multilayer_ridge,
    predict_atd_ridge,
    predict_mfcc_svm,
    predict_ssl_ridge,
    train_atd_multilayer_ridge,
    train_atd_ridge,
    train_mfcc_svm,
    train_ssl_ridge,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            PROJECT_ROOT / "results" / "logs" / "baselines.log", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger(__name__)

TARGET_LAYERS = [1, 3, 6, 9, 12, 15, 18, 21, 23]
ALL_SPEAKERS = sorted(TORGO_SEVERITY.keys())


def run_loso_evaluation(
    metadata_df: pd.DataFrame,
    ssl_index: dict,
    atd_df: pd.DataFrame,
    atd_by_layer: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """
    LOSO evaluation: leave one dysarthric speaker out, train on rest.

    Args:
        metadata_df: Full metadata DataFrame.
        ssl_index: SSL index dict.
        atd_df: Layer-9 ATD scores.
        atd_by_layer: ATD scores per layer.

    Returns:
        Results DataFrame.
    """
    dysarthric = [s for s, sev in TORGO_SEVERITY.items() if sev > 0]
    controls = [s for s, sev in TORGO_SEVERITY.items() if sev == 0]

    all_results = {m: {"y_true": [], "y_pred": []} for m in [
        "MFCC + SVR", "WavLM Pool + Ridge", "ATD-L9 + Ridge", "ATD-All + Ridge"
    ]}

    print(f"\n{'='*60}")
    print(f"  LOSO Cross-Validation ({len(dysarthric)} folds)")
    print(f"{'='*60}\n")

    for held_out in dysarthric:
        train_speakers = [s for s in ALL_SPEAKERS if s != held_out]
        test_speakers = [held_out]
        sev = TORGO_SEVERITY[held_out]

        print(f"  Fold: held_out={held_out} (sev={sev})")

        # Method 1: MFCC + SVR
        try:
            m1 = train_mfcc_svm(train_speakers, metadata_df, PROJECT_ROOT)
            yt1, yp1, _ = predict_mfcc_svm(m1, test_speakers, metadata_df, PROJECT_ROOT)
            all_results["MFCC + SVR"]["y_true"].extend(yt1)
            all_results["MFCC + SVR"]["y_pred"].extend(yp1)
        except Exception as e:
            logger.error(f"MFCC+SVR failed for {held_out}: {e}")

        # Method 2: WavLM Pool + Ridge
        try:
            m2 = train_ssl_ridge(train_speakers, metadata_df, ssl_index, PROJECT_ROOT)
            yt2, yp2, _ = predict_ssl_ridge(m2, test_speakers, metadata_df, ssl_index, PROJECT_ROOT)
            all_results["WavLM Pool + Ridge"]["y_true"].extend(yt2)
            all_results["WavLM Pool + Ridge"]["y_pred"].extend(yp2)
        except Exception as e:
            logger.error(f"WavLM+Ridge failed for {held_out}: {e}")

        # Method 3: ATD-L9 + Ridge
        try:
            m3 = train_atd_ridge(train_speakers, atd_df)
            yt3, yp3, _ = predict_atd_ridge(m3, test_speakers, atd_df)
            all_results["ATD-L9 + Ridge"]["y_true"].extend(yt3)
            all_results["ATD-L9 + Ridge"]["y_pred"].extend(yp3)
        except Exception as e:
            logger.error(f"ATD-L9+Ridge failed for {held_out}: {e}")

        # Method 4: ATD-All + Ridge
        try:
            m4 = train_atd_multilayer_ridge(train_speakers, atd_by_layer)
            yt4, yp4, _ = predict_atd_multilayer_ridge(m4, test_speakers, atd_by_layer)
            all_results["ATD-All + Ridge"]["y_true"].extend(yt4)
            all_results["ATD-All + Ridge"]["y_pred"].extend(yp4)
        except Exception as e:
            logger.error(f"ATD-All+Ridge failed for {held_out}: {e}")

    # Evaluate all methods
    table_rows = []
    for method, data in all_results.items():
        if len(data["y_true"]) > 0:
            yt = np.array(data["y_true"])
            yp = np.array(data["y_pred"])
            metrics = evaluate_regression_model(yt, yp, method)
            table_rows.append(metrics)

    return pd.DataFrame(table_rows)


def run_split_b_evaluation(
    metadata_df: pd.DataFrame,
    ssl_index: dict,
    atd_df: pd.DataFrame,
    atd_by_layer: dict[int, pd.DataFrame],
    splits: dict,
) -> pd.DataFrame:
    """
    Split B: train on control+mild, test on severe only.

    Args:
        metadata_df: Full metadata.
        ssl_index: SSL index.
        atd_df: ATD scores.
        atd_by_layer: ATD per layer.
        splits: Splits dict from splits.json.

    Returns:
        Results DataFrame.
    """
    split_b = splits.get("split_B", {})
    train_speakers = split_b.get("train_speakers", [])
    test_speakers = split_b.get("test_speakers", [])

    if not train_speakers or not test_speakers:
        # Fallback: train on control+mild, test on severe
        train_speakers = [s for s, sev in TORGO_SEVERITY.items() if sev <= 1]
        test_speakers = [s for s, sev in TORGO_SEVERITY.items() if sev == 2]

    print(f"\n{'='*60}")
    print(f"  Split B: Severity Generalization")
    print(f"  Train: {train_speakers}")
    print(f"  Test: {test_speakers}")
    print(f"{'='*60}\n")

    table_rows = []

    # Method 1: MFCC + SVR
    try:
        m1 = train_mfcc_svm(train_speakers, metadata_df, PROJECT_ROOT)
        yt1, yp1, _ = predict_mfcc_svm(m1, test_speakers, metadata_df, PROJECT_ROOT)
        table_rows.append(evaluate_regression_model(yt1, yp1, "MFCC + SVR"))
    except Exception as e:
        logger.error(f"Split B MFCC+SVR failed: {e}")

    # Method 2: WavLM Pool + Ridge
    try:
        m2 = train_ssl_ridge(train_speakers, metadata_df, ssl_index, PROJECT_ROOT)
        yt2, yp2, _ = predict_ssl_ridge(m2, test_speakers, metadata_df, ssl_index, PROJECT_ROOT)
        table_rows.append(evaluate_regression_model(yt2, yp2, "WavLM Pool + Ridge"))
    except Exception as e:
        logger.error(f"Split B WavLM+Ridge failed: {e}")

    # Method 3: ATD-L9 + Ridge
    try:
        m3 = train_atd_ridge(train_speakers, atd_df)
        yt3, yp3, _ = predict_atd_ridge(m3, test_speakers, atd_df)
        table_rows.append(evaluate_regression_model(yt3, yp3, "ATD-L9 + Ridge"))
    except Exception as e:
        logger.error(f"Split B ATD-L9+Ridge failed: {e}")

    # Method 4: ATD-All + Ridge
    try:
        m4 = train_atd_multilayer_ridge(train_speakers, atd_by_layer)
        yt4, yp4, _ = predict_atd_multilayer_ridge(m4, test_speakers, atd_by_layer)
        table_rows.append(evaluate_regression_model(yt4, yp4, "ATD-All + Ridge"))
    except Exception as e:
        logger.error(f"Split B ATD-All+Ridge failed: {e}")

    return pd.DataFrame(table_rows)


def main() -> None:
    """Run all baselines."""
    print(f"{'='*60}")
    print(f"  Baseline Evaluation — Phase 5")
    print(f"{'='*60}\n")

    # Load data
    metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    metadata_df['filepath_processed'] = metadata_df['filepath_processed'].astype(str)
    print(f"Metadata: {len(metadata_df)} utterances")

    with open(PROJECT_ROOT / "data/embeddings/ssl_index.json") as f:
        ssl_index = json.load(f)

    with open(PROJECT_ROOT / "data/splits/splits.json") as f:
        splits = json.load(f)

    # Load ATD scores for layer 9 (primary)
    atd_path = PROJECT_ROOT / "results/atd_scores.csv"
    atd_df = pd.read_csv(atd_path)
    print(f"ATD scores (L9): {len(atd_df)} utterances")

    # Load multi-layer ATD scores (if available from layer analysis)
    atd_by_layer = {}
    for layer in TARGET_LAYERS:
        lpath = PROJECT_ROOT / f"results/atd_scores_layer{layer}.csv"
        if lpath.exists():
            atd_by_layer[layer] = pd.read_csv(lpath)
        else:
            # Use layer 9 as fallback
            atd_by_layer[layer] = atd_df
    print(f"Multi-layer ATD: {len(atd_by_layer)} layers loaded")

    # Run LOSO
    loso_results = run_loso_evaluation(metadata_df, ssl_index, atd_df, atd_by_layer)

    # Run Split B
    splitb_results = run_split_b_evaluation(metadata_df, ssl_index, atd_df, atd_by_layer, splits)

    # Combine into Table 1
    if not splitb_results.empty:
        splitb_spearman = splitb_results.set_index("method")["spearman_rho"]
        loso_results["split_b_spearman"] = loso_results["method"].map(
            lambda m: splitb_spearman.get(m, np.nan)
        )
    else:
        loso_results["split_b_spearman"] = np.nan

    # Save Table 1
    table_path = PROJECT_ROOT / "results/tables/table1_main_results.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    loso_results.to_csv(table_path, index=False)

    # Print Table 1
    print(f"\n{'='*60}")
    print(f"  TABLE 1: Main Results (LOSO + Split B)")
    print(f"{'='*60}")
    print(loso_results.to_string(index=False))
    print(f"\n  Saved to: {table_path}")
    print(f"{'='*60}")

    # Also save verification output
    lines = [
        "=" * 60,
        "  Phase 5 Verification — Table 1",
        "=" * 60,
        "",
        loso_results.to_string(index=False),
        "",
    ]

    # Check if ATD beats MFCC
    atd_rows = loso_results[loso_results["method"].str.contains("ATD")]
    mfcc_rows = loso_results[loso_results["method"] == "MFCC + SVR"]
    if not atd_rows.empty and not mfcc_rows.empty:
        atd_best = atd_rows["spearman_rho"].max()
        mfcc_rho = mfcc_rows["spearman_rho"].values[0]
        lines.append(f"ATD best Spearman rho: {atd_best:.4f}")
        lines.append(f"MFCC Spearman rho: {mfcc_rho:.4f}")
        lines.append(f"ATD wins: {atd_best > mfcc_rho}")

    report_path = PROJECT_ROOT / "results/logs/phase5_baselines.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
