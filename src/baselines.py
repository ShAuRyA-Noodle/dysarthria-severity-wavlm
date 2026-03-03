"""Baseline methods for dysarthria severity estimation (e.g., acoustic features, x-vectors)."""

import logging
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

logger = logging.getLogger(__name__)

RANDOM_SEED = 42

TORGO_SEVERITY = {
    "F01": 2, "F03": 1, "F04": 2,
    "M01": 2, "M02": 2, "M03": 1, "M04": 2, "M05": 1,
    "FC01": 0, "FC02": 0, "FC03": 0,
    "MC01": 0, "MC02": 0, "MC03": 0, "MC04": 0,
}

LAYER_POSITION_MAP = {1: 0, 3: 1, 6: 2, 9: 3, 12: 4, 15: 5, 18: 6, 21: 7, 23: 8}


def extract_mfcc_features(audio_path: str, sr: int = 16000) -> Optional[np.ndarray]:
    """
    Extract 58-dim acoustic feature vector from audio file.

    Features:
        - 13 MFCCs mean + 13 delta mean + 13 delta-delta mean = 39
        - 13 MFCCs std = 13
        - F0 mean, std = 2
        - Energy mean, std = 2
        - ZCR mean = 1
        - Duration proxy = 1
        Total = 58

    Args:
        audio_path: Path to WAV file.
        sr: Sample rate.

    Returns:
        Feature vector of shape (58,) or None on failure.
    """
    try:
        audio_path_str = str(audio_path)
        y, sr = librosa.load(audio_path_str, sr=sr)
        if len(y) < 512:
            return None

        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # (13, T)
        delta = librosa.feature.delta(mfccs)
        delta2 = librosa.feature.delta(mfccs, order=2)

        mfcc_mean = mfccs.mean(axis=1)      # (13,)
        delta_mean = delta.mean(axis=1)      # (13,)
        delta2_mean = delta2.mean(axis=1)    # (13,)
        mfcc_std = mfccs.std(axis=1)         # (13,)

        # F0
        f0, voiced, _ = librosa.pyin(y, fmin=50, fmax=500, sr=sr)
        f0_valid = f0[~np.isnan(f0)]
        f0_mean = f0_valid.mean() if len(f0_valid) > 0 else 0
        f0_std = f0_valid.std() if len(f0_valid) > 0 else 0

        # Energy (RMS)
        rms = librosa.feature.rms(y=y).flatten()
        energy_mean = rms.mean()
        energy_std = rms.std()

        # ZCR
        zcr = librosa.feature.zero_crossing_rate(y).flatten()
        zcr_mean = zcr.mean()

        # Duration / voiced frames proxy
        n_voiced = np.sum(voiced) if voiced is not None else 1
        duration = len(y) / sr
        rate_proxy = duration / max(n_voiced, 1)

        features = np.concatenate([
            mfcc_mean, delta_mean, delta2_mean, mfcc_std,
            [f0_mean, f0_std, energy_mean, energy_std, zcr_mean, rate_proxy],
        ])

        return features.astype(np.float32)

    except Exception as e:
        logger.debug(f"MFCC extraction failed for {audio_path}: {e}")
        return None


def train_mfcc_svm(
    train_speakers: list[str],
    metadata_df: pd.DataFrame,
    project_root: Path,
) -> dict:
    """
    Baseline 1: MFCC + SVR with RBF kernel.

    Args:
        train_speakers: List of speaker IDs for training.
        metadata_df: Full metadata DataFrame.
        project_root: Project root for audio paths.

    Returns:
        Dict with trained model, scaler, and metadata.
    """
    train_mask = metadata_df["speaker_id"].isin(train_speakers)
    train_df = metadata_df[train_mask]

    features, labels = [], []
    for _, row in train_df.iterrows():
        fpath = row["filepath_processed"]
        if pd.isna(fpath):
            continue
        audio_path = project_root / str(fpath)
        feat = extract_mfcc_features(str(audio_path))
        if feat is not None:
            features.append(feat)
            labels.append(TORGO_SEVERITY.get(row["speaker_id"], -1))

    X = np.stack(features)
    y = np.array(labels, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = SVR(kernel="rbf", C=10.0, gamma="scale")
    model.fit(X_scaled, y)

    return {"model": model, "scaler": scaler, "method": "MFCC + SVR"}


def predict_mfcc_svm(
    model_dict: dict,
    test_speakers: list[str],
    metadata_df: pd.DataFrame,
    project_root: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Predict severity using MFCC + SVR for test speakers.

    Returns:
        Tuple of (y_true, y_pred, speaker_ids).
    """
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    test_mask = metadata_df["speaker_id"].isin(test_speakers)
    test_df = metadata_df[test_mask]

    # Speaker-level: average features per speaker
    speaker_features = {}
    for _, row in test_df.iterrows():
        fpath = row["filepath_processed"]
        if pd.isna(fpath):
            continue
        spk = row["speaker_id"]
        audio_path = project_root / str(fpath)
        feat = extract_mfcc_features(str(audio_path))
        if feat is not None:
            speaker_features.setdefault(spk, []).append(feat)

    y_true, y_pred, spk_ids = [], [], []
    for spk, feats in speaker_features.items():
        X_spk = scaler.transform(np.stack(feats))
        preds = model.predict(X_spk)
        y_pred.append(preds.mean())
        y_true.append(TORGO_SEVERITY.get(spk, -1))
        spk_ids.append(spk)

    return np.array(y_true), np.array(y_pred), spk_ids


def extract_wavlm_mean_pool(
    utterance_id: str,
    ssl_index: dict,
    layer_idx: int = 9,
    project_root: Optional[Path] = None,
) -> Optional[np.ndarray]:
    """
    Mean pool WavLM layer embeddings to single vector (1024,).

    Args:
        utterance_id: Utterance ID.
        ssl_index: SSL index dict.
        layer_idx: Transformer layer number.
        project_root: Project root.

    Returns:
        Mean-pooled vector of shape (1024,) or None.
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent

    entry = ssl_index.get(utterance_id)
    if entry is None or entry.get("wavlm_large") is None:
        return None

    emb_path = project_root / str(entry["wavlm_large"])
    if not emb_path.exists():
        return None

    emb = np.load(str(emb_path))  # (9, T, 1024)
    pos = LAYER_POSITION_MAP.get(layer_idx, 3)
    return emb[pos].mean(axis=0)  # (1024,)


def train_ssl_ridge(
    train_speakers: list[str],
    metadata_df: pd.DataFrame,
    ssl_index: dict,
    project_root: Path,
    layer_idx: int = 9,
) -> dict:
    """
    Baseline 2: WavLM Mean Pool + Ridge regression.

    Args:
        train_speakers: Speaker IDs for training.
        metadata_df: Metadata DataFrame.
        ssl_index: SSL index.
        project_root: Project root.
        layer_idx: Layer to use.

    Returns:
        Dict with trained model, scaler.
    """
    train_mask = metadata_df["speaker_id"].isin(train_speakers)
    train_df = metadata_df[train_mask]

    # Speaker-level features
    speaker_features = {}
    for _, row in train_df.iterrows():
        spk = row["speaker_id"]
        feat = extract_wavlm_mean_pool(row["utterance_id"], ssl_index, layer_idx, project_root)
        if feat is not None:
            speaker_features.setdefault(spk, []).append(feat)

    X, y = [], []
    for spk, feats in speaker_features.items():
        X.append(np.mean(feats, axis=0))
        y.append(TORGO_SEVERITY.get(spk, -1))

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_scaled, y)

    return {"model": model, "scaler": scaler, "method": "WavLM Pool + Ridge"}


def predict_ssl_ridge(
    model_dict: dict,
    test_speakers: list[str],
    metadata_df: pd.DataFrame,
    ssl_index: dict,
    project_root: Path,
    layer_idx: int = 9,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict with WavLM Pool + Ridge."""
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    test_mask = metadata_df["speaker_id"].isin(test_speakers)
    test_df = metadata_df[test_mask]

    speaker_features = {}
    for _, row in test_df.iterrows():
        spk = row["speaker_id"]
        feat = extract_wavlm_mean_pool(row["utterance_id"], ssl_index, layer_idx, project_root)
        if feat is not None:
            speaker_features.setdefault(spk, []).append(feat)

    y_true, y_pred, spk_ids = [], [], []
    for spk, feats in speaker_features.items():
        X_spk = scaler.transform([np.mean(feats, axis=0)])
        y_pred.append(float(model.predict(X_spk)[0]))
        y_true.append(TORGO_SEVERITY.get(spk, -1))
        spk_ids.append(spk)

    return np.array(y_true), np.array(y_pred), spk_ids


def get_speaker_atd_features(
    speaker_id: str,
    atd_df: pd.DataFrame,
) -> Optional[np.ndarray]:
    """
    Get ATD features for a speaker: [mean, std, median].

    Args:
        speaker_id: Speaker ID.
        atd_df: ATD scores DataFrame.

    Returns:
        Feature vector (3,) or None.
    """
    valid = atd_df[(atd_df["speaker_id"] == speaker_id) & atd_df["atd_score"].notna()]
    if len(valid) == 0:
        return None
    return np.array([valid["atd_score"].mean(), valid["atd_score"].std(), valid["atd_score"].median()])


def train_atd_ridge(
    train_speakers: list[str],
    atd_df: pd.DataFrame,
) -> dict:
    """
    Baseline 3: ATD + Ridge (simple, interpretable).

    Features: [mean_atd, std_atd, median_atd] per speaker.

    Args:
        train_speakers: Training speaker IDs.
        atd_df: UTterance-level ATD scores.

    Returns:
        Trained model dict.
    """
    X, y = [], []
    for spk in train_speakers:
        feat = get_speaker_atd_features(spk, atd_df)
        if feat is not None:
            X.append(feat)
            y.append(TORGO_SEVERITY.get(spk, -1))

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    model.fit(X_scaled, y)

    return {"model": model, "scaler": scaler, "method": "ATD-L9 + Ridge"}


def predict_atd_ridge(
    model_dict: dict,
    test_speakers: list[str],
    atd_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict with ATD + Ridge."""
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    y_true, y_pred, spk_ids = [], [], []
    for spk in test_speakers:
        feat = get_speaker_atd_features(spk, atd_df)
        if feat is not None:
            X = scaler.transform([feat])
            y_pred.append(float(model.predict(X)[0]))
            y_true.append(TORGO_SEVERITY.get(spk, -1))
            spk_ids.append(spk)

    return np.array(y_true), np.array(y_pred), spk_ids


def get_speaker_multilayer_atd_features(
    speaker_id: str,
    atd_by_layer: dict[int, pd.DataFrame],
) -> Optional[np.ndarray]:
    """
    Get multi-layer ATD features: [mean, std, median] × 9 layers = 27 dims.

    Args:
        speaker_id: Speaker ID.
        atd_by_layer: Dict mapping layer → ATD DataFrame.

    Returns:
        Feature vector (27,) or None.
    """
    feats = []
    for layer in sorted(atd_by_layer.keys()):
        f = get_speaker_atd_features(speaker_id, atd_by_layer[layer])
        if f is None:
            return None
        feats.append(f)
    return np.concatenate(feats)


def train_atd_multilayer_ridge(
    train_speakers: list[str],
    atd_by_layer: dict[int, pd.DataFrame],
) -> dict:
    """
    Baseline 4: ATD Multi-layer + Ridge (full method).

    Features: ATD [mean, std, median] at ALL 9 layers = 27 features.

    Args:
        train_speakers: Training speaker IDs.
        atd_by_layer: Dict mapping layer → ATD DataFrame.

    Returns:
        Trained model dict.
    """
    X, y = [], []
    for spk in train_speakers:
        feat = get_speaker_multilayer_atd_features(spk, atd_by_layer)
        if feat is not None:
            X.append(feat)
            y.append(TORGO_SEVERITY.get(spk, -1))

    X = np.stack(X)
    y = np.array(y, dtype=np.float32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = Ridge(alpha=10.0, random_state=RANDOM_SEED)
    model.fit(X_scaled, y)

    return {"model": model, "scaler": scaler, "method": "ATD-All + Ridge"}


def predict_atd_multilayer_ridge(
    model_dict: dict,
    test_speakers: list[str],
    atd_by_layer: dict[int, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predict with ATD Multi-layer + Ridge."""
    model = model_dict["model"]
    scaler = model_dict["scaler"]

    y_true, y_pred, spk_ids = [], [], []
    for spk in test_speakers:
        feat = get_speaker_multilayer_atd_features(spk, atd_by_layer)
        if feat is not None:
            X = scaler.transform([feat])
            y_pred.append(float(model.predict(X)[0]))
            y_true.append(TORGO_SEVERITY.get(spk, -1))
            spk_ids.append(spk)

    return np.array(y_true), np.array(y_pred), spk_ids


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
) -> dict:
    """
    Compute all regression evaluation metrics.

    Metrics:
        - RMSE (primary — lower is better)
        - MAE
        - Spearman rho
        - Pearson r
        - Within-1-level accuracy

    Args:
        y_true: Ground truth severity scores.
        y_pred: Predicted severity scores.
        model_name: Name for display.

    Returns:
        Dict with all metrics.
    """
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    if len(y_true) >= 3:
        rho, rho_p = spearmanr(y_true, y_pred)
        r, r_p = pearsonr(y_true, y_pred)
    else:
        rho, rho_p = 0.0, 1.0
        r, r_p = 0.0, 1.0

    within_1 = float(np.mean(np.abs(y_true - y_pred) <= 1.0))

    result = {
        "method": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "spearman_rho": round(float(rho), 4),
        "pearson_r": round(float(r), 4),
        "within_1_acc": round(within_1, 4),
    }

    return result
