"""
Experiment 42: Fine-tuned-head SSL baseline on TORGO with LOSO evaluation.

Addresses reviewer concern that the paper's only baseline was MFCC + SVR (2015).
Builds a frozen WavLM-Large layer-9 mean-pool feature + downstream regression
head, evaluated under leave-one-speaker-out, mirroring the protocol used by
Hernandez et al. (2020) and Schu et al. (2023) for SSL-based dysarthria severity
estimation. The backbone is frozen — full backbone fine-tune on TORGO (n=15)
would be degenerate; the standard SSL-baseline reproduction is frozen feature +
trainable head.

Outputs: results/finetuned_ssl_baseline.json

Run with: /c/Users/Dell/.conda/envs/dysarthria/python.exe experiments/42_finetuned_ssl_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.seed_utils import set_global_seed  # noqa: E402

METADATA_CSV = PROJECT_ROOT / "data" / "processed" / "metadata" / "metadata_unified.csv"
EMBED_ROOT = PROJECT_ROOT / "data" / "embeddings" / "wavlm_large"
LAYER_INDICES_JSON = EMBED_ROOT / "layer_indices.json"
SPEAKER_ATD_CSV = PROJECT_ROOT / "results" / "speaker_atd.csv"

FEATURES_DIR = PROJECT_ROOT / "data" / "processed" / "features"
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
CACHE_NPY = FEATURES_DIR / "torgo_utt_mean_l9.npy"
CACHE_META = FEATURES_DIR / "torgo_utt_mean_l9_meta.csv"

# MFCC baseline cache (58-d acoustic vector per utterance; see src/baselines.py).
MFCC_CACHE_NPY = FEATURES_DIR / "torgo_utt_mfcc58.npy"
MFCC_CACHE_META = FEATURES_DIR / "torgo_utt_mfcc58_meta.csv"
AUDIO_ROOT = PROJECT_ROOT / "data" / "processed" / "audio_16khz"

OUTPUT_JSON = PROJECT_ROOT / "results" / "finetuned_ssl_baseline.json"

TARGET_LAYER = 9
RANDOM_SEED = 42


def _atomic_dump_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def _write_error_report(message: str, tb: str) -> None:
    _atomic_dump_json(
        {
            "error": message,
            "traceback": tb,
            "wavlm_l9_meanpool_ridge_loso": None,
            "wavlm_l9_meanpool_linsvr_loso": None,
            "mfcc_svr_rbf_loso": None,
            "atd_zero_shot_for_reference": None,
            "method_notes": (
                "Run failed before metrics could be computed. See 'error' and "
                "'traceback' fields."
            ),
        },
        OUTPUT_JSON,
    )


def load_torgo_metadata() -> pd.DataFrame:
    df = pd.read_csv(METADATA_CSV)
    torgo = df[df["dataset"] == "torgo"].copy()
    torgo = torgo.reset_index(drop=True)
    return torgo


def build_or_load_features(torgo: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Mean-pool layer-9 WavLM-Large embeddings for every TORGO utterance.

    Reuses cached .npy if shape/index match the current TORGO metadata.
    Falls back to streaming the per-utterance multi-layer .npy files and
    extracting layer 9. If a per-utterance file is missing, the row is dropped.
    """
    # Resolve layer-9 index inside the stored (n_layers, T, 1024) tensor.
    with open(LAYER_INDICES_JSON, "r", encoding="utf-8") as f:
        target_layers = json.load(f)["target_layers"]
    if TARGET_LAYER not in target_layers:
        raise RuntimeError(
            f"Cached extractor did not include layer {TARGET_LAYER}. "
            f"Available: {target_layers}"
        )
    layer_axis_idx = target_layers.index(TARGET_LAYER)

    if CACHE_NPY.exists() and CACHE_META.exists():
        X_cached = np.load(CACHE_NPY)
        meta_cached = pd.read_csv(CACHE_META)
        if (
            len(meta_cached) == X_cached.shape[0]
            and X_cached.shape[1] == 1024
            and set(meta_cached["utterance_id"]).issubset(set(torgo["utterance_id"]))
        ):
            print(
                f"[cache] reusing cached features: X={X_cached.shape}, "
                f"meta={len(meta_cached)} rows from {CACHE_NPY}"
            )
            return X_cached.astype(np.float32, copy=False), meta_cached

    # Build fresh.
    rows = []
    feats = []
    n_missing = 0
    t0 = time.time()
    for i, r in enumerate(torgo.itertuples(index=False)):
        utt_id = r.utterance_id
        spk = r.speaker_id
        npy_path = EMBED_ROOT / spk / f"{utt_id}.npy"
        if not npy_path.exists():
            n_missing += 1
            continue
        try:
            stacked = np.load(npy_path)  # (n_layers, T, 1024)
            if stacked.ndim != 3 or stacked.shape[0] <= layer_axis_idx:
                n_missing += 1
                continue
            layer = stacked[layer_axis_idx]  # (T, 1024)
            pooled = layer.mean(axis=0).astype(np.float32)  # (1024,)
            feats.append(pooled)
            rows.append(
                {
                    "utterance_id": utt_id,
                    "speaker_id": spk,
                    "severity_score": int(r.severity_score),
                    "is_control": bool(r.is_control),
                }
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed on {utt_id}: {e}")
            n_missing += 1
            continue
        if (i + 1) % 1000 == 0:
            print(
                f"  pooled {i + 1}/{len(torgo)} "
                f"(elapsed {time.time() - t0:.1f}s, missing={n_missing})"
            )

    X = np.stack(feats, axis=0).astype(np.float32)
    meta = pd.DataFrame(rows)
    np.save(CACHE_NPY, X)
    meta.to_csv(CACHE_META, index=False)
    print(
        f"[cache] wrote {CACHE_NPY} shape={X.shape}, meta rows={len(meta)}, "
        f"missing={n_missing}"
    )
    return X, meta


def build_or_load_mfcc_features(torgo: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """Extract the 58-d MFCC/acoustic feature vector for every TORGO utterance.

    Uses src.baselines.extract_mfcc_features so the honest MFCC baseline shares
    the *identical* front-end as the paper's strawmanned MFCC+SVR baseline.
    Audio paths are reconstructed from <speaker>/<utterance_id>.wav under
    data/processed/audio_16khz (metadata 'filepath_processed' holds stale
    absolute paths from a different machine). Cached to disk for reuse.
    """
    from src.baselines import extract_mfcc_features

    if MFCC_CACHE_NPY.exists() and MFCC_CACHE_META.exists():
        X_cached = np.load(MFCC_CACHE_NPY)
        meta_cached = pd.read_csv(MFCC_CACHE_META)
        if (
            len(meta_cached) == X_cached.shape[0]
            and X_cached.shape[1] == 58
            and set(meta_cached["utterance_id"]).issubset(set(torgo["utterance_id"]))
        ):
            print(
                f"[cache] reusing MFCC features: X={X_cached.shape}, "
                f"meta={len(meta_cached)} rows from {MFCC_CACHE_NPY}"
            )
            return X_cached.astype(np.float32, copy=False), meta_cached

    rows = []
    feats = []
    n_missing = 0
    t0 = time.time()
    n_total = len(torgo)
    for i, r in enumerate(torgo.itertuples(index=False)):
        utt_id = r.utterance_id
        spk = r.speaker_id
        wav_path = AUDIO_ROOT / spk / f"{utt_id}.wav"
        if not wav_path.exists():
            n_missing += 1
            continue
        feat = extract_mfcc_features(str(wav_path))
        if feat is None or feat.shape[0] != 58:
            n_missing += 1
            continue
        feats.append(feat.astype(np.float32))
        rows.append(
            {
                "utterance_id": utt_id,
                "speaker_id": spk,
                "severity_score": int(r.severity_score),
                "is_control": bool(r.is_control),
            }
        )
        if (i + 1) % 500 == 0:
            print(
                f"  mfcc {i + 1}/{n_total} "
                f"(elapsed {time.time() - t0:.1f}s, missing={n_missing})"
            )

    X = np.stack(feats, axis=0).astype(np.float32)
    meta = pd.DataFrame(rows)
    # Guard: replace any non-finite values (e.g. degenerate F0) with 0 so the
    # scaler/SVR are well-defined. Rare; reported via the warn count.
    n_nonfinite = int(np.sum(~np.isfinite(X)))
    if n_nonfinite:
        print(f"  [warn] {n_nonfinite} non-finite MFCC values -> set to 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    np.save(MFCC_CACHE_NPY, X)
    meta.to_csv(MFCC_CACHE_META, index=False)
    print(
        f"[cache] wrote {MFCC_CACHE_NPY} shape={X.shape}, meta rows={len(meta)}, "
        f"missing={n_missing}"
    )
    return X, meta


def run_loso_svr_rbf(X: np.ndarray, meta: pd.DataFrame) -> dict:
    """LOSO RBF-SVR with nested GroupKFold CV over (C, gamma).

    This is the *fairly tuned* MFCC+SVR baseline: same RBF-SVR front-end as the
    paper's MFCC+SVR (src/baselines.py train_mfcc_svm), but with the
    hyperparameters honestly selected inside each LOSO fold rather than fixed at
    C=10, gamma='scale'. Scaler is fit on the inner-train split only (nested,
    leakage-free) and on the LOSO-train fold only for the final refit.
    """
    from sklearn.svm import SVR
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y_utt = meta["severity_score"].to_numpy(dtype=float)
    speakers = meta["speaker_id"].to_numpy()
    speaker_list = sorted(np.unique(speakers).tolist())

    svr_Cs = [0.1, 1.0, 10.0, 100.0]
    svr_gammas = ["scale", 0.001, 0.01, 0.1]
    param_grid = [{"C": c, "gamma": g} for c in svr_Cs for g in svr_gammas]

    def make_model(p):
        return SVR(kernel="rbf", C=p["C"], gamma=p["gamma"])

    speaker_true: list[float] = []
    speaker_pred: list[float] = []
    speaker_isctrl: list[bool] = []
    speaker_order: list[str] = []
    selected_hparams: list[dict] = []

    for held_spk in speaker_list:
        train_mask = speakers != held_spk
        test_mask = speakers == held_spk
        X_tr_raw = X[train_mask]
        y_tr = y_utt[train_mask]
        groups_tr = speakers[train_mask]
        X_te_raw = X[test_mask]

        n_train_speakers = len(np.unique(groups_tr))
        n_splits = min(5, n_train_speakers)
        gkf = GroupKFold(n_splits=n_splits)

        best_p = None
        best_score = np.inf
        for p in param_grid:
            cv_errs = []
            for tr_idx, va_idx in gkf.split(X_tr_raw, y_tr, groups=groups_tr):
                inner_scaler = StandardScaler()
                X_in_tr = inner_scaler.fit_transform(X_tr_raw[tr_idx])
                X_in_va = inner_scaler.transform(X_tr_raw[va_idx])
                m = make_model(p)
                m.fit(X_in_tr, y_tr[tr_idx])
                pred = m.predict(X_in_va)
                cv_errs.append(float(np.mean((pred - y_tr[va_idx]) ** 2)))
            mean_err = float(np.mean(cv_errs))
            if mean_err < best_score:
                best_score = mean_err
                best_p = p

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
        final_model = make_model(best_p)
        final_model.fit(X_tr, y_tr)
        utt_pred = final_model.predict(X_te)

        spk_pred = float(np.mean(utt_pred))
        spk_true = float(np.mean(y_utt[test_mask]))
        spk_isctrl = bool(meta.loc[test_mask, "is_control"].iloc[0])

        speaker_true.append(spk_true)
        speaker_pred.append(spk_pred)
        speaker_isctrl.append(spk_isctrl)
        speaker_order.append(held_spk)
        selected_hparams.append(
            {
                "held_out_speaker": held_spk,
                "selected": {"C": float(best_p["C"]), "gamma": best_p["gamma"]},
                "inner_cv_mse": best_score,
                "n_test_utts": int(test_mask.sum()),
            }
        )

        print(
            f"  [mfcc_svr_rbf] held-out {held_spk}: "
            f"true={spk_true:.0f} pred={spk_pred:.3f} "
            f"best_hp={best_p} n_te_utts={int(test_mask.sum())}"
        )

    metrics = compute_metrics(
        np.array(speaker_true), np.array(speaker_pred), np.array(speaker_isctrl)
    )
    metrics["speakers"] = speaker_order
    metrics["y_true_speaker"] = [float(v) for v in speaker_true]
    metrics["y_pred_speaker"] = [float(v) for v in speaker_pred]
    metrics["per_fold_selected_hyperparams"] = selected_hparams
    return metrics


def compute_metrics(
    y_true_speaker: np.ndarray,
    y_pred_speaker: np.ndarray,
    is_control_speaker: np.ndarray,
) -> dict:
    """Speaker-level regression metrics + rank-accuracy (dys above ctrl)."""
    from scipy.stats import spearmanr, pearsonr

    y_true = np.asarray(y_true_speaker, dtype=float)
    y_pred = np.asarray(y_pred_speaker, dtype=float)

    rho, rho_p = spearmanr(y_true, y_pred)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        pear_r = float("nan")
    else:
        pear_r, _ = pearsonr(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    # Rank accuracy: fraction of (dys, ctrl) pairs where pred_dys > pred_ctrl.
    is_ctrl = np.asarray(is_control_speaker, dtype=bool)
    ctrl_preds = y_pred[is_ctrl]
    dys_preds = y_pred[~is_ctrl]
    if len(ctrl_preds) == 0 or len(dys_preds) == 0:
        rank_acc = float("nan")
    else:
        wins = 0
        total = 0
        for d in dys_preds:
            for c in ctrl_preds:
                total += 1
                if d > c:
                    wins += 1
                elif d == c:
                    wins += 0.5
        rank_acc = wins / total

    return {
        "n": int(len(y_true)),
        "rho": float(rho),
        "rho_p": float(rho_p),
        "pearson_r": float(pear_r),
        "rmse": rmse,
        "mae": mae,
        "rank_accuracy_dys_over_ctrl": float(rank_acc),
    }


def run_loso_regression(
    X: np.ndarray,
    meta: pd.DataFrame,
    model_kind: str,
) -> dict:
    """Leave-one-speaker-out evaluation with internal 5-fold CV for hyperparams.

    model_kind: 'ridge' | 'linsvr'
    """
    from sklearn.linear_model import Ridge
    from sklearn.svm import LinearSVR
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    y_utt = meta["severity_score"].to_numpy(dtype=float)
    speakers = meta["speaker_id"].to_numpy()
    speaker_list = sorted(np.unique(speakers).tolist())

    speaker_true: list[float] = []
    speaker_pred: list[float] = []
    speaker_isctrl: list[bool] = []
    speaker_order: list[str] = []
    selected_hparams: list[dict] = []

    ridge_alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]
    svr_Cs = [0.01, 0.1, 1.0, 10.0]

    for held_spk in speaker_list:
        train_mask = speakers != held_spk
        test_mask = speakers == held_spk
        X_tr_raw = X[train_mask]
        y_tr = y_utt[train_mask]
        groups_tr = speakers[train_mask]
        X_te_raw = X[test_mask]

        # Internal 5-fold group CV to pick the hyperparam.
        # GroupKFold: cap n_splits at min(5, n_train_speakers).
        n_train_speakers = len(np.unique(groups_tr))
        n_splits = min(5, n_train_speakers)
        gkf = GroupKFold(n_splits=n_splits)

        if model_kind == "ridge":
            param_grid = [{"alpha": a} for a in ridge_alphas]

            def make_model(p):
                return Ridge(alpha=p["alpha"], random_state=RANDOM_SEED)

        elif model_kind == "linsvr":
            param_grid = [{"C": c} for c in svr_Cs]

            def make_model(p):
                # squared_epsilon_insensitive loss (L2) is far better-conditioned
                # than the default epsilon_insensitive (L1) for high-dimensional
                # standardized features (d=1024, n~8000): the default loss with
                # high C fails to converge in 20k liblinear iters, wasting compute
                # and yielding an under-fit solution. The squared loss converges
                # reliably at tol=1e-3 in a few thousand iters. This *strengthens*
                # the SVR baseline (better optimum), it does not weaken the search.
                return LinearSVR(
                    C=p["C"],
                    loss="squared_epsilon_insensitive",
                    random_state=RANDOM_SEED,
                    tol=1e-3,
                    max_iter=10000,
                    dual="auto",
                )

        else:
            raise ValueError(f"Unknown model_kind={model_kind}")

        # Fully-nested hyperparameter search: the StandardScaler is fit on the
        # inner-training split only (never on the inner-validation split), so the
        # hyperparameter selection itself is leakage-free. The held-out LOSO
        # speaker is excluded from every inner split by construction.
        best_p = None
        best_score = np.inf
        for p in param_grid:
            cv_errs = []
            for tr_idx, va_idx in gkf.split(X_tr_raw, y_tr, groups=groups_tr):
                inner_scaler = StandardScaler()
                X_in_tr = inner_scaler.fit_transform(X_tr_raw[tr_idx])
                X_in_va = inner_scaler.transform(X_tr_raw[va_idx])
                m = make_model(p)
                m.fit(X_in_tr, y_tr[tr_idx])
                pred = m.predict(X_in_va)
                cv_errs.append(float(np.mean((pred - y_tr[va_idx]) ** 2)))
            mean_err = float(np.mean(cv_errs))
            if mean_err < best_score:
                best_score = mean_err
                best_p = p

        # Refit on the full LOSO training fold with the best hyperparam.
        # Scaler is fit on the training fold only, then applied to the held-out
        # test speaker (no test leakage).
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_raw)
        X_te = scaler.transform(X_te_raw)
        final_model = make_model(best_p)
        final_model.fit(X_tr, y_tr)
        utt_pred = final_model.predict(X_te)

        # Aggregate to per-speaker prediction (mean across that speaker's utts).
        spk_pred = float(np.mean(utt_pred))
        spk_true = float(np.mean(y_utt[test_mask]))  # constant across utts
        spk_isctrl = bool(meta.loc[test_mask, "is_control"].iloc[0])

        speaker_true.append(spk_true)
        speaker_pred.append(spk_pred)
        speaker_isctrl.append(spk_isctrl)
        speaker_order.append(held_spk)
        selected_hparams.append(
            {
                "held_out_speaker": held_spk,
                "selected": {k: float(v) for k, v in best_p.items()},
                "inner_cv_mse": best_score,
                "n_test_utts": int(test_mask.sum()),
            }
        )

        print(
            f"  [{model_kind}] held-out {held_spk}: "
            f"true={spk_true:.0f} pred={spk_pred:.3f} "
            f"best_hp={best_p} n_te_utts={int(test_mask.sum())}"
        )

    metrics = compute_metrics(
        np.array(speaker_true), np.array(speaker_pred), np.array(speaker_isctrl)
    )
    metrics["speakers"] = speaker_order
    metrics["y_true_speaker"] = [float(v) for v in speaker_true]
    metrics["y_pred_speaker"] = [float(v) for v in speaker_pred]
    metrics["per_fold_selected_hyperparams"] = selected_hparams
    return metrics


def compute_atd_zero_shot_metrics() -> dict:
    """ATD is zero-shot (no training); LOSO is trivially the full-set metrics."""
    if not SPEAKER_ATD_CSV.exists():
        return {"error": f"speaker_atd.csv not found at {SPEAKER_ATD_CSV}"}
    atd = pd.read_csv(SPEAKER_ATD_CSV)
    # Filter to TORGO speakers only (just in case).
    torgo_speakers = {
        "F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05",
        "FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04",
    }
    atd = atd[atd["speaker_id"].isin(torgo_speakers)].copy()
    atd = atd.sort_values("speaker_id").reset_index(drop=True)

    y_true = atd["severity_score"].to_numpy(dtype=float)
    y_pred = atd["mean_atd"].to_numpy(dtype=float)
    is_ctrl = atd["is_control"].to_numpy(dtype=bool)

    metrics = compute_metrics(y_true, y_pred, is_ctrl)
    metrics["speakers"] = atd["speaker_id"].tolist()
    metrics["y_true_speaker"] = [float(v) for v in y_true]
    metrics["y_pred_speaker"] = [float(v) for v in y_pred]
    metrics["note"] = (
        "ATD is zero-shot (no training, no model). LOSO is equivalent to the "
        "full-set evaluation because each speaker's ATD is computed against "
        "reference embeddings from disjoint control speakers. Note that the "
        "ATD scale (mean reference-distance) is not in the {0,1,2} severity "
        "space, so RMSE/MAE are scale-mismatched and meaningful only for "
        "rho/rank-accuracy comparison."
    )
    return metrics


def main() -> int:
    set_global_seed(RANDOM_SEED)
    print("=" * 72)
    print(" Fine-tuned-head SSL baseline on TORGO (LOSO)")
    print("=" * 72)

    torgo = load_torgo_metadata()
    print(f"TORGO metadata rows: {len(torgo)} | speakers: {torgo['speaker_id'].nunique()}")

    X, meta = build_or_load_features(torgo)
    print(f"Feature matrix: X={X.shape}, meta rows={len(meta)}")
    print(f"Speakers in features: {sorted(meta['speaker_id'].unique())}")
    print(f"Per-speaker utt counts:\n{meta['speaker_id'].value_counts().to_dict()}")

    print("\n--- WavLM-pool Ridge LOSO ---")
    ridge_metrics = run_loso_regression(X, meta, model_kind="ridge")
    print(
        f"\nRidge LOSO speaker-level: rho={ridge_metrics['rho']:.3f} "
        f"(p={ridge_metrics['rho_p']:.4f}) | pearson={ridge_metrics['pearson_r']:.3f} | "
        f"rmse={ridge_metrics['rmse']:.3f} | mae={ridge_metrics['mae']:.3f} | "
        f"rank_acc={ridge_metrics['rank_accuracy_dys_over_ctrl']:.3f}"
    )

    print("\n--- WavLM-pool LinearSVR LOSO ---")
    svr_metrics = run_loso_regression(X, meta, model_kind="linsvr")
    print(
        f"\nLinearSVR LOSO speaker-level: rho={svr_metrics['rho']:.3f} "
        f"(p={svr_metrics['rho_p']:.4f}) | pearson={svr_metrics['pearson_r']:.3f} | "
        f"rmse={svr_metrics['rmse']:.3f} | mae={svr_metrics['mae']:.3f} | "
        f"rank_acc={svr_metrics['rank_accuracy_dys_over_ctrl']:.3f}"
    )

    print("\n--- MFCC+SVR (RBF) LOSO — fairly tuned ---")
    mfcc_X, mfcc_meta = build_or_load_mfcc_features(torgo)
    print(f"MFCC feature matrix: X={mfcc_X.shape}, meta rows={len(mfcc_meta)}")
    mfcc_svr_metrics = run_loso_svr_rbf(mfcc_X, mfcc_meta)
    print(
        f"\nMFCC+SVR(RBF) LOSO speaker-level: rho={mfcc_svr_metrics['rho']:.3f} "
        f"(p={mfcc_svr_metrics['rho_p']:.4f}) | pearson={mfcc_svr_metrics['pearson_r']:.3f} | "
        f"rmse={mfcc_svr_metrics['rmse']:.3f} | mae={mfcc_svr_metrics['mae']:.3f} | "
        f"rank_acc={mfcc_svr_metrics['rank_accuracy_dys_over_ctrl']:.3f}"
    )

    print("\n--- ATD zero-shot reference ---")
    atd_metrics = compute_atd_zero_shot_metrics()
    if "error" not in atd_metrics:
        print(
            f"ATD speaker-level: rho={atd_metrics['rho']:.3f} "
            f"(p={atd_metrics['rho_p']:.4f}) | pearson={atd_metrics['pearson_r']:.3f} | "
            f"rmse={atd_metrics['rmse']:.3f} (scale-mismatched) | "
            f"mae={atd_metrics['mae']:.3f} (scale-mismatched) | "
            f"rank_acc={atd_metrics['rank_accuracy_dys_over_ctrl']:.3f}"
        )
    else:
        print(f"ATD metrics unavailable: {atd_metrics['error']}")

    payload = {
        "wavlm_l9_meanpool_ridge_loso": ridge_metrics,
        "wavlm_l9_meanpool_linsvr_loso": svr_metrics,
        "mfcc_svr_rbf_loso": mfcc_svr_metrics,
        "atd_zero_shot_for_reference": atd_metrics,
        "method_notes": (
            "Frozen WavLM-Large; utterance feature = mean-pool of transformer "
            "layer 9 (1024-d). Regression head trained per-fold under leave-"
            "one-speaker-out across the 15 TORGO speakers. For each LOSO fold, "
            "hyperparameters were tuned by 5-fold group-CV on the 14 training "
            "speakers (Ridge alpha in {0.1,1,10,100,1000}; LinearSVR C in "
            "{0.01,0.1,1,10}). Per-utterance severity predictions were averaged "
            "to a single speaker-level prediction; metrics are reported on the "
            "n=15 speaker-level predictions. This frozen-backbone + classifier-"
            "head reproduction follows Hernandez et al. (2020) and Schu et al. "
            "(2023) in spirit; full backbone fine-tuning is degenerate at n=15 "
            "speakers. ATD is included for apples-to-apples comparison and is "
            "zero-shot (no training), so its LOSO and full-set metrics are "
            "equivalent."
        ),
        "config": {
            "backbone": "microsoft/wavlm-large (frozen)",
            "feature_layer": TARGET_LAYER,
            "pooling": "mean over time",
            "ssl_regression_heads": [
                "sklearn.linear_model.Ridge",
                "sklearn.svm.LinearSVR (squared_epsilon_insensitive)",
            ],
            "mfcc_baseline": (
                "58-d MFCC/acoustic features (src.baselines.extract_mfcc_features) "
                "+ RBF SVR; C in {0.1,1,10,100}, gamma in {scale,1e-3,1e-2,1e-1} "
                "selected by nested GroupKFold CV per LOSO fold"
            ),
            "cv_for_hp": "<=5-fold GroupKFold by speaker on the training fold; "
                         "StandardScaler fit on inner-train split only (nested)",
            "loso_speakers": sorted(meta["speaker_id"].unique().tolist()),
            "n_utterances_wavlm": int(len(meta)),
            "n_utterances_mfcc": int(len(mfcc_meta)),
            "per_fold_hyperparams_in": "each *_loso block's per_fold_selected_hyperparams",
            "random_seed": RANDOM_SEED,
            "seeded_via": "src.seed_utils.set_global_seed(42)",
            "reproduce_all_note": (
                "Should be wired into reproduce_all.py orchestrator (separate "
                "agent owns that file)."
            ),
        },
    }
    _atomic_dump_json(payload, OUTPUT_JSON)
    print(f"\nWrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    try:
        rc = main()
        sys.exit(rc)
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print("FATAL:", e, file=sys.stderr)
        print(tb, file=sys.stderr)
        _write_error_report(str(e), tb)
        sys.exit(1)
