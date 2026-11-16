"""Shared linear-probe machinery (fairly-tuned Ridge), reused by the pooled-WavLM
probe metric and the cross-corpus transfer test.

This mirrors the project's nested-CV protocol
(``experiments/veritas/CX_cross_corpus_severity.py``) exactly — StandardScaler
fit inside each split (leakage-free), alpha chosen by GroupKFold CV on the
training speakers — so numbers reproduce the committed cross-corpus results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
INNER_CV_FOLDS = 3
SEED = 42


def _make_ridge(alpha: float):
    from sklearn.linear_model import Ridge

    return Ridge(alpha=alpha, random_state=SEED)


def nested_pick_alpha(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray,
    alphas=RIDGE_ALPHAS, folds: int = INNER_CV_FOLDS
) -> float:
    """Pick Ridge alpha by GroupKFold CV (StandardScaler fit inside each split)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return alphas[len(alphas) // 2]
    gkf = GroupKFold(n_splits=min(folds, n_groups))
    best_a, best = alphas[0], np.inf
    for a in alphas:
        errs = []
        for tr, va in gkf.split(X, y, groups=groups):
            sc = StandardScaler()
            m = _make_ridge(a)
            m.fit(sc.fit_transform(X[tr]), y[tr])
            errs.append(float(np.mean((m.predict(sc.transform(X[va])) - y[va]) ** 2)))
        if np.mean(errs) < best:
            best, best_a = float(np.mean(errs)), a
    return best_a


def within_loso_predict(
    X: np.ndarray, meta: pd.DataFrame, group_col: str = "group",
    y_col: str = "severity_rank"
) -> dict[str, float]:
    """Leave-one-speaker-out Ridge; returns {speaker_id -> mean utt prediction}.

    Speaker-disjoint: each held-out speaker's utterances are predicted by a
    probe trained (with nested alpha selection) on all OTHER speakers.
    """
    from sklearn.preprocessing import StandardScaler

    groups = meta[group_col].to_numpy()
    speaker = meta["speaker_id"].to_numpy()
    y = meta[y_col].to_numpy(float)
    out: dict[str, float] = {}
    for held in np.unique(groups):
        tr = groups != held
        te = groups == held
        a = nested_pick_alpha(X[tr], y[tr], groups[tr])
        sc = StandardScaler()
        m = _make_ridge(a)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        spk = str(speaker[te][0])
        out[spk] = float(np.mean(pred))
    return out


def loco_transfer_predict(
    bundles: dict[str, "object"], held: str
) -> dict[str, float]:
    """Train on pooled OTHER corpora, predict held-out corpus speakers.

    Per-corpus unsupervised standardisation (CMVN) + per-corpus speaker z-rank
    targets (never pool raw heterogeneous labels). Held-out corpus uses its OWN
    standardisation (label-free). Returns {speaker_id -> mean utt prediction}.
    """
    from sklearn.preprocessing import StandardScaler

    from src.cross_corpus.normalization import zrank_severity

    train_names = [c for c in bundles if c != held]
    Xtr_parts, ytr_parts, gtr_parts = [], [], []
    for c in train_names:
        b = bundles[c]
        Xc = StandardScaler().fit_transform(b.X_utt)
        sp = b.meta.groupby("group")["severity_rank"].first()
        zr = pd.Series(zrank_severity(sp.to_numpy()), index=sp.index)
        Xtr_parts.append(Xc)
        ytr_parts.append(b.meta["group"].map(zr).to_numpy(float))
        gtr_parts.append(b.meta["group"].to_numpy())
    Xtr = np.concatenate(Xtr_parts, 0)
    ytr = np.concatenate(ytr_parts, 0)
    gtr = np.concatenate(gtr_parts, 0)

    bh = bundles[held]
    Xte = StandardScaler().fit_transform(bh.X_utt)
    a = nested_pick_alpha(Xtr, ytr, gtr)
    m = _make_ridge(a)
    m.fit(Xtr, ytr)
    pred = m.predict(Xte)
    d = bh.meta.copy()
    d["_pred"] = pred
    g = d.groupby("speaker_id")["_pred"].mean()
    return {str(k): float(v) for k, v in g.items()}
