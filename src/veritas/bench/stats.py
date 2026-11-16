"""Statistical primitives for the VERITAS validity battery.

Small, dependency-light, deterministic. Every function here operates on REAL
per-speaker vectors; nothing is simulated. Seed-controlled where randomness is
used (permutation / bootstrap / classifier splits).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def speaker_spearman(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float, int]:
    """Speaker-level Spearman rho + p-value + n. NaN-safe (degenerate -> nan)."""
    y = np.asarray(y_true, dtype=float)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y, s = y[mask], s[mask]
    n = int(len(y))
    if n < 3 or np.std(y) == 0 or np.std(s) == 0:
        return float("nan"), float("nan"), n
    rho, p = spearmanr(y, s)
    return float(rho), float(p), n


def partial_spearman(
    y: np.ndarray, x: np.ndarray, z: np.ndarray
) -> tuple[float, float, int]:
    """Partial Spearman rho of (y, x) controlling for z.

    Rank-transform all three variables, then apply the first-order partial
    *Pearson* correlation formula to the ranks. This is the standard
    rank-based partial correlation. p-value from a t-test with df = n - 3.
    """
    from scipy.stats import rankdata, t as tdist

    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    y, x, z = y[mask], x[mask], z[mask]
    n = int(len(y))
    if n < 4:
        return float("nan"), float("nan"), n
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    if np.std(ry) == 0 or np.std(rx) == 0 or np.std(rz) == 0:
        return float("nan"), float("nan"), n

    def _r(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    r_yx, r_yz, r_xz = _r(ry, rx), _r(ry, rz), _r(rx, rz)
    denom = np.sqrt(max((1.0 - r_yz**2) * (1.0 - r_xz**2), 1e-12))
    r_part = (r_yx - r_yz * r_xz) / denom
    r_part = float(np.clip(r_part, -1.0, 1.0))
    # two-sided p via t distribution
    df = n - 3
    if df <= 0 or abs(r_part) >= 1.0:
        p = float("nan")
    else:
        tstat = r_part * np.sqrt(df / max(1.0 - r_part**2, 1e-12))
        p = float(2.0 * tdist.sf(abs(tstat), df))
    return r_part, p, n


def retention(sub_rho: float, base_rho: float) -> float:
    """Signed fraction of the base correlation retained by a sub-condition.

    retention = sub_rho / base_rho. Returns nan if the base is ~0 (undefined).
    A value near 1.0 means the correlation survived the ablation; near 0 (or
    negative) means it collapsed.
    """
    if not np.isfinite(base_rho) or abs(base_rho) < 1e-9:
        return float("nan")
    return float(sub_rho / base_rho)


def speaker_identity_auc(
    X: np.ndarray, speaker_ids: np.ndarray, seed: int = 42, n_splits: int = 5
) -> tuple[float, int]:
    """Macro one-vs-rest AUC for predicting SPEAKER identity from features.

    A leakage probe: high AUC means the representation encodes *who* is speaking
    (identity), which is what inflates reference-based severity metrics via
    speaker leakage. Utterances are split with a stratified group-free
    StratifiedKFold on speaker labels (each speaker appears in train and test),
    multinomial logistic regression is fit per fold, and out-of-fold posteriors
    give a macro OVR AUC. Chance = 0.5.

    Returns (macro_auc, n_speakers). NaN if too few utterances/speakers.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import LabelBinarizer, StandardScaler

    X = np.asarray(X, dtype=float)
    y = np.asarray(speaker_ids)
    classes = np.unique(y)
    n_spk = int(len(classes))
    if n_spk < 2 or len(y) < n_spk * 2:
        return float("nan"), n_spk

    # need >= n_splits members in the smallest class for stratification
    min_count = min(int(np.sum(y == c)) for c in classes)
    k = int(min(n_splits, max(2, min_count)))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    lb = LabelBinarizer().fit(classes)
    oof = np.zeros((len(y), n_spk), dtype=float)
    filled = np.zeros(len(y), dtype=bool)
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(
            max_iter=2000, C=1.0, random_state=seed,
        )
        clf.fit(sc.transform(X[tr]), y[tr])
        proba = clf.predict_proba(sc.transform(X[te]))
        # align classifier's class order to the global class order
        col = {c: j for j, c in enumerate(clf.classes_)}
        for j, c in enumerate(classes):
            if c in col:
                oof[te, j] = proba[:, col[c]]
        filled[te] = True
    Y = lb.transform(y)
    if Y.shape[1] == 1:  # binary LabelBinarizer edge case
        Y = np.hstack([1 - Y, Y])
    try:
        auc = float(roc_auc_score(Y[filled], oof[filled], average="macro", multi_class="ovr"))
    except ValueError:
        return float("nan"), n_spk
    return auc, n_spk
