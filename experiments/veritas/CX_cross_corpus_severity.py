"""VERITAS cross-corpus severity harness — within-corpus + leave-one-corpus-out.

The real test of whether a FROZEN WavLM-Large severity signal generalises across
etiologies and languages at scale. Corpus-agnostic: it runs on whatever corpora
are on disk (via ``src.veritas.corpora.registry``) and lists exactly what each
not-yet-ready corpus needs to activate — no edits required when a corpus lands.

Protocol (reuses the fairly-tuned nested-CV probe of experiments/42):
  * Feature: frozen WavLM-Large layer-9 mean-pool per utterance (1024-d);
    speaker prediction = mean over that speaker's utterances.
  * WITHIN corpus: leave-one-speaker-out; per fold, Ridge/LinearSVR
    hyper-parameter chosen by GroupKFold CV on the training speakers
    (StandardScaler fit inside each inner split -> leakage-free).
  * LEAVE-ONE-CORPUS-OUT: train on the pooled OTHER corpora, test on the
    held-out corpus. Labels harmonised to a per-corpus speaker z-rank (never
    pool raw heterogeneous labels). Features aligned by per-corpus standardisation
    (unsupervised CMVN; test corpus uses its OWN stats -> no label leakage).
  * Metric: speaker-level Spearman rho (+ rank-accuracy of affected>control).

Honesty: cross-corpus generalisation is historically HARD (this project's own
ATD failed it). Whatever the numbers are, they are reported straight.

Run:
  C:/Users/Dell/.conda/envs/dysarthria/python.exe experiments/veritas/CX_cross_corpus_severity.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

# LinearSVR (liblinear) can hit max_iter on the collinear 1024-d WavLM features;
# the solution is stable and tracks Ridge closely (Ridge is the primary probe).
# Silence the per-fold warning spam so the harness re-runs cleanly.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import src.veritas as _veritas  # noqa: E402,F401  (offline HF-cache bootstrap)
from src.cross_corpus.normalization import zrank_severity  # noqa: E402
from src.seed_utils import set_global_seed  # noqa: E402
from src.veritas.corpora import features as feat_mod  # noqa: E402
from src.veritas.corpora import registry  # noqa: E402

RANDOM_SEED = 42
DEVICE = "cuda:0"
OUTPUT_JSON = PROJECT_ROOT / "results" / "veritas" / "cross_corpus_severity.json"

RIDGE_ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
SVR_CS = [0.1, 1.0]  # trimmed: LinearSVR is the cost driver across ~130 LOSO folds
INNER_CV_FOLDS = 3   # inner GroupKFold folds for the nested hyper-parameter search

# Cap utterances per speaker for the probe (seeded). The task is speaker-level
# severity, so a bounded per-speaker sample preserves the speaker-mean signal
# while keeping the harness fast and cleanly re-runnable as corpora are added.
MAX_UTTS_PER_SPEAKER = 80


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, is_control: np.ndarray) -> dict:
    from scipy.stats import pearsonr, spearmanr

    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if len(y_true) < 3 or np.std(y_true) == 0:
        return {"n": int(len(y_true)), "rho": float("nan"), "rho_p": float("nan"),
                "pearson_r": float("nan"), "rank_accuracy_affected_over_control": float("nan")}
    rho, rho_p = spearmanr(y_true, y_pred)
    pear = float("nan") if np.std(y_pred) == 0 else float(pearsonr(y_true, y_pred)[0])
    is_ctrl = np.asarray(is_control, bool)
    ctrl, aff = y_pred[is_ctrl], y_pred[~is_ctrl]
    if len(ctrl) and len(aff):
        wins = sum((d > c) + 0.5 * (d == c) for d in aff for c in ctrl)
        rank_acc = float(wins / (len(aff) * len(ctrl)))
    else:
        rank_acc = float("nan")
    return {"n": int(len(y_true)), "rho": float(rho), "rho_p": float(rho_p),
            "pearson_r": pear, "rank_accuracy_affected_over_control": rank_acc}


def _make_model(kind: str, p: dict):
    from sklearn.linear_model import Ridge
    from sklearn.svm import LinearSVR

    if kind == "ridge":
        return Ridge(alpha=p["alpha"], random_state=RANDOM_SEED)
    return LinearSVR(C=p["C"], loss="squared_epsilon_insensitive",
                     random_state=RANDOM_SEED, tol=1e-3, max_iter=2000, dual="auto")


def _grid(kind: str) -> list[dict]:
    return [{"alpha": a} for a in RIDGE_ALPHAS] if kind == "ridge" else [{"C": c} for c in SVR_CS]


def _nested_pick(kind, X_tr, y_tr, groups_tr):
    """Pick hyper-parameter by GroupKFold CV (StandardScaler fit inside split)."""
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    n_groups = len(np.unique(groups_tr))
    if n_groups < 2 or len(_grid(kind)) == 1:
        return _grid(kind)[0]
    gkf = GroupKFold(n_splits=min(INNER_CV_FOLDS, n_groups))
    best_p, best = None, np.inf
    for p in _grid(kind):
        errs = []
        for tr, va in gkf.split(X_tr, y_tr, groups=groups_tr):
            sc = StandardScaler()
            m = _make_model(kind, p)
            m.fit(sc.fit_transform(X_tr[tr]), y_tr[tr])
            errs.append(float(np.mean((m.predict(sc.transform(X_tr[va])) - y_tr[va]) ** 2)))
        if np.mean(errs) < best:
            best, best_p = float(np.mean(errs)), p
    return best_p


def _cap_per_speaker(X: np.ndarray, meta: pd.DataFrame, cap: int, seed: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Seeded subsample of at most ``cap`` utterances per speaker (group)."""
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for _, idx in meta.groupby("group").groups.items():
        idx = np.asarray(idx)
        if len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        keep.extend(idx.tolist())
    keep = sorted(keep)
    return X[keep], meta.iloc[keep].reset_index(drop=True)


def _agg_speaker(meta: pd.DataFrame, utt_pred: np.ndarray) -> pd.DataFrame:
    d = meta.copy()
    d["pred"] = utt_pred
    g = d.groupby("group").agg(
        pred=("pred", "mean"),
        y=("severity_rank", "first"),
        is_control=("is_control", "first"),
    ).reset_index()
    return g


# ---------------------------------------------------------------------------
# within-corpus leave-one-speaker-out
# ---------------------------------------------------------------------------
def run_within_loso(X: np.ndarray, meta: pd.DataFrame, kind: str) -> dict:
    from sklearn.preprocessing import StandardScaler

    groups = meta["group"].to_numpy()
    y = meta["severity_rank"].to_numpy(float)
    speaker_rows = []
    for held in sorted(np.unique(groups)):
        tr, te = groups != held, groups == held
        p = _nested_pick(kind, X[tr], y[tr], groups[tr])
        sc = StandardScaler()
        m = _make_model(kind, p)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        pred = m.predict(sc.transform(X[te]))
        speaker_rows.append({
            "group": held, "pred": float(np.mean(pred)),
            "y": float(y[te][0]), "is_control": bool(meta.loc[te, "is_control"].iloc[0]),
        })
    sp = pd.DataFrame(speaker_rows)
    met = compute_metrics(sp["y"].to_numpy(), sp["pred"].to_numpy(), sp["is_control"].to_numpy())
    met["n_speakers"] = int(len(sp))
    return met


# ---------------------------------------------------------------------------
# leave-one-corpus-out
# ---------------------------------------------------------------------------
def run_loco(corpus_data: dict[str, tuple[np.ndarray, pd.DataFrame]], held: str, kind: str) -> dict:
    """Train on pooled OTHER corpora (per-corpus standardised features + z-rank
    targets), test on the held-out corpus (its own standardisation)."""
    from sklearn.preprocessing import StandardScaler

    train_names = [c for c in corpus_data if c != held]
    Xtr_parts, ytr_parts, gtr_parts = [], [], []
    for c in train_names:
        Xc, mc = corpus_data[c]
        Xc = StandardScaler().fit_transform(Xc)  # per-corpus CMVN (unsupervised)
        # per-corpus speaker z-rank target, broadcast to utterances
        sp = mc.groupby("group")["severity_rank"].first()
        zr = pd.Series(zrank_severity(sp.to_numpy()), index=sp.index)
        Xtr_parts.append(Xc)
        ytr_parts.append(mc["group"].map(zr).to_numpy(float))
        gtr_parts.append(mc["group"].to_numpy())
    Xtr = np.concatenate(Xtr_parts, 0)
    ytr = np.concatenate(ytr_parts, 0)
    gtr = np.concatenate(gtr_parts, 0)

    Xte_raw, mte = corpus_data[held]
    Xte = StandardScaler().fit_transform(Xte_raw)  # held-out uses ITS OWN stats (label-free)

    p = _nested_pick(kind, Xtr, ytr, gtr)
    m = _make_model(kind, p)
    m.fit(Xtr, ytr)
    sp = _agg_speaker(mte, m.predict(Xte))
    met = compute_metrics(sp["y"].to_numpy(), sp["pred"].to_numpy(), sp["is_control"].to_numpy())
    met["n_speakers"] = int(len(sp))
    met["train_corpora"] = train_names
    met["selected_hyperparam"] = {k: float(v) for k, v in p.items()}
    return met


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def _atomic_dump(payload: dict) -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_JSON.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(OUTPUT_JSON)


def main() -> int:
    set_global_seed(RANDOM_SEED)
    print("=" * 74)
    print(" VERITAS cross-corpus severity harness (within + leave-one-corpus-out)")
    print("=" * 74)

    statuses = registry.all_status()
    for st in statuses:
        tag = "OK " if st.available else "skip"
        print(f"[{tag}] {st.name:10s} {st.label_kind:17s} n~{st.n_speakers_hint:<3d} {st.reason}")

    data = registry.load_available(severity_targets_only=True)
    if not data:
        raise RuntimeError("no severity-target corpus available on disk")

    # Build per-corpus features (utterance-level) + a corpus::speaker group key.
    corpus_data: dict[str, tuple[np.ndarray, pd.DataFrame]] = {}
    corpora_info: dict[str, dict] = {}
    for name, recs in data.items():
        t0 = time.time()
        X, meta = feat_mod.build_corpus_features(recs, device=DEVICE)
        if len(meta) == 0:
            print(f"[skip] {name}: 0 usable feature rows")
            continue
        meta = meta.reset_index(drop=True)
        meta["group"] = meta["corpus"] + "::" + meta["speaker_id"]
        n_utts_full = int(len(meta))
        X, meta = _cap_per_speaker(X, meta, MAX_UTTS_PER_SPEAKER, RANDOM_SEED)
        corpus_data[name] = (X, meta)
        levels = sorted(meta.groupby("group")["severity_rank"].first().unique().tolist())
        corpora_info[name] = {
            "n_speakers": int(meta["group"].nunique()),
            "n_utterances_used": int(len(meta)),
            "n_utterances_available": n_utts_full,
            "language": recs[0].language,
            "etiologies": sorted(meta["etiology"].unique().tolist()),
            "label_kind": next(s.label_kind for s in statuses if s.name == name),
            "severity_levels": levels,
            "n_severity_levels": len(levels),
        }
        print(f"[feat] {name}: X={X.shape}, speakers={corpora_info[name]['n_speakers']}, "
              f"levels={levels} ({time.time() - t0:.0f}s)")

    # ---- within-corpus ----
    within = {}
    for name, (X, meta) in corpus_data.items():
        if corpora_info[name]["n_severity_levels"] < 2 or corpora_info[name]["n_speakers"] < 4:
            within[name] = {"skipped": "need >=2 severity levels and >=4 speakers"}
            continue
        print(f"\n--- within {name} (LOSO) ---")
        res = {}
        for kind in ("ridge", "linsvr"):
            res[kind] = run_within_loso(X, meta, kind)
            print(f"  {kind:7s}: rho={res[kind]['rho']:+.3f} (p={res[kind]['rho_p']:.3g}) "
                  f"rank_acc={res[kind]['rank_accuracy_affected_over_control']:.3f} "
                  f"n={res[kind]['n_speakers']}")
        within[name] = res

    # ---- leave-one-corpus-out ----
    loco = {}
    if len(corpus_data) >= 2:
        for held in corpus_data:
            print(f"\n--- LOCO: train on others, test on {held} ---")
            res = {}
            for kind in ("ridge", "linsvr"):
                res[kind] = run_loco(corpus_data, held, kind)
                print(f"  {kind:7s}: rho={res[kind]['rho']:+.3f} (p={res[kind]['rho_p']:.3g}) "
                      f"rank_acc={res[kind]['rank_accuracy_affected_over_control']:.3f} "
                      f"train={res[kind]['train_corpora']}")
            loco[held] = res
    else:
        loco = {"skipped": "leave-one-corpus-out needs >=2 available corpora"}

    skipped = [
        {"name": s.name, "label_kind": s.label_kind, "is_severity_target": s.is_severity_target,
         "reason": s.reason, "activation": s.activation, "n_speakers_hint": s.n_speakers_hint}
        for s in statuses if not s.available
    ]

    payload = {
        "config": {
            "random_seed": RANDOM_SEED,
            "feature": "frozen WavLM-Large layer-9 mean-pool (1024-d), speaker=mean over utts",
            "device": DEVICE,
            "probe": "Ridge / LinearSVR, nested GroupKFold CV per fold (reuses exp 42 protocol)",
            "within_eval": "leave-one-speaker-out",
            "loco_eval": "train pooled others -> test held-out corpus; per-corpus speaker "
                         "z-rank targets; per-corpus unsupervised standardisation (CMVN)",
            "ridge_alphas": RIDGE_ALPHAS,
            "svr_Cs": SVR_CS,
            "inner_cv_folds": INNER_CV_FOLDS,
            "max_utts_per_speaker": MAX_UTTS_PER_SPEAKER,
        },
        "corpora_on_disk": corpora_info,
        "within_corpus": within,
        "leave_one_corpus_out": loco,
        "skipped_corpora": skipped,
        "confounds_flagged": [
            "Speech material differs across corpora (TORGO words+sentences EN; IPVS "
            "sustained vowels/DDK/read passages IT; EasyCall voice commands IT) — a "
            "cross-corpus task-mismatch confound.",
            "Silence/VAD: TORGO features come from the project's existing non-VAD-trimmed "
            "L9 cache; other corpora are mean-pooled over full clips. Silence dilutes the "
            "pooled feature and is a known TORGO confound (see exp 43_silence_confound). "
            "Not VAD-trimmed here; flagged.",
            "IPVS labels are binary PD-vs-HC (no graded UPDRS/H&Y in the public release); "
            "IPVS controls mix young + elderly while all PD are older (age confound; "
            "ipvs.load(elderly_only=True) restricts to the age-matched subset).",
            "EasyCall labels are binary dysarthric-vs-healthy (graded TOM severity is in "
            "the source paper, not the audio package).",
        ],
        "honest_summary": _summary(within, loco, corpora_info),
    }
    _atomic_dump(payload)
    print(f"\nWrote {OUTPUT_JSON}")
    print("\n" + payload["honest_summary"])
    return 0


def _summary(within: dict, loco: dict, info: dict) -> str:
    disk = ", ".join("{}(n={})".format(k, v["n_speakers"]) for k, v in info.items())
    lines = ["On-disk corpora: " + disk + "."]
    for name, res in within.items():
        if "ridge" in res:
            lines.append(f"WITHIN {name}: ridge rho={res['ridge']['rho']:+.3f}, "
                         f"linsvr rho={res['linsvr']['rho']:+.3f} (speaker LOSO).")
    if isinstance(loco, dict) and "skipped" not in loco:
        for held, res in loco.items():
            lines.append(f"LOCO test-{held}: ridge rho={res['ridge']['rho']:+.3f}, "
                         f"linsvr rho={res['linsvr']['rho']:+.3f} "
                         f"(train {res['ridge']['train_corpora']}).")
    else:
        lines.append("LOCO: skipped (needs >=2 corpora).")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        print("FATAL:", e, file=sys.stderr)
        print(tb, file=sys.stderr)
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_JSON.write_text(json.dumps({"error": str(e), "traceback": tb}, indent=2), encoding="utf-8")
        sys.exit(1)
