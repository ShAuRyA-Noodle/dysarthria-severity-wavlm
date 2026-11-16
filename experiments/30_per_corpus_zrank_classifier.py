"""Per-corpus z-rank normalised differential-dx classifier.

Replaces the v1 pooled-raw-ATD classifier whose macro-F1=0.573 result
had Parkinson speakers uniformly mis-classified as Control. The
failure was caused by pooling raw ATD across corpora whose internal
scales differ.

This v2:
  1. Z-rank-normalises mean_atd within each corpus (TORGO / UASpeech / IPVS).
  2. Builds a 4-class label aligned across corpora.
  3. LOSO logistic regression on the z-ranked feature.
  4. Reports macro-F1, per-class metrics, confusion matrix.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TORGO_SEVERITY
from src.cross_corpus.normalization import zrank_severity
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _zrank(values: np.ndarray) -> np.ndarray:
    """Rank-based z normalisation (invariant to monotone scale)."""
    ranks = np.argsort(np.argsort(values)) + 1.0
    if len(ranks) < 2:
        return np.zeros_like(ranks)
    return (ranks - ranks.mean()) / (ranks.std() or 1.0)


def main() -> None:
    set_global_seed(RANDOM_SEED)

    # Load per-corpus speaker tables.
    torgo = pd.read_csv(PROJECT_ROOT / "results/speaker_atd.csv")
    torgo = torgo[["speaker_id", "mean_atd", "severity_score"]].copy()
    torgo["corpus"] = "TORGO"
    torgo["label_4cls"] = torgo["severity_score"].map(
        {0: "Control", 1: "Mild-Mod", 2: "Severe"}
    )

    parts = [torgo]
    ua_p = PROJECT_ROOT / "results/uaspeech_speaker_atd.csv"
    if ua_p.exists():
        ua = pd.read_csv(ua_p)
        ua = ua.copy(); ua["corpus"] = "UASpeech"
        ua["speaker_id"] = "UA_" + ua["speaker_id"].astype(str)
        if "intelligibility" in ua.columns:
            ua["label_4cls"] = ua["intelligibility"].apply(
                lambda x: "Control" if x >= 75 else ("Mild-Mod" if x >= 25 else "Severe")
            )
        else:
            ua["label_4cls"] = "Mild-Mod"
        ua = ua[["speaker_id", "mean_atd", "corpus", "label_4cls"]]
        parts.append(ua)

    ipvs_p = PROJECT_ROOT / "results/cross_corpus/ipvs_speaker_atd_layer9.csv"
    if ipvs_p.exists():
        ipvs = pd.read_csv(ipvs_p)
        ipvs = ipvs.copy(); ipvs["corpus"] = "IPVS"
        ipvs["label_4cls"] = ipvs["is_control"].astype(bool).map(
            {True: "Control", False: "Parkinson"}
        )
        ipvs = ipvs[["speaker_id", "mean_atd", "corpus", "label_4cls"]]
        parts.append(ipvs)

    df = pd.concat(parts, ignore_index=True)
    logger.info("pooled n=%d, by corpus=%s", len(df), dict(df["corpus"].value_counts()))
    logger.info("classes: %s", dict(df["label_4cls"].value_counts()))

    # Per-corpus z-rank of mean_atd. THIS is the v1-fix.
    df["atd_zrank"] = (
        df.groupby("corpus")["mean_atd"].transform(lambda s: _zrank(s.to_numpy()))
    )

    X = df[["atd_zrank"]].to_numpy()
    y = df["label_4cls"].to_numpy()
    # LOSO.
    preds = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        train = np.array([j for j in range(len(y)) if j != i])
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial",
                                 random_state=RANDOM_SEED)
        clf.fit(X[train], y[train])
        preds[i] = clf.predict(X[i:i+1])[0]

    classes = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=classes)
    macro_f1 = float(f1_score(y, preds, average="macro"))
    rep = classification_report(y, preds, labels=classes, output_dict=True)

    out = {
        "n_speakers": int(len(y)),
        "feature": "atd_zrank (per-corpus rank z-normalised mean ATD)",
        "loso_macro_f1": macro_f1,
        "labels_in_order": classes,
        "confusion_matrix": cm.tolist(),
        "per_class": rep,
        "by_corpus": dict(df["corpus"].value_counts()),
        "class_distribution": dict(pd.Series(y).value_counts()),
    }
    out_path = PROJECT_ROOT / "results/differential_dx_zrank_loso.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({"macro_f1": macro_f1, "classes": classes}, indent=2))
    print("confusion (true x pred):")
    print(pd.DataFrame(cm, index=classes, columns=classes).to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right"); ax.set_yticklabels(classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"v2 z-rank LOSO confusion (macro-F1={macro_f1:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(PROJECT_ROOT / f"results/figures/fig_diff_dx_zrank_confusion{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
