"""T3 — Differential-diagnosis classifier across corpora.

Per-corpus speaker-level features pooled into one classification task.
Four labels: control / mild dys / severe dys / Parkinson.

TORGO: severity 0/1/2 -> Control / Mild / Severe.
IPVS:  group HC_* / PD  -> Control / Parkinson.
UASpeech: intelligibility >75% -> Control, else Dys (collapse mild+severe by
  median split since UA labels are continuous).

Speaker feature vectors: mean ATD + ATD std + n_utterances (3-D) from each
corpus's speaker_atd CSV. Logistic-regression classifier with LOSO over all
pooled speakers. Reports macro-F1, per-class precision/recall, confusion
matrix.
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
from sklearn.metrics import confusion_matrix, f1_score, classification_report

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED, TORGO_SEVERITY
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CLASS_LABELS = ["Control", "Mild-Mod", "Severe", "Parkinson"]


def _load_torgo() -> pd.DataFrame:
    df = pd.read_csv(PROJECT_ROOT / "results/speaker_atd.csv")
    df["corpus"] = "TORGO"
    df["label_4cls"] = df["severity_score"].map({0: "Control", 1: "Mild-Mod", 2: "Severe"})
    df["std_atd"] = df.get("std_atd", df["mean_atd"] * 0.0)
    df["n_utts"] = df.get("n_utterances", 1)
    return df[["speaker_id", "corpus", "mean_atd", "std_atd", "n_utts", "label_4cls"]]


def _load_uaspeech() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "results/uaspeech_speaker_atd.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["corpus"] = "UASpeech"
    # UASpeech: high intelligibility -> Control; low -> Severe; mid -> Mild
    if "intelligibility" in df.columns:
        df["label_4cls"] = df["intelligibility"].apply(
            lambda x: "Control" if x >= 75 else ("Mild-Mod" if x >= 25 else "Severe")
        )
    else:
        df["label_4cls"] = "Mild-Mod"
    df["std_atd"] = df.get("std_atd", df["mean_atd"] * 0.0)
    df["n_utts"] = df.get("n_utterances", df.get("n_utts", 1))
    return df[["speaker_id", "corpus", "mean_atd", "std_atd", "n_utts", "label_4cls"]]


def _load_ipvs() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "results/cross_corpus/ipvs_speaker_atd_layer9.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df["corpus"] = "IPVS"
    df["label_4cls"] = df["is_control"].astype(bool).map(
        {True: "Control", False: "Parkinson"}
    )
    df["std_atd"] = 0.0
    df["n_utts"] = df.get("n_utts", 1)
    return df[["speaker_id", "corpus", "mean_atd", "std_atd", "n_utts", "label_4cls"]]


def main() -> None:
    set_global_seed(RANDOM_SEED)
    parts = [_load_torgo()]
    ua = _load_uaspeech()
    if ua is not None:
        # Prefix UA speaker IDs to avoid collision with TORGO.
        ua = ua.copy(); ua["speaker_id"] = "UA_" + ua["speaker_id"].astype(str)
        parts.append(ua)
    ipvs = _load_ipvs()
    if ipvs is not None:
        parts.append(ipvs)
    df = pd.concat(parts, ignore_index=True)
    logger.info("pooled speakers: %d  by corpus=%s",
                len(df), dict(df["corpus"].value_counts()))
    logger.info("class distribution: %s", dict(df["label_4cls"].value_counts()))

    X = df[["mean_atd", "std_atd", "n_utts"]].fillna(0.0).to_numpy(dtype=float)
    y = df["label_4cls"].to_numpy()
    speakers = df["speaker_id"].to_numpy()

    # LOSO over pooled speakers.
    preds = np.empty(len(y), dtype=object)
    for i in range(len(y)):
        train_idx = np.array([j for j in range(len(y)) if j != i])
        clf = LogisticRegression(max_iter=1000, multi_class="multinomial",
                                 random_state=RANDOM_SEED)
        clf.fit(X[train_idx], y[train_idx])
        preds[i] = clf.predict(X[i:i+1])[0]

    classes_present = sorted(set(y))
    cm = confusion_matrix(y, preds, labels=classes_present)
    macro_f1 = float(f1_score(y, preds, average="macro"))
    cls_rep = classification_report(y, preds, labels=classes_present, output_dict=True)

    out = {
        "n_speakers": int(len(y)),
        "corpora": dict(df["corpus"].value_counts()),
        "class_distribution": dict(pd.Series(y).value_counts()),
        "loso_macro_f1": macro_f1,
        "labels_in_order": classes_present,
        "confusion_matrix": cm.tolist(),
        "per_class": cls_rep,
    }
    with open(PROJECT_ROOT / "results/differential_dx_loso.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(json.dumps({"loso_macro_f1": macro_f1, "classes": classes_present}, indent=2))
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(cm, index=classes_present, columns=classes_present).to_string())

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes_present))); ax.set_yticks(range(len(classes_present)))
    ax.set_xticklabels(classes_present, rotation=45, ha="right")
    ax.set_yticklabels(classes_present)
    for i in range(len(classes_present)):
        for j in range(len(classes_present)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="black" if cm[i, j] < cm.max() / 2 else "white", fontsize=11)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"LOSO confusion (macro-F1 = {macro_f1:.3f})")
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    for ext in (".png", ".pdf"):
        fig.savefig(PROJECT_ROOT / f"results/figures/fig_diff_dx_confusion{ext}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
