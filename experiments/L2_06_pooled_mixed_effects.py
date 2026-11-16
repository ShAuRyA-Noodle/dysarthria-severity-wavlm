"""4-corpus pooled ATD analysis.

Three complementary models:

  A) Severity grading. TORGO + UASpeech only (both have severity variation).
       atd_z ~ severity + C(corpus) + (1|speaker)
     Establishes β_severity with tight per-utterance CI.

  B) Dysarthric-vs-healthy binary classification across all 4 corpora.
       atd_z ~ is_dysarthric + C(corpus) + (1|speaker)
     ATD elevation is dysarthric-specific iff L2/VCTK accented healthy
     speakers cluster with TORGO/UA controls, NOT with dysarthric speakers.

  C) Per-population mean atd_z table (TORGO_ctrl, TORGO_dys, UA_ctrl, UA_dys,
     L2, VCTK) plus pairwise Cohen's d. Reader-friendly forest plot.

Output:
  results/cross_corpus_pooled_mixed_effects.json
  results/cross_corpus_pooled_mixed_effects_summary.txt
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_torgo() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "results/atd_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).dropna(subset=["atd_score"]).copy()
    return pd.DataFrame({
        "corpus": "TORGO",
        "speaker_id": "TORGO_" + df["speaker_id"].astype(str),
        "atd": df["atd_score"].astype(float),
        "severity": df["severity_score"].astype(int),
        "is_accented": 0,
    })


def _load_uaspeech() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "results/uaspeech_atd_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).dropna(subset=["atd_score"]).copy()
    sev_col = next((c for c in ("severity_proxy", "severity_score", "severity")
                    if c in df.columns), None)
    if sev_col is None and "intelligibility_score" in df.columns:
        df["severity_proxy"] = pd.cut(df["intelligibility_score"],
                                      bins=[-0.1, 50, 75, 200], labels=[2, 1, 0]).astype(int)
        sev_col = "severity_proxy"
    if sev_col is None:
        return None
    return pd.DataFrame({
        "corpus": "UASpeech",
        "speaker_id": "UA_" + df["speaker_id"].astype(str),
        "atd": df["atd_score"].astype(float),
        "severity": pd.to_numeric(df[sev_col], errors="coerce").fillna(0).astype(int),
        "is_accented": 0,
    })


def _load_l2arctic() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "results/l2arctic_atd_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).dropna(subset=["atd"]).copy()
    return pd.DataFrame({
        "corpus": "L2ARCTIC",
        "speaker_id": "L2_" + df["speaker_id"].astype(str),
        "atd": df["atd"].astype(float),
        "severity": 0,
        "is_accented": 1,
    })


def _load_vctk() -> pd.DataFrame | None:
    p = PROJECT_ROOT / "results/vctk_atd_scores.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p).dropna(subset=["atd"]).copy()
    return pd.DataFrame({
        "corpus": "VCTK",
        "speaker_id": "VCTK_" + df["speaker_id"].astype(str),
        "atd": df["atd"].astype(float),
        "severity": 0,
        "is_accented": 0,
    })


def _zscore_by_corpus(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["atd_z"] = df.groupby("corpus")["atd"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0)
    )
    return df


def _cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    if pooled == 0:
        return float("nan")
    return float((x.mean() - y.mean()) / pooled)


def _fit_severity_model(df_sev: pd.DataFrame) -> dict:
    """OLS with cluster-robust SE on speaker_id (speaker × severity collinear
    within corpus, so mixed-effects on speaker is degenerate)."""
    res = smf.ols("atd_z ~ severity + C(corpus)", data=df_sev).fit(
        cov_type="cluster", cov_kwds={"groups": df_sev["speaker_id"]}
    )
    ci = res.conf_int(alpha=0.05).loc["severity"]
    return {
        "formula": "atd_z ~ severity + C(corpus); cluster-robust SE on speaker",
        "n_obs": int(len(df_sev)),
        "n_speakers": int(df_sev["speaker_id"].nunique()),
        "n_clusters": int(df_sev["speaker_id"].nunique()),
        "beta_severity": float(res.params["severity"]),
        "se_severity": float(res.bse["severity"]),
        "ci_severity_low": float(ci[0]),
        "ci_severity_high": float(ci[1]),
        "p_severity": float(res.pvalues["severity"]),
        "r_squared": float(res.rsquared),
        "summary": str(res.summary()),
    }


def _fit_binary_dysarthric_model(df_all: pd.DataFrame) -> dict:
    res = smf.ols("atd_z ~ is_dysarthric + C(corpus)", data=df_all).fit(
        cov_type="cluster", cov_kwds={"groups": df_all["speaker_id"]}
    )
    ci = res.conf_int(alpha=0.05).loc["is_dysarthric"]
    return {
        "formula": "atd_z ~ is_dysarthric + C(corpus); cluster-robust SE on speaker",
        "n_obs": int(len(df_all)),
        "n_speakers": int(df_all["speaker_id"].nunique()),
        "n_clusters": int(df_all["speaker_id"].nunique()),
        "beta_is_dysarthric": float(res.params["is_dysarthric"]),
        "se": float(res.bse["is_dysarthric"]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p_value": float(res.pvalues["is_dysarthric"]),
        "r_squared": float(res.rsquared),
        "summary": str(res.summary()),
    }


def main() -> None:
    set_global_seed(42)
    parts = [p for p in (_load_torgo(), _load_uaspeech(), _load_l2arctic(), _load_vctk())
             if p is not None]
    if not parts:
        raise SystemExit("no corpus data found")
    df = pd.concat(parts, ignore_index=True)
    df = _zscore_by_corpus(df)
    df["is_dysarthric"] = (df["severity"] > 0).astype(int)
    logger.info("pooled n_obs=%d  speakers=%d  corpora=%s",
                len(df), df["speaker_id"].nunique(), df["corpus"].value_counts().to_dict())

    # Per-population stratum stats.
    def _pop_label(r) -> str:
        if r["corpus"] in ("TORGO", "UASpeech"):
            return f"{r['corpus']}_sev{int(r['severity'])}"
        return r["corpus"]
    df["population"] = df.apply(_pop_label, axis=1)
    pop_stats = df.groupby("population")["atd_z"].agg(["mean", "std", "count"]).round(4)
    logger.info("per-population atd_z means:\n%s", pop_stats.to_string())

    # Model A: severity grading on TORGO + UASpeech.
    df_sev = df[df["corpus"].isin(["TORGO", "UASpeech"])].copy()
    model_a = _fit_severity_model(df_sev)

    # Model B: dysarthric vs healthy on all 4 corpora.
    model_b = _fit_binary_dysarthric_model(df)

    # Pairwise Cohen's d on per-population atd_z.
    pops = list(pop_stats.index)
    pairs: dict[str, float] = {}
    for i in range(len(pops)):
        for j in range(i + 1, len(pops)):
            x = df[df["population"] == pops[i]]["atd_z"].to_numpy()
            y = df[df["population"] == pops[j]]["atd_z"].to_numpy()
            pairs[f"{pops[i]} vs {pops[j]}"] = _cohens_d(x, y)

    out = {
        "n_observations": int(len(df)),
        "n_speakers": int(df["speaker_id"].nunique()),
        "corpus_counts": df["corpus"].value_counts().to_dict(),
        "per_population_atd_z": pop_stats.to_dict("index"),
        "model_a_severity_grading_torgo_ua": {k: v for k, v in model_a.items() if k != "summary"},
        "model_b_dysarthric_binary_all_corpora": {k: v for k, v in model_b.items() if k != "summary"},
        "pairwise_cohens_d_on_z_atd": pairs,
        "interpretation": (
            "Model A: severity beta on z-ATD pooled across TORGO + UASpeech. "
            "Model B: dysarthric-vs-healthy contrast across 4 corpora. If "
            "L2ARCTIC + VCTK z-ATD means hug TORGO/UA control z-ATD (NOT "
            "dysarthric), ATD is specific to dysarthric pathology rather "
            "than to pronunciation deviation generally."
        ),
    }
    out_path = PROJECT_ROOT / "results/cross_corpus_pooled_mixed_effects.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    (PROJECT_ROOT / "results/cross_corpus_pooled_mixed_effects_summary.txt").write_text(
        f"=== MODEL A (severity grading) ===\n{model_a['summary']}\n\n"
        f"=== MODEL B (dysarthric binary) ===\n{model_b['summary']}",
        encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
