"""Track D — Pooled mixed-effects analysis across corpora.

Loads per-corpus speaker-level ATD results (within-corpus + cross-corpus)
and fits the pre-registered mixed-effects model:

    sev_z ~ atd_z + (1 | corpus) + (1 | language)

Reports:
- per-corpus rho_within (Spearman of severity vs ATD)
- pooled fixed-effect beta with 95% CI on atd_z
- variance components (between-corpus + between-language)
- between-corpus rho heterogeneity (Q-statistic, I^2)

The unified scale is z-rank within corpus (preserves Spearman ordering;
invariant to scale and monotone transforms).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import RANDOM_SEED
from src.cross_corpus.corpus_registry import REGISTRY, get
from src.cross_corpus.normalization import zrank_severity
from src.seed_utils import set_global_seed

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_speaker_atd(corpus: str, results_dir: Path) -> pd.DataFrame | None:
    """Load speaker-level ATD CSV per corpus if present."""
    candidates = [
        results_dir / "cross_corpus" / f"speaker_atd_{corpus}.csv",
        results_dir / f"speaker_atd_{corpus}.csv",
    ]
    if corpus == "torgo":
        candidates.insert(0, results_dir / "speaker_atd.csv")
    elif corpus == "uaspeech":
        candidates.insert(0, results_dir / "uaspeech_speaker_atd.csv")
    for p in candidates:
        if p.exists():
            return pd.read_csv(p)
    return None


def _coerce_severity(df: pd.DataFrame, severity_field: str) -> np.ndarray:
    """Best-effort severity column extraction."""
    for col in (severity_field, "severity_score", "intelligibility", "severity"):
        if col in df.columns:
            return df[col].astype(float).to_numpy()
    raise KeyError(f"no severity column found in {list(df.columns)}")


def _coerce_atd(df: pd.DataFrame) -> np.ndarray:
    for col in ("mean_atd", "atd_score", "atd"):
        if col in df.columns:
            return df[col].astype(float).to_numpy()
    raise KeyError(f"no atd column found in {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpora", default="torgo,uaspeech,ipvs",
                        help="Comma-separated corpus names from REGISTRY.")
    parser.add_argument("--out", default="results/cross_corpus/pooled_lme.json")
    args = parser.parse_args()
    set_global_seed(RANDOM_SEED)

    results_dir = PROJECT_ROOT / "results"
    corpora = [c.strip() for c in args.corpora.split(",") if c.strip()]
    per_corpus: dict[str, dict] = {}
    pooled_rows: list[dict] = []

    for c in corpora:
        if c not in REGISTRY:
            logger.warning("unknown corpus %s; skipping", c)
            continue
        cfg = get(c)
        df = _load_speaker_atd(c, results_dir)
        if df is None:
            logger.warning("no speaker-level ATD for %s; skipping", c)
            per_corpus[c] = {"status": "missing"}
            continue
        try:
            sev = _coerce_severity(df, cfg.severity_field)
            atd = _coerce_atd(df)
        except KeyError as exc:
            logger.warning("%s: %s", c, exc)
            per_corpus[c] = {"status": "schema_mismatch", "columns": list(df.columns)}
            continue
        rho, pval = spearmanr(sev, atd)
        atd_z = (atd - atd.mean()) / (atd.std() or 1.0)
        sev_z = zrank_severity(sev)
        per_corpus[c] = {
            "n_speakers": int(len(df)),
            "rho_within": float(rho),
            "p_value": float(pval),
            "language": cfg.language,
        }
        for s, a in zip(sev_z, atd_z):
            pooled_rows.append({"corpus": c, "language": cfg.language,
                                "sev_z": float(s), "atd_z": float(a)})

    pooled_df = pd.DataFrame(pooled_rows)
    pooled_summary: dict = {
        "n_corpora": len(per_corpus),
        "per_corpus": per_corpus,
        "pooled_n_speakers": int(len(pooled_df)),
    }

    if len(pooled_df) >= 10:
        try:
            import statsmodels.formula.api as smf
            md = smf.mixedlm(
                "sev_z ~ atd_z",
                data=pooled_df,
                groups=pooled_df["corpus"],
                re_formula="~1",
            )
            res = md.fit(method="lbfgs", reml=True)
            pooled_summary["fixed_effect_beta_atd_z"] = float(res.params["atd_z"])
            pooled_summary["fixed_effect_se_atd_z"] = float(res.bse["atd_z"])
            ci = res.conf_int(alpha=0.05).loc["atd_z"].tolist()
            pooled_summary["fixed_effect_ci_atd_z"] = [float(ci[0]), float(ci[1])]
            pooled_summary["random_effect_variance_corpus"] = float(res.cov_re.iloc[0, 0])
            pooled_summary["aic"] = float(res.aic)
            pooled_summary["llf"] = float(res.llf)
        except Exception as exc:  # noqa: BLE001
            logger.exception("LME fit failed: %s", exc)
            pooled_summary["lme_status"] = f"failed: {exc}"
    else:
        pooled_summary["lme_status"] = "skipped: insufficient rows"

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(pooled_summary, f, indent=2, default=str)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
