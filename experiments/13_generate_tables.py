"""
Experiment 13 — Generate paper_numbers.json and all LaTeX tables.

Usage:
    python experiments/13_generate_tables.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.seed_utils import set_global_seed  # noqa: E402


def generate_paper_numbers() -> dict:
    """Extract every key number used in the paper."""
    print("=" * 60)
    print("  Task 8.1 — Generating paper_numbers.json")
    print("=" * 60)

    atd = pd.read_csv(PROJECT_ROOT / "results/atd_scores.csv")
    speaker_atd = pd.read_csv(PROJECT_ROOT / "results/speaker_atd.csv")
    validation = json.load(open(PROJECT_ROOT / "results/atd_validation_report.json"))
    layers = pd.read_csv(PROJECT_ROOT / "results/tables/layer_analysis.csv")
    table1 = pd.read_csv(PROJECT_ROOT / "results/tables/table1_main_results.csv")
    ablations = pd.read_csv(PROJECT_ROOT / "results/tables/table2_ablations.csv")
    loso = pd.read_csv(PROJECT_ROOT / "results/tables/loso_results.csv")
    confound = json.load(open(PROJECT_ROOT / "results/atd_confound_analysis.json"))
    metadata = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")
    ref_manifest = json.load(open(PROJECT_ROOT / "data/processed/references/reference_manifest.json"))

    valid_atd = atd.dropna(subset=["atd_score"])

    # ── Dataset stats ──
    dur_col = "duration_sec" if "duration_sec" in metadata.columns else "duration"
    total_hours = round(float(metadata[dur_col].sum()) / 3600, 2) if dur_col in metadata.columns else 0.0

    n_ref_words = len(ref_manifest.get("words_with_references", []))
    n_total_refs = ref_manifest.get("total_references", 0)

    # ── Severity group means ──
    ctrl_mean = float(speaker_atd[speaker_atd.severity_score == 0].mean_atd.mean())
    mild_mean = float(speaker_atd[speaker_atd.severity_score == 1].mean_atd.mean())
    severe_mean = float(speaker_atd[speaker_atd.severity_score == 2].mean_atd.mean())

    # ── Table 1 numbers ──
    def t1_val(method_substr, col):
        row = table1[table1.method.str.contains(method_substr, regex=False)]
        if row.empty:
            return 0.0
        return float(row[col].values[0])

    atd_all_rmse = t1_val("ATD-All", "rmse_loso")
    mfcc_rmse = t1_val("MFCC", "rmse_loso")
    atd_all_loso = t1_val("ATD-All", "spearman_loso")
    mfcc_loso = t1_val("MFCC", "spearman_loso")

    rmse_improv = round((mfcc_rmse - atd_all_rmse) / mfcc_rmse * 100, 1) if mfcc_rmse > 0 else 0.0
    loso_mult = round(atd_all_loso / max(abs(mfcc_loso), 0.001), 1)

    # ── Ablation numbers ──
    rho_col = "spearman_global"

    def abl_val(substr):
        row = ablations[ablations.variant.str.contains(substr, regex=False)]
        if row.empty:
            return 0.0
        return float(row[rho_col].values[0])

    # ── LOSO ──
    n_correct = int(loso.correct_rank.sum())
    n_tested = int(len(loso))

    # ── Confound ──
    pearson_r = float(confound.get("atd_duration_pearson_r", 0.0))

    # ── Anomalies ──
    def spk_atd(spk):
        row = speaker_atd[speaker_atd.speaker_id == spk]
        return round(float(row.mean_atd.values[0]), 1) if len(row) > 0 else 0.0

    # ── Cohen's d ──
    cohens_d = float(validation.get("cohens_d_mild_severe", validation.get("cohens_d", 0.0)))
    cohens_d_cs = float(validation.get("cohens_d_control_severe", 0.0))

    # ── Best layer ──
    best_layer = int(layers.loc[layers.spearman_rho.idxmax(), "layer_idx"])
    best_rho = round(float(layers.spearman_rho.max()), 3)

    numbers = {
        "dataset": {
            "total_speakers": int(metadata.speaker_id.nunique()),
            "dysarthric_speakers": 8,
            "control_speakers": 7,
            "severe_speakers": 5,
            "mild_mod_speakers": 3,
            "total_utterances": int(len(metadata)),
            "valid_atd_utterances": int(len(valid_atd)),
            "total_reference_words": n_ref_words,
            "total_reference_trajectories": n_total_refs,
            "total_hours": total_hours,
        },
        "atd_validation": {
            "speaker_spearman_rho": round(float(validation["speaker_spearman"]), 3),
            "speaker_spearman_p": round(float(validation["p_value"]), 4),
            "utterance_spearman_rho": round(float(validation["utterance_spearman"]), 3),
            "cohens_d_mild_severe": round(cohens_d, 3),
            "cohens_d_control_severe": round(cohens_d_cs, 3),
            "control_mean_atd": round(ctrl_mean, 1),
            "mild_mean_atd": round(mild_mean, 1),
            "severe_mean_atd": round(severe_mean, 1),
            "monotonic_ordering": bool(ctrl_mean < mild_mean < severe_mean),
        },
        "layer_analysis": {
            "best_layer": best_layer,
            "best_layer_rho": best_rho,
            "layer9_rho": round(float(layers[layers.layer_idx == 9].spearman_rho.values[0]), 3),
            "all_layers_significant": bool((layers.p_value < 0.01).all()),
            "n_layers_tested": int(len(layers)),
        },
        "main_results": {
            "atd_l9_rmse": round(t1_val("ATD-L9", "rmse_loso"), 3),
            "atd_all_rmse": round(atd_all_rmse, 3),
            "mfcc_rmse": round(mfcc_rmse, 3),
            "ssl_pool_rmse": round(t1_val("WavLM", "rmse_loso"), 3),
            "atd_l9_loso_rho": round(t1_val("ATD-L9", "spearman_loso"), 3),
            "atd_all_loso_rho": round(atd_all_loso, 3),
            "mfcc_loso_rho": round(mfcc_loso, 3),
            "ssl_pool_loso_rho": round(t1_val("WavLM", "spearman_loso"), 3),
            "atd_l9_global_rho": round(t1_val("ATD-L9", "spearman_global"), 3),
            "atd_all_global_rho": round(t1_val("ATD-All", "spearman_global"), 3),
            "mfcc_global_rho": round(t1_val("MFCC", "spearman_global"), 3),
            "rmse_improvement_pct": rmse_improv,
            "loso_rho_improvement_x": loso_mult,
        },
        "loso": {
            "total_speakers_tested": n_tested,
            "correctly_ranked": n_correct,
            "rank_accuracy_pct": round(n_correct / max(n_tested, 1) * 100, 1),
        },
        "ablations": {
            "full_atd_l1_rho": round(abl_val("Layer 1"), 3),
            "full_atd_l9_rho": round(abl_val("Layer 9"), 3),
            "no_dtw_rho": round(abl_val("No DTW"), 3),
            "no_pca_rho": round(abl_val("No PCA"), 3),
            "mean_pool_rho": round(abl_val("Mean-pool"), 3),
            "mfcc_trajectory_rho": round(abl_val("MFCC"), 3),
            "dtw_contribution": round(abl_val("Layer 9") - abl_val("No DTW"), 3),
        },
        "confound": {
            "atd_duration_pearson_r": round(pearson_r, 3),
            "is_rate_confounded": bool(abs(pearson_r) > 0.5),
        },
        "anomalies": {
            "F04_mean_atd": spk_atd("F04"),
            "F04_severity": 2,
            "M05_mean_atd": spk_atd("M05"),
            "M05_severity": 1,
        },
    }

    out_path = PROJECT_ROOT / "results/paper_numbers.json"
    with open(out_path, "w") as f:
        json.dump(numbers, f, indent=2)

    print(json.dumps(numbers, indent=2))
    print(f"\n  Saved: {out_path}")
    return numbers


def generate_latex_tables():
    """Generate 3 publication-ready LaTeX tables."""
    print(f"\n{'='*60}")
    print(f"  Task 8.2 — Generating LaTeX Tables")
    print(f"{'='*60}")

    tex_dir = PROJECT_ROOT / "paper/tables"
    tex_dir.mkdir(parents=True, exist_ok=True)

    # ── TABLE 1: Main Results ──────────────────────────────────────────
    t1 = pd.read_csv(PROJECT_ROOT / "results/tables/table1_main_results.csv")

    # Find best values
    best_rmse = t1["rmse_loso"].min()
    best_global = t1["spearman_global"].max()
    best_loso = t1["spearman_loso"].max()
    w1_col = "within_1_loso" if "within_1_loso" in t1.columns else None
    best_w1 = t1[w1_col].max() if w1_col else None

    def bold_if(val, best, fmt=".3f", higher_better=True):
        s = f"{val:{fmt}}"
        if higher_better:
            return f"\\textbf{{{s}}}" if val == best else s
        else:
            return f"\\textbf{{{s}}}" if val == best else s

    tex1 = r"""\begin{table}[t]
\centering
\caption{Severity prediction performance on TORGO. LOSO = Leave-One-Speaker-Out
across 8 dysarthric speakers. $\rho$ = Spearman rank correlation. Best results \textbf{bolded}.}
\label{tab:main_results}
\begin{tabular}{lcccc}
\toprule
Method & RMSE$\downarrow$ & $\rho_\text{global}\uparrow$ & $\rho_\text{LOSO}\uparrow$"""
    if w1_col:
        tex1 += r" & Within-1$\uparrow$"
    tex1 += r""" \\
\midrule
"""

    for i, row in t1.iterrows():
        method = row["method"]
        # Add (ours) tag for ATD methods
        if "ATD" in method:
            method_tex = method.replace("ATD-L9", r"ATD-L\textit{k}") + " (ours)"
        else:
            method_tex = method

        rmse_s = bold_if(row["rmse_loso"], best_rmse, higher_better=False)
        global_s = bold_if(row["spearman_global"], best_global)
        loso_s = bold_if(row["spearman_loso"], best_loso)

        # Handle negative values with proper minus sign
        if row["spearman_global"] < 0:
            global_s = global_s.replace("-", "$-$")
        if row["spearman_loso"] < 0:
            loso_s = loso_s.replace("-", "$-$")

        line = f"{method_tex} & {rmse_s} & {global_s} & {loso_s}"
        if w1_col:
            w1_val = row[w1_col]
            w1_s = f"{w1_val*100:.1f}\\%" if w1_val <= 1 else f"{w1_val:.1f}\\%"
            if best_w1 is not None and w1_val == best_w1:
                w1_s = f"\\textbf{{{w1_s}}}"
            line += f" & {w1_s}"
        line += r" \\"

        # Add midrule between baselines and our methods
        if i == 1:
            line += "\n\\midrule"
        tex1 += line + "\n"

    tex1 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(tex_dir / "table1_main_results.tex", "w") as f:
        f.write(tex1)
    print(f"  Saved: table1_main_results.tex")

    # ── TABLE 2: Layer Analysis ────────────────────────────────────────
    layers = pd.read_csv(PROJECT_ROOT / "results/tables/layer_analysis.csv")
    best_layer_rho = layers.spearman_rho.max()

    tex2 = r"""\begin{table}[t]
\centering
\caption{Speaker-level Spearman $\rho$ between ATD and severity score across
WavLM-Large transformer layers ($n=15$ speakers). All correlations significant at $p < 0.01$.}
\label{tab:layer_analysis}
\begin{tabular}{ccc}
\toprule
Layer & Spearman $\rho$ & $p$-value \\
\midrule
"""
    for _, row in layers.iterrows():
        layer = int(row["layer_idx"])
        rho = row["spearman_rho"]
        pval = row["p_value"]

        rho_s = f"{rho:.3f}"
        if rho == best_layer_rho:
            rho_s = f"\\textbf{{{rho_s}}}"

        p_s = f"{pval:.4f}" if pval >= 0.0001 else f"{pval:.1e}"
        tex2 += f"{layer} & {rho_s} & {p_s} \\\\\n"

    tex2 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(tex_dir / "table2_layer_analysis.tex", "w") as f:
        f.write(tex2)
    print(f"  Saved: table2_layer_analysis.tex")

    # ── TABLE 3: Ablation Study ────────────────────────────────────────
    abl = pd.read_csv(PROJECT_ROOT / "results/tables/table2_ablations.csv")
    best_abl_rho = abl["spearman_global"].max()

    tex3 = r"""\begin{table}[t]
\centering
\caption{Ablation study. Each row removes or modifies one component of ATD.
$\rho_\text{global}$ uses all 15 speakers; $\rho_\text{LOSO}$ uses only 8 dysarthric speakers.}
\label{tab:ablations}
\begin{tabular}{lccc}
\toprule
Variant & $\rho_\text{global}$ & $\rho_\text{LOSO}$ & Notes \\
\midrule
"""
    for _, row in abl.iterrows():
        variant = row["variant"].replace("—", "--")
        rho_g = f"{row['spearman_global']:.3f}"
        if row["spearman_global"] == best_abl_rho:
            rho_g = f"\\textbf{{{rho_g}}}"

        rho_l = f"{row['spearman_loso']:.3f}" if pd.notna(row.get("spearman_loso")) else "--"
        notes = str(row.get("notes", "")).replace("_", r"\_").replace("&", r"\&")
        tex3 += f"{variant} & {rho_g} & {rho_l} & {notes} \\\\\n"

    tex3 += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(tex_dir / "table3_ablations.tex", "w") as f:
        f.write(tex3)
    print(f"  Saved: table3_ablations.tex")


def verify_phase8():
    """Verification gate."""
    print(f"\n{'='*60}")
    print(f"  Phase 8 Verification Gate")
    print(f"{'='*60}")

    # Check nulls
    numbers = json.load(open(PROJECT_ROOT / "results/paper_numbers.json"))

    def find_nulls(d, path=""):
        issues = []
        for k, v in d.items():
            fp = f"{path}.{k}"
            if v is None:
                issues.append(f"NULL: {fp}")
            elif isinstance(v, dict):
                issues.extend(find_nulls(v, fp))
        return issues

    nulls = find_nulls(numbers)
    if nulls:
        print("  ❌ NULL VALUES FOUND:")
        for n in nulls:
            print(f"    {n}")
    else:
        print("  ✅ paper_numbers.json: no null values")

    # Check tables
    all_ok = True
    for tex in ["table1_main_results", "table2_layer_analysis", "table3_ablations"]:
        path = PROJECT_ROOT / "paper/tables" / f"{tex}.tex"
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        ok = exists and size > 100
        print(f"  {'✅' if ok else '❌'} {tex}.tex ({size} bytes)")
        if not ok:
            all_ok = False

    if not nulls and all_ok:
        print(f"\n{'='*60}")
        print(f"  ✅ PHASE 8 COMPLETE")
        print(f"{'='*60}")
    else:
        print(f"\n  ❌ Fix issues above before proceeding")


def main():
    set_global_seed(42)
    generate_paper_numbers()
    generate_latex_tables()
    verify_phase8()


if __name__ == "__main__":
    main()
