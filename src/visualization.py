"""Publication-quality figure generation for ATD paper (Phase 7)."""

import json
import logging
import pickle
import shutil
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch, FancyArrowPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, linregress, mannwhitneyu

logger = logging.getLogger(__name__)

# ── Global style ───────────────────────────────────────────────────────
STYLE = {
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 300,
}
CB_PALETTE = sns.color_palette("colorblind")

SEVERITY_COLORS = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
GENDER_COLORS = {"M": "#1565C0", "F": "#AD1457"}


def _apply_style(ax):
    """Remove top/right spines and add subtle grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.5, color="gray")


def _save_figure(fig, name: str, project_root: Path):
    """Save as PNG + PDF to results/figures/ and paper/figures/."""
    for d in ["results/figures", "paper/figures"]:
        out = project_root / d
        out.mkdir(parents=True, exist_ok=True)
        fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(out / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    print(f"  Saved: {name}.png + .pdf")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1 — Layer Analysis
# ═══════════════════════════════════════════════════════════════════════
def fig1_layer_analysis(project_root: Path) -> str:
    """Line plot of Spearman ρ across 9 WavLM layers."""
    plt.rcParams.update(STYLE)
    df = pd.read_csv(project_root / "results/tables/layer_analysis.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_style(ax)

    layers = df["layer_idx"].values
    rhos = df["spearman_rho"].values

    # Main line
    ax.plot(layers, rhos, color=CB_PALETTE[0], linewidth=2.5, marker="o",
            markersize=9, markerfacecolor="white", markeredgewidth=2,
            markeredgecolor=CB_PALETTE[0], zorder=5, label="ATD Spearman ρ")

    # Reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.7,
               label="Reference (ρ = 0.5)")
    ax.axhline(y=0.756, color=CB_PALETTE[1], linestyle="--", linewidth=1.2,
               alpha=0.7, label="Phase 4 Global ρ = 0.756")

    # Shade high-performance zone
    ax.axvspan(1, 15, alpha=0.06, color=CB_PALETTE[0])

    # Annotate best layer
    best_idx = df["spearman_rho"].idxmax()
    best_layer = int(df.loc[best_idx, "layer_idx"])
    best_rho = df.loc[best_idx, "spearman_rho"]
    ax.annotate(f"Layer {best_layer}\n(ρ = {best_rho:.3f}, Best)",
                xy=(best_layer, best_rho), xytext=(best_layer + 3, best_rho + 0.04),
                fontsize=10, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    # Annotate Layer 9
    l9_rho = df.loc[df.layer_idx == 9, "spearman_rho"].values[0]
    ax.annotate(f"Layer 9\n(ρ = {l9_rho:.3f})",
                xy=(9, l9_rho), xytext=(9, l9_rho - 0.06),
                fontsize=10, ha="center",
                arrowprops=dict(arrowstyle="->", color="black", lw=1),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.9))

    # Text box
    ax.text(0.03, 0.05, "All layers: p < 0.01\nn = 15 speakers",
            transform=ax.transAxes, fontsize=10,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    ax.set_xlabel("WavLM-Large Transformer Layer")
    ax.set_ylabel("Spearman ρ (ATD vs. Severity Score)")
    ax.set_ylim(0.45, 0.85)
    ax.set_xticks(layers)
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()

    _save_figure(fig, "fig1_layer_analysis", project_root)
    plt.close(fig)
    return "fig1_layer_analysis"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2 — Trajectory Visualization (UMAP)
# ═══════════════════════════════════════════════════════════════════════
def fig2_atd_trajectories(project_root: Path, word: str = "yes") -> str:
    """UMAP visualization of articulatory trajectories."""
    from umap import UMAP

    plt.rcParams.update(STYLE)

    ssl_index = json.load(open(project_root / "data/embeddings/ssl_index.json"))
    ref_manifest = json.load(open(project_root / "data/processed/references/reference_manifest.json"))
    atd_df = pd.read_csv(project_root / "results/atd_scores.csv")

    from src.atd_metric import LAYER_POSITION_MAP, TORGO_SEVERITY
    layer_idx = 9
    layer_pos = LAYER_POSITION_MAP[layer_idx]

    controls = ["FC01", "FC02", "FC03"]
    mild_spk = "F03"
    severe_spk = "F01"

    # Collect trajectories
    groups = {"Control": controls, "Mild": [mild_spk], "Severe": [severe_spk]}
    all_frames = []
    group_meta = []  # (group_name, speaker, frame_idx, n_frames)

    for gname, speakers in groups.items():
        for spk in speakers:
            # Find utterances of this word for this speaker
            for uid, entry in ssl_index.items():
                if entry.get("speaker") != spk:
                    continue
                if entry.get("word") != word:
                    continue
                emb_path = project_root / str(entry.get("wavlm_large", ""))
                if not emb_path.exists():
                    continue
                emb = np.load(str(emb_path))
                traj = emb[layer_pos]  # (T, 1024)
                for fi in range(len(traj)):
                    all_frames.append(traj[fi])
                    group_meta.append((gname, spk, fi, len(traj), uid))
                break  # Just one utterance per speaker

    if not all_frames:
        # Fallback: try another word
        print(f"  No data for word '{word}', trying 'no'...")
        word = "no"
        for gname, speakers in groups.items():
            for spk in speakers:
                for uid, entry in ssl_index.items():
                    if entry.get("speaker") != spk or entry.get("word") != word:
                        continue
                    emb_path = project_root / str(entry.get("wavlm_large", ""))
                    if not emb_path.exists():
                        continue
                    emb = np.load(str(emb_path))
                    traj = emb[layer_pos]
                    for fi in range(len(traj)):
                        all_frames.append(traj[fi])
                        group_meta.append((gname, spk, fi, len(traj), uid))
                    break

    if not all_frames:
        print("  WARNING: Could not find trajectory data. Skipping Fig 2.")
        # Create a placeholder
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.text(0.5, 0.5, "Trajectory data not available for visualization",
                ha="center", va="center", fontsize=14, transform=ax.transAxes)
        _save_figure(fig, "fig2_atd_trajectories", project_root)
        plt.close(fig)
        return "fig2_atd_trajectories"

    all_frames = np.array(all_frames)
    print(f"  UMAP fitting on {len(all_frames)} frames...")

    # UMAP reduction
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = reducer.fit_transform(all_frames)

    # Also load reference trajectory
    ref_key = f"{word}_{layer_idx}"
    word_refs = ref_manifest.get("word_layer_ref", {})
    ref_info = word_refs.get(ref_key)
    ref_coords = None
    if ref_info:
        ref_path = project_root / ref_info["path"]
        if ref_path.exists():
            with open(ref_path, "rb") as f:
                ref_data = pickle.load(f)
            ref_traj = ref_data["reference"]  # (T_ref, 64)
            pca = ref_data.get("pca_model")
            if pca:
                ref_1024 = pca.inverse_transform(ref_traj)
                ref_coords = reducer.transform(ref_1024)

    # Get ATD scores for annotation
    atd_scores = {}
    for spk in controls + [mild_spk, severe_spk]:
        spk_atd = atd_df[(atd_df.speaker_id == spk) & atd_df.atd_score.notna()]
        if len(spk_atd) > 0:
            atd_scores[spk] = spk_atd.atd_score.mean()
    ctrl_atd = np.mean([atd_scores.get(s, 0) for s in controls if s in atd_scores])

    # Plot: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panel_config = [
        ("Control", controls, f"Control Speakers\n(n=3, Severity=0)", f"ATD ≈ {ctrl_atd:.0f}"),
        ("Mild", [mild_spk], f"Mild-Moderate ({mild_spk})\n(Severity=1)", f"ATD = {atd_scores.get(mild_spk, 0):.1f}"),
        ("Severe", [severe_spk], f"Severe ({severe_spk})\n(Severity=2)", f"ATD = {atd_scores.get(severe_spk, 0):.1f}"),
    ]

    for ax, (gname, speakers, title, atd_text) in zip(axes, panel_config):
        _apply_style(ax)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Plot reference in gray
        if ref_coords is not None:
            ax.plot(ref_coords[:, 0], ref_coords[:, 1], color="gray", linewidth=3,
                    alpha=0.4, linestyle="--", label="Reference", zorder=2)

        # Plot each speaker's trajectory
        spk_colors = [CB_PALETTE[i % len(CB_PALETTE)] for i in range(len(speakers))]
        for si, spk in enumerate(speakers):
            mask = [(m[0] == gname and m[1] == spk) for m in group_meta]
            idxs = [i for i, m in enumerate(mask) if m]
            if not idxs:
                continue
            pts = coords_2d[idxs]
            n_frames = len(pts)

            # Temporal gradient coloring
            segments = np.array([[pts[j], pts[j + 1]] for j in range(n_frames - 1)])
            if len(segments) == 0:
                continue
            colors_grad = plt.cm.coolwarm(np.linspace(0, 1, len(segments)))
            lc = LineCollection(segments, colors=colors_grad, linewidths=2, alpha=0.8, zorder=3)
            ax.add_collection(lc)

            # Start and end markers
            ax.scatter(pts[0, 0], pts[0, 1], marker="^", color="#2196F3", s=60, zorder=6, edgecolors="black", linewidths=0.5)
            ax.scatter(pts[-1, 0], pts[-1, 1], marker="s", color="#F44336", s=60, zorder=6, edgecolors="black", linewidths=0.5)

        # ATD annotation
        ax.text(0.05, 0.95, atd_text, transform=ax.transAxes, fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

        ax.set_xlabel("UMAP 1", fontsize=10)
        ax.set_ylabel("UMAP 2", fontsize=10)

    fig.suptitle(f'Articulatory Trajectories in WavLM Latent Space: "{word}"',
                 fontsize=14, fontweight="bold", y=1.02)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color="gray", linewidth=2, linestyle="--", label="Reference"),
        plt.scatter([], [], marker="^", color="#2196F3", s=40, label="Start"),
        plt.scatter([], [], marker="s", color="#F44336", s=40, label="End"),
    ]
    fig.tight_layout()
    _save_figure(fig, "fig2_atd_trajectories", project_root)
    plt.close(fig)
    return "fig2_atd_trajectories"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3 — ATD vs Severity Scatter
# ═══════════════════════════════════════════════════════════════════════
def fig3_atd_scatter(project_root: Path) -> str:
    """Speaker-level ATD vs severity with regression."""
    plt.rcParams.update(STYLE)
    df = pd.read_csv(project_root / "results/speaker_atd.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    _apply_style(ax)

    # Detect gender from speaker_id
    df["gender"] = df["speaker_id"].apply(lambda s: "F" if s.startswith("F") else "M")

    # Jitter severity
    np.random.seed(42)
    jitter = np.random.uniform(-0.08, 0.08, len(df))
    x_jittered = df["severity_score"].values + jitter

    # Scatter with gender colors
    for gender, color in GENDER_COLORS.items():
        mask = df["gender"] == gender
        subset = df[mask]
        jit = x_jittered[mask.values]
        ax.errorbar(jit, subset["mean_atd"], yerr=subset["std_atd"],
                    fmt="none", ecolor=color, alpha=0.3, capsize=4, zorder=2)
        # Special markers for F04 and M05
        for _, row in subset.iterrows():
            idx_pos = df.index.get_loc(row.name)
            marker = "o"
            ms = max(row["n_utterances"] / 15, 30)
            if row["speaker_id"] == "F04":
                marker = "*"
                ms = 200
            elif row["speaker_id"] == "M05":
                marker = "D"
                ms = 100
            ax.scatter(jit[subset.index.get_loc(row.name)],
                       row["mean_atd"], c=color, s=ms, marker=marker,
                       edgecolors="black", linewidths=0.5, zorder=5)

    # Labels
    for _, row in df.iterrows():
        idx_pos = df.index.get_loc(row.name)
        offset = (5, 5)
        if row["speaker_id"] == "F04":
            offset = (10, -15)
        elif row["speaker_id"] == "M05":
            offset = (12, -15)
        ax.annotate(row["speaker_id"],
                    (x_jittered[idx_pos], row["mean_atd"]),
                    textcoords="offset points", xytext=offset,
                    fontsize=8, alpha=0.8)

    # Regression line
    slope, intercept, r_val, p_val, stderr = linregress(
        df["severity_score"], df["mean_atd"])
    x_line = np.linspace(-0.2, 2.2, 100)
    ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=2, alpha=0.6, zorder=4)

    # CI band (approximate)
    n = len(df)
    y_pred = slope * df["severity_score"].values + intercept
    resid_std = np.std(df["mean_atd"].values - y_pred)
    ci = 1.96 * resid_std / np.sqrt(n)
    ax.fill_between(x_line, slope * x_line + intercept - ci,
                     slope * x_line + intercept + ci,
                     alpha=0.08, color="black")

    # Spearman
    rho, pval = spearmanr(df["severity_score"], df["mean_atd"])
    ax.text(0.03, 0.95, f"Spearman ρ = {rho:.3f}\np = {pval:.4f}\nn = {n} speakers",
            transform=ax.transAxes, fontsize=11, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Control\n(n=7)", "Mild-Mod\n(n=3)", "Severe\n(n=5)"])
    ax.set_ylabel("Mean ATD Score")
    ax.set_xlim(-0.3, 2.3)

    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GENDER_COLORS["M"],
                    markersize=8, markeredgecolor="black", label="Male"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=GENDER_COLORS["F"],
                    markersize=8, markeredgecolor="black", label="Female"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="gray",
                    markersize=12, markeredgecolor="black", label="F04 (Anomaly)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, "fig3_atd_scatter", project_root)
    plt.close(fig)
    return "fig3_atd_scatter"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4 — ATD Distribution Violins
# ═══════════════════════════════════════════════════════════════════════
def fig4_atd_distributions(project_root: Path) -> str:
    """Violin + strip plot of utterance-level ATD by severity."""
    plt.rcParams.update(STYLE)
    df = pd.read_csv(project_root / "results/atd_scores.csv")
    valid = df.dropna(subset=["atd_score"])

    fig, ax = plt.subplots(figsize=(8, 7))
    _apply_style(ax)

    # Prepare data
    sev_order = [0, 1, 2]
    palette = [SEVERITY_COLORS[s] for s in sev_order]

    # Violin
    parts = ax.violinplot(
        [valid[valid.severity_score == s]["atd_score"].values for s in sev_order],
        positions=sev_order, showextrema=False, widths=0.7)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(palette[i])
        pc.set_alpha(0.4)
        pc.set_edgecolor("none")

    # Median lines
    for s in sev_order:
        vals = valid[valid.severity_score == s]["atd_score"].values
        med = np.median(vals)
        ax.hlines(med, s - 0.2, s + 0.2, color="white", linewidth=3, zorder=6)
        ax.hlines(med, s - 0.2, s + 0.2, color=palette[s], linewidth=1.5, zorder=7)

    # Strip plot
    for s in sev_order:
        vals = valid[valid.severity_score == s]["atd_score"].values
        jitter = np.random.RandomState(42).uniform(-0.15, 0.15, len(vals))
        ax.scatter(s + jitter, vals, c=palette[s], alpha=0.15, s=3, zorder=3)

    # Counts
    counts = [len(valid[valid.severity_score == s]) for s in sev_order]
    labels = [f"Control\n(n={counts[0]:,})", f"Mild-Mod\n(n={counts[1]:,})", f"Severe\n(n={counts[2]:,})"]
    ax.set_xticks(sev_order)
    ax.set_xticklabels(labels)
    ax.set_ylabel("ATD Score (Utterance Level)")

    # Significance brackets
    def add_bracket(ax, x1, x2, y, text, h=3):
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color="black", linewidth=1)
        ax.text((x1 + x2) / 2, y + h + 1, text, ha="center", fontsize=9)

    ymax = valid["atd_score"].quantile(0.98)
    # Control vs Mild
    ctrl = valid[valid.severity_score == 0]["atd_score"].values
    mild = valid[valid.severity_score == 1]["atd_score"].values
    severe = valid[valid.severity_score == 2]["atd_score"].values

    _, p_cm = mannwhitneyu(ctrl, mild, alternative="less")
    _, p_ms = mannwhitneyu(mild, severe, alternative="less")
    _, p_cs = mannwhitneyu(ctrl, severe, alternative="less")

    add_bracket(ax, 0, 1, ymax + 5, f"p = {p_cm:.1e}" if p_cm > 1e-4 else "p < 0.001")
    add_bracket(ax, 1, 2, ymax + 15, f"p = {p_ms:.3f}" if p_ms > 0.001 else "p < 0.001")
    add_bracket(ax, 0, 2, ymax + 30, f"p = {p_cs:.1e}" if p_cs > 1e-4 else "p < 0.001")

    # Text box
    if len(mild) > 0 and len(severe) > 0:
        from src.atd_metric import _cohens_d
        d = _cohens_d(mild, severe)
        ax.text(0.97, 0.05, f"Cohen's d (Mild vs Severe) = {d:.3f}",
                transform=ax.transAxes, fontsize=10, ha="right",
                bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9))

    fig.tight_layout()
    _save_figure(fig, "fig4_atd_distributions", project_root)
    plt.close(fig)
    return "fig4_atd_distributions"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5 — Results Comparison (Table 1 Visual)
# ═══════════════════════════════════════════════════════════════════════
def fig5_results_comparison(project_root: Path) -> str:
    """Horizontal bar chart comparing baselines."""
    plt.rcParams.update(STYLE)
    df = pd.read_csv(project_root / "results/tables/table1_main_results.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods = df["method"].tolist()
    n = len(methods)
    y_pos = np.arange(n)

    # Colors: gray for baselines, blue for ATD
    colors = ["#9E9E9E" if "ATD" not in m else "#1565C0" for m in methods]

    # ── Left panel: RMSE ──
    _apply_style(ax1)
    rmse_col = "rmse_loso" if "rmse_loso" in df.columns else "rmse"
    rmse_vals = df[rmse_col].values
    bars1 = ax1.barh(y_pos, rmse_vals, color=colors, edgecolor="white", height=0.6)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(methods)
    ax1.set_xlabel("RMSE (Lower = Better)")
    ax1.set_title("RMSE (LOSO)", fontsize=13, fontweight="bold")

    # Value labels
    for i, (v, bar) in enumerate(zip(rmse_vals, bars1)):
        ax1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=10)

    # Improvement arrow
    if n >= 4:
        mfcc_rmse = rmse_vals[0]
        atd_rmse = rmse_vals[-1]
        pct = (mfcc_rmse - atd_rmse) / mfcc_rmse * 100
        ax1.annotate(f"↓ {pct:.0f}%", xy=(atd_rmse, n - 1), xytext=(mfcc_rmse + 0.03, n - 1.5),
                     fontsize=10, fontweight="bold", color="#1565C0",
                     arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    # ── Right panel: LOSO Spearman ──
    _apply_style(ax2)
    rho_col = "spearman_loso" if "spearman_loso" in df.columns else "spearman_rho"
    rho_vals = df[rho_col].values
    bars2 = ax2.barh(y_pos, rho_vals, color=colors, edgecolor="white", height=0.6)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(methods)
    ax2.set_xlabel("Spearman ρ — LOSO (Higher = Better)")
    ax2.set_title("Spearman ρ (LOSO)", fontsize=13, fontweight="bold")
    ax2.axvline(x=0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    for i, (v, bar) in enumerate(zip(rho_vals, bars2)):
        offset = 0.02 if v >= 0 else -0.08
        ax2.text(v + offset, i, f"{v:.3f}", va="center", fontsize=10)

    # Multiplier annotation
    if n >= 4 and rho_vals[0] != 0:
        mult = rho_vals[-1] / max(rho_vals[0], 0.001)
        ax2.annotate(f"{mult:.0f}× higher", xy=(rho_vals[-1], n - 1),
                     xytext=(rho_vals[-1] + 0.05, n - 1.5),
                     fontsize=10, fontweight="bold", color="#1565C0",
                     arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    fig.suptitle("ATD vs. Baseline Methods (LOSO Evaluation, n = 8 Dysarthric Speakers)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.text(0.5, -0.02, "LOSO = Leave-One-Speaker-Out  |  Tests generalization to unseen speakers",
             ha="center", fontsize=10, color="gray")

    fig.tight_layout()
    _save_figure(fig, "fig5_results_comparison", project_root)
    plt.close(fig)
    return "fig5_results_comparison"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6 — Speaker Heatmap
# ═══════════════════════════════════════════════════════════════════════
def fig6_speaker_heatmap(project_root: Path) -> str:
    """Heatmap of per-speaker, per-word ATD profiles."""
    plt.rcParams.update(STYLE)
    from src.atd_metric import TORGO_SEVERITY

    df = pd.read_csv(project_root / "results/atd_scores.csv")
    valid = df.dropna(subset=["atd_score"])

    # Pivot: speaker × word
    pivot = valid.groupby(["speaker_id", "word"])["atd_score"].mean().reset_index()
    pivot_wide = pivot.pivot(index="speaker_id", columns="word", values="atd_score")

    # Top 20 words by variance (most discriminative)
    word_var = pivot_wide.var().dropna().sort_values(ascending=False)
    top_words = word_var.head(20).index.tolist()
    heatmap_data = pivot_wide[top_words]

    # Sort rows by severity then mean ATD
    speaker_sev = pd.DataFrame({
        "speaker_id": list(TORGO_SEVERITY.keys()),
        "severity": list(TORGO_SEVERITY.values()),
    })
    heatmap_data = heatmap_data.join(speaker_sev.set_index("speaker_id"))
    heatmap_data["mean_row"] = heatmap_data[top_words].mean(axis=1)
    heatmap_data = heatmap_data.sort_values(["severity", "mean_row"])

    severity_col = heatmap_data["severity"]
    heatmap_data = heatmap_data[top_words]

    # Sort columns by overall mean ATD
    col_order = heatmap_data.mean().sort_values(ascending=False).index
    heatmap_data = heatmap_data[col_order]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(heatmap_data, cmap="RdYlGn_r", ax=ax, linewidths=0.5,
                linecolor="white", cbar_kws={"label": "ATD Score"},
                xticklabels=True, yticklabels=True)

    # Color row labels by severity
    for i, (spk, sev) in enumerate(zip(heatmap_data.index, severity_col)):
        label = ax.get_yticklabels()[i]
        label.set_color(SEVERITY_COLORS.get(sev, "black"))
        label.set_fontweight("bold" if sev > 0 else "normal")

    # Add severity group lines
    prev_sev = None
    for i, sev in enumerate(severity_col):
        if prev_sev is not None and sev != prev_sev:
            ax.axhline(y=i, color="white", linewidth=3)
        prev_sev = sev

    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("")
    ax.set_title("Per-Speaker Articulatory Deviation Profiles Across Words",
                 fontsize=14, fontweight="bold")

    # Legend for severity colors
    legend_elements = [
        Patch(facecolor=SEVERITY_COLORS[0], label="Control"),
        Patch(facecolor=SEVERITY_COLORS[1], label="Mild-Mod"),
        Patch(facecolor=SEVERITY_COLORS[2], label="Severe"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.15, 1),
              title="Severity", framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, "fig6_speaker_heatmap", project_root)
    plt.close(fig)
    return "fig6_speaker_heatmap"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 7 — Cross-corpus scatter (UA-Speech ATD vs intelligibility)
# ═══════════════════════════════════════════════════════════════════════
INTELL_GROUP_COLORS = {
    "very_low": "#D32F2F",  # <25%
    "low": "#FF9800",       # 25-50%
    "mid": "#FDD835",       # 50-75%
    "high": "#4CAF50",      # >=75%
}


def fig7_cross_corpus_scatter(project_root: Path) -> str:
    """Cross-corpus scatter: UA-Speech ATD vs intelligibility."""
    plt.rcParams.update(STYLE)
    df = pd.read_csv(project_root / "results" / "uaspeech_speaker_atd.csv")

    fig, ax = plt.subplots(figsize=(10, 7))
    _apply_style(ax)

    # Severity groups
    def sev_group(score):
        if score < 25:
            return "very_low"
        elif score < 50:
            return "low"
        elif score < 75:
            return "mid"
        return "high"

    df["sev_group"] = df["intelligibility_score"].apply(sev_group)
    group_labels = {"very_low": "Very Low (<25%)", "low": "Low (25-50%)",
                    "mid": "Mid (50-75%)", "high": "High (≥75%)"}

    # Scatter by group
    for grp, color in INTELL_GROUP_COLORS.items():
        mask = df["sev_group"] == grp
        subset = df[mask]
        if len(subset) == 0:
            continue
        ax.scatter(subset["intelligibility_score"], subset["mean_atd"],
                   c=color, s=120, edgecolors="black", linewidths=0.5,
                   zorder=5, label=group_labels[grp])
        ax.errorbar(subset["intelligibility_score"], subset["mean_atd"],
                    yerr=subset["std_atd"], fmt="none", ecolor=color,
                    alpha=0.3, capsize=4, zorder=2)

    # Speaker labels
    for _, row in df.iterrows():
        ax.annotate(row["speaker_id"],
                    (row["intelligibility_score"], row["mean_atd"]),
                    textcoords="offset points", xytext=(6, 4),
                    fontsize=8, alpha=0.8)

    # Regression line + 95% CI
    x = df["intelligibility_score"].values
    y = df["mean_atd"].values
    slope, intercept, r_val, p_val_reg, stderr = linregress(x, y)
    x_line = np.linspace(0, 100, 200)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "k--", linewidth=2, alpha=0.6, zorder=4)

    # CI band
    n = len(df)
    y_pred = slope * x + intercept
    resid_std = np.std(y - y_pred)
    se = resid_std * np.sqrt(1/n + (x_line - x.mean())**2 / np.sum((x - x.mean())**2))
    ci = 1.96 * se
    ax.fill_between(x_line, y_line - ci, y_line + ci, alpha=0.08, color="black")

    # Spearman
    rho, pval = spearmanr(df["intelligibility_score"], df["mean_atd"])
    textbox = (f"Spearman ρ = {rho:.3f} (p={pval:.4f})\n"
               f"Zero-shot cross-corpus\n"
               f"References: UA-Speech controls\n"
               f"Method designed on TORGO")
    ax.text(0.03, 0.95, textbox, transform=ax.transAxes, fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax.set_xlabel("Intelligibility Score (%)")
    ax.set_ylabel("Mean ATD Score")
    ax.set_title("Cross-Corpus ATD Validation on UA-Speech (Zero-Shot)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(-5, 105)
    ax.legend(loc="lower right", framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, "fig7_cross_corpus_scatter", project_root)
    plt.close(fig)
    return "fig7_cross_corpus_scatter"


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 8 — Combined dataset comparison (TORGO + UA-Speech)
# ═══════════════════════════════════════════════════════════════════════
def fig8_cross_dataset(project_root: Path) -> str:
    """Two-panel figure: TORGO vs UA-Speech ATD results side by side."""
    plt.rcParams.update(STYLE)

    torgo_df = pd.read_csv(project_root / "results" / "speaker_atd.csv")
    ua_df = pd.read_csv(project_root / "results" / "uaspeech_speaker_atd.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)

    # ── Left: TORGO ──────────────────────────────────────────────
    _apply_style(ax1)

    torgo_colors = {0: "#4CAF50", 1: "#FF9800", 2: "#F44336"}
    torgo_labels = {0: "Control", 1: "Mild-Mod", 2: "Severe"}

    np.random.seed(42)
    jitter = np.random.uniform(-0.08, 0.08, len(torgo_df))

    for sev in sorted(torgo_df["severity_score"].unique()):
        mask = torgo_df["severity_score"] == sev
        subset = torgo_df[mask]
        jit = subset["severity_score"].values + jitter[mask.values]
        ax1.scatter(jit, subset["mean_atd"], c=torgo_colors[sev],
                    s=100, edgecolors="black", linewidths=0.5, zorder=5,
                    label=torgo_labels[sev])
        ax1.errorbar(jit, subset["mean_atd"], yerr=subset["std_atd"],
                     fmt="none", ecolor=torgo_colors[sev], alpha=0.3,
                     capsize=4, zorder=2)

    # Speaker labels
    for _, row in torgo_df.iterrows():
        idx = torgo_df.index.get_loc(row.name)
        ax1.annotate(row["speaker_id"],
                     (row["severity_score"] + jitter[idx], row["mean_atd"]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7, alpha=0.8)

    # Regression
    slope_t, int_t, _, _, _ = linregress(torgo_df["severity_score"], torgo_df["mean_atd"])
    x_t = np.linspace(-0.3, 2.3, 50)
    ax1.plot(x_t, slope_t * x_t + int_t, "k--", lw=1.5, alpha=0.5)

    rho_t, p_t = spearmanr(torgo_df["severity_score"], torgo_df["mean_atd"])
    ax1.text(0.03, 0.95, f"TORGO: ρ = {rho_t:.3f}\np = {p_t:.4f}\nn = {len(torgo_df)}",
             transform=ax1.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Control", "Mild-Mod", "Severe"])
    ax1.set_xlabel("Severity Score")
    ax1.set_ylabel("Mean ATD Score")
    ax1.set_title("TORGO Dataset", fontsize=13, fontweight="bold")
    ax1.set_xlim(-0.4, 2.4)
    ax1.legend(loc="lower right", fontsize=9, framealpha=0.9)

    # ── Right: UA-Speech ─────────────────────────────────────────
    _apply_style(ax2)

    for grp, color in INTELL_GROUP_COLORS.items():
        if "severity_group" in ua_df.columns:
            mask = ua_df["severity_group"] == grp
        else:
            def _grp(s):
                if s < 25: return "very_low"
                elif s < 50: return "low"
                elif s < 75: return "mid"
                return "high"
            mask = ua_df["intelligibility_score"].apply(_grp) == grp
        subset = ua_df[mask]
        if len(subset) == 0:
            continue
        grp_label = {"very_low": "Very Low", "low": "Low",
                     "mid": "Mid", "high": "High"}[grp]
        ax2.scatter(subset["intelligibility_score"], subset["mean_atd"],
                    c=color, s=100, edgecolors="black", linewidths=0.5,
                    zorder=5, label=grp_label)
        ax2.errorbar(subset["intelligibility_score"], subset["mean_atd"],
                     yerr=subset["std_atd"], fmt="none", ecolor=color,
                     alpha=0.3, capsize=4, zorder=2)

    # Speaker labels
    for _, row in ua_df.iterrows():
        ax2.annotate(row["speaker_id"],
                     (row["intelligibility_score"], row["mean_atd"]),
                     textcoords="offset points", xytext=(5, 4),
                     fontsize=7, alpha=0.8)

    # Regression
    slope_u, int_u, _, _, _ = linregress(ua_df["intelligibility_score"], ua_df["mean_atd"])
    x_u = np.linspace(-5, 105, 100)
    ax2.plot(x_u, slope_u * x_u + int_u, "k--", lw=1.5, alpha=0.5)

    rho_u, p_u = spearmanr(ua_df["intelligibility_score"], ua_df["mean_atd"])
    ax2.text(0.03, 0.95, f"UA-Speech: ρ = {rho_u:.3f}\np = {p_u:.4f}\nn = {len(ua_df)}",
             transform=ax2.transAxes, fontsize=11, va="top",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray", alpha=0.9))

    ax2.set_xlabel("Intelligibility Score (%)")
    ax2.set_title("UA-Speech Dataset", fontsize=13, fontweight="bold")
    ax2.set_xlim(-5, 105)
    ax2.legend(loc="lower right", fontsize=9, framealpha=0.9)

    fig.suptitle("ATD Across Two Independent Dysarthric Speech Datasets",
                 fontsize=15, fontweight="bold", y=1.02)

    fig.tight_layout()
    _save_figure(fig, "fig8_cross_dataset", project_root)
    plt.close(fig)
    return "fig8_cross_dataset"
