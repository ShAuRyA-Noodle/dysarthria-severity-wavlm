#!/usr/bin/env python
"""
Generate the three VERITAS pitfalls-paper figures directly from committed
results/*.json files. Every plotted value is read from disk at run time; nothing
is hard-coded. Run:

    python paper/veritas/figures/make_figures.py

Outputs (PDF for LaTeX + PNG for quick view) into this directory:
    fig1_within_vs_loco.{pdf,png}
    fig2_leakage.{pdf,png}
    fig3_cycle_consistency.{pdf,png}

Palette: Okabe-Ito colorblind-safe qualitative colors.
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# paths
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
RESULTS = os.path.join(REPO, "results")


def load(*parts):
    with open(os.path.join(RESULTS, *parts), "r", encoding="utf-8") as fh:
        return json.load(fh)


# ----------------------------------------------------------------------------
# Okabe-Ito colorblind-safe palette
BLUE = "#0072B2"      # optimistic / naive / within / raw
VERMILION = "#D55E00"  # corrected / cross-corpus / LOCO / partial
ORANGE = "#E69F00"    # energy confound
GREEN = "#009E73"
GRAY = "#7F7F7F"

plt.rcParams.update(
    {
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.7,
        "figure.dpi": 150,
    }
)

# collect every value we actually plot, for the deliverable audit trail
PLOTTED = {}


def bar_labels(ax, bars, fmt="{:+.3f}", dy=0.012, fs=9):
    for b in bars:
        h = b.get_height()
        va = "bottom" if h >= 0 else "top"
        off = dy if h >= 0 else -dy
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + off,
            fmt.format(h),
            ha="center",
            va=va,
            fontsize=fs,
            fontweight="bold",
        )


# ============================================================================
# FIGURE 1 -- within-corpus vs leave-one-corpus-out severity rho (Ridge probe)
# ============================================================================
def fig1():
    d = load("veritas", "cross_corpus_severity.json")
    corpora = ["torgo", "ipvs", "easycall"]
    labels = ["TORGO", "IPVS", "EasyCall"]
    within = [d["within_corpus"][c]["ridge"]["rho"] for c in corpora]
    loco = [d["leave_one_corpus_out"][c]["ridge"]["rho"] for c in corpora]
    loco_p = [d["leave_one_corpus_out"][c]["ridge"]["rho_p"] for c in corpora]
    PLOTTED["fig1_within_ridge"] = dict(zip(corpora, within))
    PLOTTED["fig1_loco_ridge"] = dict(zip(corpora, loco))

    import numpy as np

    x = np.arange(len(corpora))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    b1 = ax.bar(x - w / 2, within, w, label="Within-corpus (leave-one-speaker-out)",
                color=BLUE, edgecolor="black", linewidth=0.5)
    b2 = ax.bar(x + w / 2, loco, w, label="Leave-one-corpus-out (transfer)",
                color=VERMILION, edgecolor="black", linewidth=0.5)
    bar_labels(ax, b1, dy=0.015)
    bar_labels(ax, b2, dy=0.015)

    # mark non-significant transfer bars
    for xi, (v, p) in enumerate(zip(loco, loco_p)):
        if p >= 0.05:
            ax.text(xi + w / 2, v + 0.075, "n.s.", ha="center", va="bottom",
                    fontsize=9, color=VERMILION, fontweight="bold", style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(r"Severity Spearman $\rho$")
    ax.set_ylim(0, 1.0)
    ax.set_title("Within-corpus success does not transfer across corpora/etiologies",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=2,
              frameon=False, fontsize=9.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig1_within_vs_loco.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 2 -- speaker-identity leakage: Layer 1 vs Layer 9, naive vs LOCO ref
# ============================================================================
def fig2():
    d = load("deleak_report_fast.json")["by_layer"]
    import numpy as np

    layers = ["layer1", "layer9"]
    lab = ["Layer 1 (early)", "Layer 9 (mid)"]
    naive = [d[l]["leaked_rho"] for l in layers]
    loco = [d[l]["deleaked_rho"] for l in layers]
    delta = [d[l]["delta_rho_leaked_minus_deleaked"] for l in layers]
    # marginal bootstrap 95% CIs -> asymmetric error bars
    naive_ci = [d[l]["leaked_bootstrap_ci"] for l in layers]
    loco_ci = [d[l]["deleaked_bootstrap_ci"] for l in layers]
    naive_err = np.array([[n - c[0], c[1] - n] for n, c in zip(naive, naive_ci)]).T
    loco_err = np.array([[n - c[0], c[1] - n] for n, c in zip(loco, loco_ci)]).T
    PLOTTED["fig2_naive"] = dict(zip(layers, naive))
    PLOTTED["fig2_loco"] = dict(zip(layers, loco))
    PLOTTED["fig2_delta"] = dict(zip(layers, delta))

    x = np.arange(len(layers))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    b1 = ax.bar(x - w / 2, naive, w, label="Naive reference (leaked)",
                color=BLUE, edgecolor="black", linewidth=0.5,
                yerr=naive_err, capsize=3,
                error_kw=dict(ecolor=GRAY, lw=1.0, alpha=0.8))
    b2 = ax.bar(x + w / 2, loco, w, label="Leave-one-control-out reference",
                color=VERMILION, edgecolor="black", linewidth=0.5,
                yerr=loco_err, capsize=3,
                error_kw=dict(ecolor=GRAY, lw=1.0, alpha=0.8))
    bar_labels(ax, b1, dy=0.015)
    bar_labels(ax, b2, dy=0.015)

    # annotate the paired drop (guard against negative-zero formatting)
    for xi, dl in enumerate(delta):
        if abs(dl) < 1e-6:
            txt = r"$\Delta=0.000$ (robust)"
            col = GREEN
        else:
            txt = rf"$\Delta={-dl:+.3f}$"
            col = VERMILION
        ax.annotate(
            txt,
            xy=(xi, max(naive[xi], loco[xi]) + 0.28),
            ha="center", fontsize=10, fontweight="bold", color=col,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylabel(r"Severity Spearman $\rho$ (TORGO)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Early-layer reference-based severity is inflated by speaker leakage",
                 fontsize=11, fontweight="bold")
    ax.legend(loc="lower center", frameon=False, fontsize=9, ncol=1)
    fig.text(0.5, -0.02,
             "Error bars: marginal bootstrap 95% CI (n=15); the paired within-layer $\\Delta$ is the leakage estimate.",
             ha="center", fontsize=7.5, color=GRAY)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig2_leakage.{ext}"), bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# FIGURE 3 -- cycle-consistency "faithfulness" is an energy artifact
# ============================================================================
def fig3():
    d = load("veritas", "h1_faithfulness_vs_inversion_error.json")
    raw = d["pooled_spearman_plain"]
    partial = d["partial_spearman_plain_ctrl_energy"]
    faith_energy = d["spearman_faith_energy"]
    err_energy = d["spearman_err_energy"]
    n_spk = d["n_speakers_used"]
    n_fr = d["n_frames"]
    PLOTTED["fig3"] = dict(raw=raw, partial=partial,
                           faith_vs_energy=faith_energy, err_vs_energy=err_energy,
                           n_speakers=n_spk, n_frames=n_fr)

    import numpy as np

    vals = [raw, partial, faith_energy, err_energy]
    labs = [
        "raw\n$\\rho$(faith, err)",
        "energy-partialled\n$\\rho$(faith, err | E)",
        "$\\rho$(faith, energy)",
        "$\\rho$(err, energy)",
    ]
    colors = [BLUE, VERMILION, ORANGE, ORANGE]
    x = np.arange(len(vals))
    fig, ax = plt.subplots(figsize=(6.8, 3.9))
    bars = ax.bar(x, vals, 0.62, color=colors, edgecolor="black", linewidth=0.5)
    bars[3].set_alpha(0.72)
    ax.axhline(0, color="black", linewidth=0.9)
    bar_labels(ax, bars, dy=0.012)

    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=9)
    ax.set_ylabel(r"Spearman $\rho$")
    ax.set_ylim(-0.62, 0.36)
    ax.set_title(
        f"Cycle-consistency 'faithfulness' vs. real EMA inversion error\n"
        f"(TORGO controls, n={n_spk} speakers / {n_fr:,} frames)",
        fontsize=10.5, fontweight="bold",
    )

    # group headers at the top (clear of the tall negative bars) + divider
    ax.axvline(1.5, color=GRAY, lw=0.9, ls="--", alpha=0.7)
    ax.text(0.5, 0.31, "faithfulness $\\leftrightarrow$ inversion error",
            ha="center", va="center", fontsize=8.5, color="black", fontweight="bold")
    ax.text(2.5, 0.31, "energy confound (the driver)",
            ha="center", va="center", fontsize=8.5, color=ORANGE, fontweight="bold")
    # arrow: raw association collapses under energy control
    ax.annotate("", xy=(0.98, 0.02), xytext=(0.10, 0.135),
                arrowprops=dict(arrowstyle="->", color=VERMILION, lw=1.4,
                                connectionstyle="arc3,rad=-0.35"))
    ax.text(0.52, 0.175, "collapses to $\\approx$0\nunder energy control",
            ha="center", va="center", fontsize=7.8, color=VERMILION, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(HERE, f"fig3_cycle_consistency.{ext}"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    print("=== plotted values (all read from results/*.json) ===")
    print(json.dumps(PLOTTED, indent=2))
    print("\nWrote fig1_within_vs_loco, fig2_leakage, fig3_cycle_consistency (.pdf + .png) to")
    print(HERE)
