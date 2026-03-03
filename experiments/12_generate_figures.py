"""
Experiment 12 — Generate all 6 publication-quality figures.

Usage:
    python experiments/12_generate_figures.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import (
    fig1_layer_analysis,
    fig2_atd_trajectories,
    fig3_atd_scatter,
    fig4_atd_distributions,
    fig5_results_comparison,
    fig6_speaker_heatmap,
)


def main():
    print("=" * 60)
    print("  Phase 7 — Generating Publication Figures")
    print("=" * 60)

    figures = []

    print("\n[1/6] Layer Analysis...")
    try:
        figures.append(fig1_layer_analysis(PROJECT_ROOT))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n[2/6] ATD Trajectories (UMAP)...")
    try:
        figures.append(fig2_atd_trajectories(PROJECT_ROOT, word="yes"))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n[3/6] ATD vs Severity Scatter...")
    try:
        figures.append(fig3_atd_scatter(PROJECT_ROOT))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n[4/6] ATD Distribution Violins...")
    try:
        figures.append(fig4_atd_distributions(PROJECT_ROOT))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n[5/6] Results Comparison...")
    try:
        figures.append(fig5_results_comparison(PROJECT_ROOT))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n[6/6] Speaker Heatmap...")
    try:
        figures.append(fig6_speaker_heatmap(PROJECT_ROOT))
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()

    # Verification
    print(f"\n{'='*60}")
    print(f"  Phase 7 Verification Gate")
    print(f"{'='*60}")

    expected = [
        "fig1_layer_analysis",
        "fig2_atd_trajectories",
        "fig3_atd_scatter",
        "fig4_atd_distributions",
        "fig5_results_comparison",
        "fig6_speaker_heatmap",
    ]
    all_good = True
    for fig in expected:
        for ext in [".png", ".pdf"]:
            path = PROJECT_ROOT / "paper" / "figures" / f"{fig}{ext}"
            exists = path.exists()
            size_kb = path.stat().st_size / 1024 if exists else 0
            status = "✅" if exists and size_kb > 5 else "❌"
            print(f"  {status} {fig}{ext} ({size_kb:.0f} KB)")
            if not exists or size_kb < 5:
                all_good = False

    if all_good:
        print(f"\n{'='*60}")
        print(f"  ✅ PHASE 7 COMPLETE — All figures generated!")
        print(f"{'='*60}")
    else:
        print(f"\n  ❌ Missing figures — fix errors above")


if __name__ == "__main__":
    main()
