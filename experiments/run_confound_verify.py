"""Run confound analysis and Phase 6 verification gate."""
import json
import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis import speaking_rate_confound_test

print("=" * 60)
print("  Confound Analysis — Speaking Rate")
print("=" * 60)

atd_df = pd.read_csv(PROJECT_ROOT / "results/atd_scores.csv")
metadata_df = pd.read_csv(PROJECT_ROOT / "data/processed/metadata/metadata_unified.csv")

result = speaking_rate_confound_test(atd_df, metadata_df, project_root=PROJECT_ROOT)
print(f"\n  Pearson r (ATD vs duration): {result['atd_duration_pearson_r']:.4f}")
print(f"  p-value: {result['atd_duration_pearson_p']}")
print(f"  Verdict: {result['verdict']}")

# === VERIFICATION GATE ===
print(f"\n{'='*60}")
print(f"  Phase 6 Verification Gate")
print(f"{'='*60}")

# CHECK 1
print("\nCHECK 1: LOSO results")
loso = pd.read_csv(PROJECT_ROOT / "results/tables/loso_results.csv")
print(loso.to_string(index=False))
correct = loso.correct_rank.sum()
print(f"\nRank accuracy: {correct}/{len(loso)} speakers")
check1 = correct >= 5

# CHECK 2
print("\nCHECK 2: Ablation table")
ablations = pd.read_csv(PROJECT_ROOT / "results/tables/table2_ablations.csv")
print(ablations.to_string(index=False))
check2 = len(ablations) >= 6
print(f"\nAblation rows: {len(ablations)} (need >= 6): {'PASS' if check2 else 'FAIL'}")

# CHECK 3
print("\nCHECK 3: Confound analysis")
confound = json.load(open(PROJECT_ROOT / "results/atd_confound_analysis.json"))
r_val = confound["atd_duration_pearson_r"]
print(f"ATD-duration correlation: {r_val:.3f}")
check3 = abs(r_val) < 0.5
print("NOT rate-confounded" if check3 else "INVESTIGATE rate confound")

if check1 and check2 and check3:
    print(f"\n{'='*60}")
    print(f"  PHASE 6 COMPLETE")
    print(f"{'='*60}")
else:
    print(f"\n  Some checks need attention: C1={check1}, C2={check2}, C3={check3}")
