import pandas as pd

# CHECK 1
print("\n--- CHECK 1: Layer analysis complete ---")
layers = pd.read_csv("results/tables/layer_analysis.csv")
print(layers.to_string())
best = layers.loc[layers.spearman_rho.idxmax()]
print(f"\n BEST LAYER: {int(best.layer_idx)} (rho={best.spearman_rho:.3f})")

# CHECK 2
print("\n--- CHECK 2: Table 1 complete ---")
results = pd.read_csv("results/tables/table1_main_results.csv")
print(results.to_string())

# CHECK 3
print("\n--- CHECK 3: ATD beats MFCC baseline ---")
atd_rho = results[results.method.str.contains("ATD")]["spearman_rho"].max()
mfcc_rho = results[results.method.str.contains("MFCC")]["spearman_rho"].values[0]
print(f"ATD best rho: {atd_rho:.3f}")
print(f"MFCC rho: {mfcc_rho:.3f}")
print("ATD WINS" if atd_rho > mfcc_rho else "INVESTIGATE")
