import json
import numpy as np
import pandas as pd

atd = pd.read_csv("results/atd_scores.csv").dropna(subset=["atd_score"])
ctrl = atd[atd.severity_score == 0]["atd_score"].values
mild = atd[atd.severity_score == 1]["atd_score"].values
severe = atd[atd.severity_score == 2]["atd_score"].values

def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    pooled = np.sqrt(((n1-1)*g1.var(ddof=1) + (n2-1)*g2.var(ddof=1)) / (n1+n2-2))
    return float((g2.mean() - g1.mean()) / pooled) if pooled > 0 else 0.0

d_ms = cohens_d(mild, severe)
d_cs = cohens_d(ctrl, severe)
print(f"Cohen's d (mild vs severe) = {d_ms:.3f}")
print(f"Cohen's d (ctrl vs severe) = {d_cs:.3f}")

nums = json.load(open("results/paper_numbers.json"))
nums["atd_validation"]["cohens_d_mild_severe"] = round(d_ms, 3)
nums["atd_validation"]["cohens_d_control_severe"] = round(d_cs, 3)
with open("results/paper_numbers.json", "w") as f:
    json.dump(nums, f, indent=2)
print("Updated paper_numbers.json")
