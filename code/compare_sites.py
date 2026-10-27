# Compare the two site runs of step 6.   Run in Colab after BOTH runs are finished:   %run compare_sites.py
# Needs: outputs6/ (Sylhet)  and  outputs6_sunamganj/ (Sunamganj)
import numpy as np, pandas as pd

A = pd.read_csv("outputs6/stream_drops.csv");            A = A[A.base_rate.astype(str) == "sample"]
B = pd.read_csv("outputs6_sunamganj/stream_drops.csv");  B = B[B.base_rate.astype(str) == "sample"]
RA = pd.read_csv("outputs6/results_all.csv");            RB = pd.read_csv("outputs6_sunamganj/results_all.csv")
key = ["model", "lead_days", "stream"]
M = A.merge(B, on=key, suffixes=("_sylhet", "_sunamganj"))

def summary(d, tag):
    return dict(site=tag, cells=len(d), positive_drop=int((d.d_auc > 0).sum()),
                sig_positive=int(((d.auc_sig) & (d.d_auc > 0)).sum()),
                sig_negative=int(((d.auc_sig) & (d.d_auc < 0)).sum()),
                mean_drop=round(d.d_auc.mean(), 4))
S = pd.DataFrame([summary(A, "sylhet"), summary(B, "sunamganj")])
print("=== Counts per site (hard subset, leave-one-year-out) ===")
print(S.to_string(index=False))

r = np.corrcoef(M.d_auc_sylhet, M.d_auc_sunamganj)[0, 1]
same_sign = float((np.sign(M.d_auc_sylhet) == np.sign(M.d_auc_sunamganj)).mean())
print(f"\nAUC drops, Sylhet vs Sunamganj: correlation r = {r:.2f}; same sign in {100*same_sign:.0f}% of the 32 cells")

full = lambda R: R[R.config == "FULL"].pivot_table(index="model", columns="lead_days", values="auc_hard").round(3)
print("\nAUC of the FULL model, Sylhet:\n", full(RA), "\n\nAUC of the FULL model, Sunamganj:\n", full(RB))
print("\nSeason-only AUC (does not depend on the site):", RA[RA.config == 'SEASON'].groupby('model').auc_hard.first().round(3).to_dict())

M.to_csv("site_sensitivity_cells.csv", index=False); S.to_csv("site_sensitivity_summary.csv", index=False)
pd.DataFrame([dict(corr=r, same_sign=same_sign)]).to_csv("site_sensitivity_agreement.csv", index=False)
print("\nSaved: site_sensitivity_cells.csv, site_sensitivity_summary.csv, site_sensitivity_agreement.csv")
