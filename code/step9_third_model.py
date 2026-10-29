# Step 9: robustness check - does the conclusion change with a THIRD model (gradient boosting)?
#
# Why: the paper's main run uses two models (logistic regression, random forest). A reader may ask whether the
# "no stream is resolved" result is an accident of those two. Here we repeat the SAME ablation, the SAME
# leave-one-year-out validation, the SAME year-cluster bootstrap and Holm correction with gradient boosting.
# The settings of the boosting model are fixed in advance and NOT tuned (tuning on 77 events would overfit):
#     GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0)
#
# Run in Google Colab (internet needed for Open-Meteo, as in step 6):
#     !pip -q install pandas numpy scikit-learn matplotlib requests
#     upload: events.csv, step6_leadtime.py, step9_third_model.py
#     (optional, for the joint Holm over all 48 tests) upload stream_pvalues_holm.csv from your step-8 run
#     %run step9_third_model.py
# Test without internet:  import os; os.environ["SYNTHETIC"] = "1"   before %run
import os, glob, warnings
import numpy as np, pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

# ---- re-use the data / feature / validation code of step 6 (everything before its "main loop")
_src = open("step6_leadtime.py").read().split("# ------------------------------------------------------------------ main loop")[0]
exec(_src, globals())

def make(kind):
    if kind == "gb":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, subsample=0.8, random_state=0)
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1)

KIND = "gb"
rows, P = [], {}
for h in LEADS:
    Xall = raw_features(h)
    for cfg in CONFIGS:
        p = oof(Xall[cfg_cols(cfg)], KIND); P[(KIND, h, cfg)] = p
        a, pr = auc_pr(y[hard], p[hard])
        a_all = roc_auc_score(y, p)
        r = dict(model=KIND, lead_days=h, config=cfg, auc_all=a_all, auc_hard=a, prauc_hard=pr)
        for s in BASE_RATES:
            sv = y[hard].mean() if s is None else s
            r[f"REV_mean_s{'sample' if s is None else s}"] = np.nanmean(rev_curve(y[hard], p[hard], sv))
        rows.append(r)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/gb_results_all.csv", index=False)

# ---- ablation drops, year-cluster bootstrap (same method as step 6)
rng = np.random.default_rng(10)
yrs = ev.year.values
uy = np.unique(yrs[hard])
def boot_drop(h, st, s):
    pf, pa = P[(KIND, h, "FULL")], P[(KIND, h, "-" + st)]
    d_auc, d_rev = [], []
    for _ in range(B_BOOT):
        pick = rng.choice(uy, len(uy), replace=True)
        idx = np.concatenate([np.where((yrs == k) & hard)[0] for k in pick])
        if len(np.unique(y[idx])) < 2: continue
        d_auc.append(roc_auc_score(y[idx], pf[idx]) - roc_auc_score(y[idx], pa[idx]))
        d_rev.append(np.nanmean(rev_curve(y[idx], pf[idx], s)) - np.nanmean(rev_curve(y[idx], pa[idx], s)))
    return np.percentile(d_auc, [2.5, 97.5]), np.percentile(d_rev, [2.5, 97.5])

drops = []
for h in LEADS:
    for st in STREAMS:
        for s in BASE_RATES:
            sv = y[hard].mean() if s is None else s
            pf, pa = P[(KIND, h, "FULL")], P[(KIND, h, "-" + st)]
            dA = auc_pr(y[hard], pf[hard])[0] - auc_pr(y[hard], pa[hard])[0]
            dR = np.nanmean(rev_curve(y[hard], pf[hard], sv)) - np.nanmean(rev_curve(y[hard], pa[hard], sv))
            (a_lo, a_hi), (r_lo, r_hi) = boot_drop(h, st, sv)
            drops.append(dict(model=KIND, lead_days=h, stream=st, base_rate=("sample" if s is None else s),
                              d_auc=dA, d_auc_lo=a_lo, d_auc_hi=a_hi, d_rev=dR, d_rev_lo=r_lo, d_rev_hi=r_hi,
                              auc_sig=(a_lo > 0) or (a_hi < 0), rev_sig=(r_lo > 0) or (r_hi < 0)))
dr = pd.DataFrame(drops); dr.to_csv(f"{OUT}/gb_stream_drops.csv", index=False)

# ---- p-values (year-cluster bootstrap, two-sided) and Holm correction (same method as step 8)
def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj
B2 = 2000
idx_by_year = {k: np.where((yrs == k) & hard)[0] for k in uy}
prow = []
for h in LEADS:
    for st in STREAMS:
        pf, pa = P[(KIND, h, "FULL")], P[(KIND, h, "-" + st)]
        obs = roc_auc_score(y[hard], pf[hard]) - roc_auc_score(y[hard], pa[hard]); d = []
        for _ in range(B2):
            pick = rng.choice(uy, len(uy), replace=True); ii = np.concatenate([idx_by_year[k] for k in pick])
            if len(np.unique(y[ii])) < 2: continue
            d.append(roc_auc_score(y[ii], pf[ii]) - roc_auc_score(y[ii], pa[ii]))
        d = np.array(d); p_two = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
        prow.append(dict(model=KIND, lead_days=h, stream=st, d_auc=obs, p_raw=max(p_two, 1 / len(d))))
T = pd.DataFrame(prow); T["p_holm_16"] = holm(T.p_raw.values)

# joint Holm over all 48 tests = 32 (logistic + random forest, from step 8) + 16 (gradient boosting)
cand = [c for c in ["stream_pvalues_holm.csv", f"{OUT}/stream_pvalues_holm.csv", "/content/stream_pvalues_holm.csv"] if os.path.exists(c)]
if cand:
    old = pd.read_csv(cand[0])
    allp = pd.concat([old[["model", "lead_days", "stream", "p_raw"]], T[["model", "lead_days", "stream", "p_raw"]]], ignore_index=True)
    allp["p_holm_48"] = holm(allp.p_raw.values)
    T = T.merge(allp[allp.model == KIND][["lead_days", "stream", "p_holm_48"]], on=["lead_days", "stream"])
    n_joint = len(allp)
else:
    T["p_holm_48"] = np.nan; n_joint = None
    print("NOTE: stream_pvalues_holm.csv not found, so the joint Holm over 48 tests was skipped.")
T.to_csv(f"{OUT}/gb_pvalues_holm.csv", index=False)

# ---- report
pd.set_option("display.width", 200)
v = dr[dr.base_rate == "sample"].copy()
print("\n=== Gradient boosting: AUC on the hard subset (leave-one-year-out) ===")
print(res.pivot_table(index="config", columns="lead_days", values="auc_hard").round(3))
print("\nAll-77 AUC, season only:", round(res[res.config == "SEASON"].auc_all.iloc[0], 3),
      "| all-77 AUC, FULL (by lead):", res[res.config == "FULL"].auc_all.round(3).tolist())
v["cell"] = v.d_auc.round(3).astype(str) + np.where(v.auc_sig, "*", "")
print("\n=== AUC drop when a stream is removed (positive = stream helps). * = 95% interval excludes 0 ===")
print(v.pivot_table(index="stream", columns="lead_days", values="cell", aggfunc="first"))
print(f"\npositive drops: {(v.d_auc > 0).sum()} of {len(v)} | clear positive: {((v.auc_sig) & (v.d_auc > 0)).sum()}"
      f" | clear negative: {((v.auc_sig) & (v.d_auc < 0)).sum()} | expected by chance: {0.05 * len(v):.1f}")
print(f"raw p<0.05: {(T.p_raw < 0.05).sum()} of 16 | survive Holm (16 tests): {(T.p_holm_16 < 0.05).sum()}"
      + (f" | survive Holm (joint {n_joint} tests): {(T.p_holm_48 < 0.05).sum()}" if n_joint else ""))
print(T.sort_values("p_raw").head(5).round(4).to_string(index=False))
print(f"\nSaved in {OUT}/: gb_results_all.csv, gb_stream_drops.csv, gb_pvalues_holm.csv")
