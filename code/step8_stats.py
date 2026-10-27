# Step 8: (A) proper p-values with multiple-test correction, (B) power analysis.
# Run in Colab AFTER step 6 (same folder, same events.csv):   %run step8_stats.py
# It re-uses step 6 (data is cached, so this is fast the second time).
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if "P" not in globals():                       # step 6 results not in memory -> run step 6 now
    exec(open("step6_leadtime.py").read(), globals())

B2, R_SIM, B_SIM = 2000, 60, 100               # bootstrap draws / simulation repeats / bootstrap in simulation
rng = np.random.default_rng(1)

# ======================================================================
# A. For every (model, lead, stream): is "FULL" really better than "FULL minus stream"?
#    Year-cluster bootstrap p-value (two-sided) + Holm correction over all 32 tests.
# ======================================================================
def holm(p):
    p = np.asarray(p); order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i]); adj[i] = min(1, run)
    return adj

yrs = ev.year.values
uy = np.unique(yrs[hard])
idx_by_year = {k: np.where((yrs == k) & hard)[0] for k in uy}
rows = []
for kind in ["logit", "rf"]:
    for h in LEADS:
        for st in STREAMS:
            pf, pa = P[(kind, h, "FULL")], P[(kind, h, "-" + st)]
            obs = roc_auc_score(y[hard], pf[hard]) - roc_auc_score(y[hard], pa[hard])
            d = []
            for _ in range(B2):
                pick = rng.choice(uy, len(uy), replace=True)
                ii = np.concatenate([idx_by_year[k] for k in pick])
                if len(np.unique(y[ii])) < 2: continue
                d.append(roc_auc_score(y[ii], pf[ii]) - roc_auc_score(y[ii], pa[ii]))
            d = np.array(d)
            p_two = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
            rows.append(dict(model=kind, lead_days=h, stream=st, d_auc=obs, p_raw=max(p_two, 1 / len(d))))
T = pd.DataFrame(rows)
T["p_holm"] = holm(T.p_raw.values)
T["survives_holm_5pct"] = T.p_holm < 0.05
T.to_csv(f"{OUT}/stream_pvalues_holm.csv", index=False)
print("\n=== A. Tests after multiple-test correction (Holm, 32 tests) ===")
print(T.sort_values("p_raw").head(8).round(4).to_string(index=False))
print("raw p < 0.05:", int((T.p_raw < 0.05).sum()), "of", len(T),
      "| survive Holm:", int(T.survives_holm_5pct.sum()))
print("(about 5% of 32 = 1.6 raw 'hits' are expected by chance alone)")

# ======================================================================
# B. Power analysis: how many events are needed to detect a stream that really helps?
#    Simulated world: label depends on a known 'season' signal s and one stream x with true effect beta.
#    Test = does adding x improve leave-one-year-out AUC? (same 11 'years' as the real data)
# ======================================================================
def sim_once(N, beta, n_years=11):
    g = rng.integers(0, n_years, N)
    s = rng.normal(size=N); x = rng.normal(size=N)
    z = -0.2 + 1.5 * s + beta * x
    yy = (rng.random(N) < 1 / (1 + np.exp(-z))).astype(int)
    if yy.min() == yy.max(): return None
    X0, X1 = s[:, None], np.c_[s, x]
    p0, p1 = np.full(N, np.nan), np.full(N, np.nan)
    for k in range(n_years):
        te = g == k; tr = ~te
        if te.sum() == 0 or len(np.unique(yy[tr])) < 2: continue
        p0[te] = LogisticRegression(C=0.5).fit(X0[tr], yy[tr]).predict_proba(X0[te])[:, 1]
        p1[te] = LogisticRegression(C=0.5).fit(X1[tr], yy[tr]).predict_proba(X1[te])[:, 1]
    ok = ~np.isnan(p0)
    yy, p0, p1, g = yy[ok], p0[ok], p1[ok], g[ok]
    if len(np.unique(yy)) < 2: return None
    by = {k: np.where(g == k)[0] for k in np.unique(g)}
    d = []
    for _ in range(B_SIM):
        pick = rng.choice(list(by), len(by), replace=True)
        ii = np.concatenate([by[k] for k in pick])
        if len(np.unique(yy[ii])) < 2: continue
        d.append(roc_auc_score(yy[ii], p1[ii]) - roc_auc_score(yy[ii], p0[ii]))
    lo = np.percentile(d, 2.5)
    return lo > 0                                   # "detected" = lower 95% bound above zero

Ns = [40, 77, 150, 300, 600, 1200]
betas = {"weak stream (beta=0.5)": 0.5, "moderate stream (beta=1.0)": 1.0}
pw = []
print("\n=== B. Power analysis (share of simulated studies that detect the stream) ===")
for name, b in betas.items():
    for N in Ns:
        hits = [sim_once(N, b) for _ in range(R_SIM)]
        hits = [h_ for h_ in hits if h_ is not None]
        pw.append(dict(effect=name, N_events=N, power=float(np.mean(hits))))
    print(name, [round(r["power"], 2) for r in pw if r["effect"] == name])
PW = pd.DataFrame(pw); PW.to_csv(f"{OUT}/power_analysis.csv", index=False)
print("N events:", Ns)

fig, ax = plt.subplots(figsize=(5.5, 3.6))
for name in betas:
    q = PW[PW.effect == name]; ax.plot(q.N_events, q.power, marker="o", label=name)
ax.axvline(77, color="gray", ls="--"); ax.text(80, 0.05, "our 77 events", fontsize=8)
ax.axhline(0.8, color="k", lw=0.5, ls=":")
ax.set_xscale("log"); ax.set_xlabel("number of events"); ax.set_ylabel("power (chance of detecting a real stream)")
ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(f"{OUT}/fig_power.png", dpi=150)
print(f"\nSaved: {OUT}/stream_pvalues_holm.csv, power_analysis.csv, fig_power.png")
