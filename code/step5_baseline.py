# Step 5: baseline models on the 77-event haor dataset.
# Honest evaluation = leave-one-YEAR-out (events in one year are related).
# Run:  python step5_baseline.py events.csv
import sys, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
warnings.filterwarnings("ignore")

PATH = sys.argv[1] if len(sys.argv) > 1 else "events.csv"
d = pd.read_csv(PATH, parse_dates=["date"])
d["year"] = d.date.dt.year
doy = d.date.dt.dayofyear
d["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
d["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
d["season"] = np.where(d.date.dt.month.isin([6, 7, 8, 9]), "monsoon",
              np.where(d.date.dt.month.isin([3, 4, 5]), "premonsoon", "dry"))
hard = d.season != "dry"          # the 52 events where the answer is not obvious
y = d.flood_label.values

SEASON = ["doy_sin", "doy_cos"]
STREAMS = {
    "S1_local_rain":    ["rainfall", "rain_24h", "rain_96h"],
    "S2_upstream":      ["upstream_vv"],
    "S3_soil_moisture": ["soil_moisture"],
    "S4_weather":       ["temp", "wind", "temp_anomaly"],
}
SAT = ["VV", "VH", "vv_vh_ratio", "ndwi"]
S1_NO_RAINFALL = ["rain_24h", "rain_96h"]
allS = sum(STREAMS.values(), [])

CONFIGS = {
    "A season only":                 SEASON,
    "B streams only (no season)":    allS,
    "C season + 4 streams":          SEASON + allS,
    "D season + 4 streams, S1 w/o 'rainfall'":
        SEASON + [c for c in allS if c != "rainfall"],
    "E season + 4 streams + satellite (NOT a forecast)": SEASON + allS + SAT,
}

def make(kind):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                  random_state=0, n_jobs=-1)

def prep(X, Xtr):
    # median fill using TRAIN data only (only ndwi has gaps)
    return X.fillna(Xtr.median())

def oof(cols, kind, groups):
    """Out-of-fold probabilities. groups: array of group ids to hold out."""
    p = np.full(len(d), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        Xtr, Xte = d.loc[tr, cols], d.loc[te, cols]
        Xte = prep(Xte, Xtr); Xtr = prep(Xtr, Xtr)
        m = make(kind).fit(Xtr, y[tr])
        p[te] = m.predict_proba(Xte)[:, 1]
    return p

def cluster_boot_auc(p, mask, B=500, seed=0):
    """Bootstrap over YEARS (keeps events of one year together)."""
    rng = np.random.default_rng(seed)
    yrs = d.year.values
    uy = np.unique(yrs[mask])
    out = []
    for _ in range(B):
        pick = rng.choice(uy, len(uy), replace=True)
        idx = np.concatenate([np.where((yrs == k) & mask)[0] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        out.append(roc_auc_score(y[idx], p[idx]))
    return np.percentile(out, [2.5, 97.5])

rows = []
for kind in ["logit", "rf"]:
    for name, cols in CONFIGS.items():
        for cv_name, groups in [("leave-one-year-out", d.year.values),
                                ("leave-one-event-out", np.arange(len(d)))]:
            p = oof(cols, kind, groups)
            r = dict(model=kind, config=name, cv=cv_name,
                     auc_all=roc_auc_score(y, p),
                     auc_hard=roc_auc_score(y[hard], p[hard]),
                     prauc_hard=average_precision_score(y[hard], p[hard]),
                     brier_hard=brier_score_loss(y[hard], p[hard]))
            if cv_name == "leave-one-year-out":
                lo, hi = cluster_boot_auc(p, hard.values)
                r["auc_hard_lo"], r["auc_hard_hi"] = lo, hi
            rows.append(r)

res = pd.DataFrame(rows)
res.to_csv("step5_results.csv", index=False)
pd.set_option("display.width", 200, "display.max_colwidth", 60)
print(f"events={len(d)}  flood={y.sum()}  non-flood={(1-y).sum()}  hard subset n={hard.sum()} "
      f"(flood={y[hard].sum()})")
show = res[res.cv == "leave-one-year-out"].round(3)
print("\n=== leave-one-YEAR-out (honest) ===")
print(show[["model", "config", "auc_all", "auc_hard", "auc_hard_lo", "auc_hard_hi",
            "prauc_hard", "brier_hard"]].to_string(index=False))
print("\n=== leave-one-EVENT-out (what the dataset authors use; more optimistic) ===")
print(res[res.cv == "leave-one-event-out"].round(3)[
    ["model", "config", "auc_all", "auc_hard"]].to_string(index=False))
