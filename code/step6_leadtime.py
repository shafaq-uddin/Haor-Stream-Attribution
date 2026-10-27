# Step 6: lead-time version of the 77-event study (event-anchored design).
#
# IDEA: each of the 77 events has a date t0 and a flood label (from reports, independent of the inputs).
# For lead time h = 1..4 days we ask: "if the model may only use data up to day t0-h,
# how well can it tell flood from no-flood, and which data stream matters?"
#
# WARNING: inputs are reanalysis (ERA5 / GloFAS), i.e. what really happened up to t0-h.
# This is an UPPER BOUND for a real forecast: there is no forecast error in it.
#
# Run in Google Colab (needs internet for Open-Meteo):
#     !pip -q install pandas numpy scikit-learn matplotlib requests
#     upload events.csv, then:  %run step6_leadtime.py
# Test without internet:  set SYNTHETIC = True
import os, time, warnings
import numpy as np, pandas as pd, requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ settings
EVENTS   = "events.csv"
SYNTHETIC = False                   # True = fake daily data, only to test the code
# Which haor point to use for LOCAL rain / soil moisture. Default = near Sylhet city (the main run).
# For the sensitivity run, in Colab BEFORE running this script, execute:
#     import os; os.environ["HAOR_SITE"] = "sunamganj"
SITE = os.environ.get("HAOR_SITE", "sylhet").lower()
SITES = {"sylhet": (24.89, 91.87), "sunamganj": (25.07, 91.40)}
assert SITE in SITES, f"HAOR_SITE must be one of {list(SITES)}"
TARGET   = SITES[SITE]
UP_RAIN  = (25.28, 91.72)            # Meghalaya foothills          - check/adjust
UP_RIVER = (24.83, 92.79)            # Barak river upstream (GloFAS)- check/adjust
START, END = "2014-10-01", "2024-10-31"
LEADS    = [1, 2, 3, 4]              # days
ALPHAS   = np.round(np.arange(0.05, 0.31, 0.05), 2)   # PLACEHOLDER cost/loss band
BASE_RATES = [None, 0.10]            # None = sample rate (42% flood, by design!); 0.10 = what-if
B_BOOT   = 300
CACHE = "cache6"                                      # shared: upstream series are identical for both sites
OUT = "outputs6" if SITE == "sylhet" else f"outputs6_{SITE}"   # results never overwrite each other
TARGET_NAME = "target" if SITE == "sylhet" else f"target_{SITE}"  # keeps the old Sylhet cache valid
os.makedirs(OUT, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
print(f"SITE = {SITE}  TARGET = {TARGET}  -> results in {OUT}/")

# ------------------------------------------------------------------ data
def _get(url, params, tries=5):
    for i in range(tries):
        r = requests.get(url, params=params, timeout=90)
        if r.status_code == 200: return r.json()
        if r.status_code == 429 or r.status_code >= 500: time.sleep(15 * (i + 1)); continue
        raise RuntimeError(f"API error {r.status_code}: {r.text[:300]}")
    raise RuntimeError("too many retries")

def year_chunks(a, b):
    s, e, out = pd.Timestamp(a), pd.Timestamp(b), []
    while s <= e:
        ce = min(pd.Timestamp(year=s.year, month=12, day=31), e)
        out.append((s.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d"))); s = ce + pd.Timedelta(days=1)
    return out

def fetch_era5(lat, lon, name):
    path = f"{CACHE}/era5_{name}.csv"
    if os.path.exists(path): return pd.read_csv(path, index_col=0, parse_dates=True)
    rain, soil = [], []
    for a, b in year_chunks(START, END):
        j = _get("https://archive-api.open-meteo.com/v1/archive", dict(
            latitude=lat, longitude=lon, start_date=a, end_date=b,
            daily="precipitation_sum", hourly="soil_moisture_0_to_7cm", timezone="Asia/Dhaka"))
        rain.append(pd.Series(j["daily"]["precipitation_sum"], index=pd.to_datetime(j["daily"]["time"]), dtype=float))
        h = pd.Series(j["hourly"]["soil_moisture_0_to_7cm"], index=pd.to_datetime(j["hourly"]["time"]), dtype=float)
        soil.append(h.resample("D").mean()); time.sleep(0.4)
    df = pd.DataFrame({"rain": pd.concat(rain), "soil": pd.concat(soil)}); df.to_csv(path); return df

def fetch_discharge(lat, lon, name):
    path = f"{CACHE}/glofas_{name}.csv"
    if os.path.exists(path): return pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
    j = _get("https://flood-api.open-meteo.com/v1/flood", dict(
        latitude=lat, longitude=lon, start_date=START, end_date=END, daily="river_discharge"))
    s = pd.Series(j["daily"]["river_discharge"], index=pd.to_datetime(j["daily"]["time"]), dtype=float, name="q")
    s.to_csv(path); return s

def make_synthetic(ev, seed=1):
    """Fake daily series. Storms are planted before flood events so the code has something to find."""
    r = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="D"); n = len(idx)
    season = np.clip(np.sin((idx.dayofyear.values - 100) / 365.25 * 2 * np.pi), 0, None)
    wet = r.random(n) < (0.05 + 0.5 * season)
    rl = pd.Series(r.gamma(0.8, 12.0, n) * wet, index=idx)
    ru = pd.Series(r.gamma(0.8, 14.0, n) * wet, index=idx)
    for t0, lab in zip(ev.date, ev.flood_label):
        if lab == 1:
            rl[t0 - pd.Timedelta(days=int(r.integers(0, 3)))] += r.uniform(30, 90)
            ru[t0 - pd.Timedelta(days=int(r.integers(2, 5)))] += r.uniform(40, 120)
    soil = 0.15 + 0.002 * ru.ewm(span=12).mean()
    q_up = ru.ewm(span=4).mean().shift(1).fillna(0) * 20 + r.normal(0, 5, n)
    return (pd.DataFrame({"rain": rl, "soil": soil}), pd.DataFrame({"rain": ru, "soil": soil}),
            pd.Series(q_up, index=idx))

ev = pd.read_csv(EVENTS, parse_dates=["date"])[["event_id", "date", "flood_label"]]
ev["year"] = ev.date.dt.year
y = ev.flood_label.values
if SYNTHETIC:
    loc, up, qup = make_synthetic(ev)
else:
    loc = fetch_era5(*TARGET, TARGET_NAME); up = fetch_era5(*UP_RAIN, "uprain")
    qup = fetch_discharge(*UP_RIVER, "upriver")
print("daily data:", loc.index.min().date(), "->", loc.index.max().date())

# ------------------------------------------------------------------ features
STREAMS = ["S1_local_rain", "S2_upstream_rain", "S3_soil_moisture", "S4_upstream_flow"]

def raw_features(h):
    """Features for every event, using ONLY data up to day t0-h (inclusive)."""
    rows = []
    for t0 in ev.date:
        c = t0 - pd.Timedelta(days=h)                      # last day we are allowed to see
        f = {}
        for nm, df in (("rl", loc), ("ru", up)):
            r = df["rain"]
            f[f"{nm}_1"] = r.loc[c - pd.Timedelta(days=0):c].sum()
            f[f"{nm}_3"] = r.loc[c - pd.Timedelta(days=2):c].sum()
            f[f"{nm}_7"] = r.loc[c - pd.Timedelta(days=6):c].sum()
        s = loc["soil"]
        f["sm"] = s.get(c, np.nan); f["sm_d3"] = s.get(c, np.nan) - s.get(c - pd.Timedelta(days=3), np.nan)
        q = qup.loc[c - pd.Timedelta(days=2):c]
        f["qu"] = qup.get(c, np.nan); f["qu_d1"] = qup.get(c, np.nan) - qup.get(c - pd.Timedelta(days=1), np.nan)
        f["qu_max3"] = q.max()
        doy = t0.dayofyear                                  # season of the event date (known in advance)
        f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25); f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
        rows.append(f)
    return pd.DataFrame(rows)

COLS = {"S1_local_rain": ["rl_1", "rl_3", "rl_7"], "S2_upstream_rain": ["ru_1", "ru_3", "ru_7"],
        "S3_soil_moisture": ["sm", "sm_d3"], "S4_upstream_flow": ["qu", "qu_d1", "qu_max3"]}
SEASON = ["doy_sin", "doy_cos"]
month = ev.date.dt.month
hard = (~month.isin([10, 11, 12, 1, 2])).values            # drop the easy dry-season events

def cfg_cols(cfg):
    if cfg == "SEASON": return SEASON
    if cfg == "FULL": return SEASON + sum(COLS.values(), [])
    return SEASON + sum((v for k, v in COLS.items() if k != cfg.replace("-", "")), [])

CONFIGS = ["SEASON", "FULL"] + ["-" + s for s in STREAMS]

# ------------------------------------------------------------------ models / CV
def make(kind):
    if kind == "logit":
        return make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=2, random_state=0, n_jobs=-1)

def oof(X, kind):
    """Leave-one-YEAR-out out-of-fold probabilities."""
    p = np.full(len(X), np.nan)
    for g in np.unique(ev.year):
        te = (ev.year == g).values; tr = ~te
        med = X[tr].median()
        m = make(kind).fit(X[tr].fillna(med), y[tr])
        p[te] = m.predict_proba(X[te].fillna(med))[:, 1]
    return p

def rev_curve(yy, p, s):
    """Relative Economic Value for alpha=C/L, warn if p>=alpha. s = flood base rate."""
    pos, neg = yy == 1, yy == 0
    out = []
    for a in ALPHAS:
        w = p >= a
        H = w[pos].mean(); F = w[neg].mean()
        Ef = s * H * a + s * (1 - H) + (1 - s) * F * a
        Ec = min(a, s); Ep = s * a
        out.append((Ec - Ef) / (Ec - Ep) if Ec > Ep else np.nan)
    return np.array(out)

def auc_pr(yy, p):
    return roc_auc_score(yy, p), average_precision_score(yy, p)

# ------------------------------------------------------------------ main loop
rows, P = [], {}
for kind in ["logit", "rf"]:
    for h in LEADS:
        Xall = raw_features(h)
        for cfg in CONFIGS:
            p = oof(Xall[cfg_cols(cfg)], kind); P[(kind, h, cfg)] = p
            a, pr = auc_pr(y[hard], p[hard])
            r = dict(model=kind, lead_days=h, config=cfg, auc_hard=a, prauc_hard=pr)
            for s in BASE_RATES:
                sv = y[hard].mean() if s is None else s
                r[f"REV_mean_s{'sample' if s is None else s}"] = np.nanmean(rev_curve(y[hard], p[hard], sv))
            rows.append(r)
res = pd.DataFrame(rows); res.to_csv(f"{OUT}/results_all.csv", index=False)

# ------------------------------------------------------------------ ablation drops with year-cluster bootstrap
rng = np.random.default_rng(0)
yrs = ev.year.values
uy = np.unique(yrs[hard])
def boot_drop(kind, h, st, s):
    pf, pa = P[(kind, h, "FULL")], P[(kind, h, "-" + st)]
    d_auc, d_rev = [], []
    for _ in range(B_BOOT):
        pick = rng.choice(uy, len(uy), replace=True)
        idx = np.concatenate([np.where((yrs == k) & hard)[0] for k in pick])
        if len(np.unique(y[idx])) < 2: continue
        d_auc.append(roc_auc_score(y[idx], pf[idx]) - roc_auc_score(y[idx], pa[idx]))
        d_rev.append(np.nanmean(rev_curve(y[idx], pf[idx], s)) - np.nanmean(rev_curve(y[idx], pa[idx], s)))
    return np.percentile(d_auc, [2.5, 97.5]), np.percentile(d_rev, [2.5, 97.5])

drops = []
for kind in ["logit", "rf"]:
    for h in LEADS:
        for st in STREAMS:
            for s in BASE_RATES:
                sv = y[hard].mean() if s is None else s
                pf, pa = P[(kind, h, "FULL")], P[(kind, h, "-" + st)]
                dA = auc_pr(y[hard], pf[hard])[0] - auc_pr(y[hard], pa[hard])[0]
                dR = np.nanmean(rev_curve(y[hard], pf[hard], sv)) - np.nanmean(rev_curve(y[hard], pa[hard], sv))
                (a_lo, a_hi), (r_lo, r_hi) = boot_drop(kind, h, st, sv)
                drops.append(dict(model=kind, lead_days=h, stream=st, base_rate=("sample" if s is None else s),
                                  d_auc=dA, d_auc_lo=a_lo, d_auc_hi=a_hi, d_rev=dR, d_rev_lo=r_lo, d_rev_hi=r_hi,
                                  auc_sig=(a_lo > 0) or (a_hi < 0), rev_sig=(r_lo > 0) or (r_hi < 0)))
dr = pd.DataFrame(drops); dr.to_csv(f"{OUT}/stream_drops.csv", index=False)

# ------------------------------------------------------------------ report
pd.set_option("display.width", 220)
print(f"\nevents={len(ev)}  hard subset n={hard.sum()} (flood={y[hard].sum()})")
print("\nAUC on the hard subset (leave-one-year-out). Compare each row with SEASON:")
print(res.pivot_table(index=["model", "config"], columns="lead_days", values="auc_hard").round(3))
print("\nAUC drop when a stream is removed (positive = stream helps). * = 95% interval excludes 0")
v = dr[dr.base_rate == "sample"].copy()
v["cell"] = v.d_auc.round(3).astype(str) + np.where(v.auc_sig, "*", "")
print(v.pivot_table(index=["model", "stream"], columns="lead_days", values="cell", aggfunc="first"))
print("\nREV drop (mean over alpha band), base rate = sample rate. * = interval excludes 0")
v["cellr"] = v.d_rev.round(3).astype(str) + np.where(v.rev_sig, "*", "")
print(v.pivot_table(index=["model", "stream"], columns="lead_days", values="cellr", aggfunc="first"))

# ------------------------------------------------------------------ figures
for kind in ["logit", "rf"]:
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for a_, col, title in ((ax[0], "d_auc", "AUC drop"), (ax[1], "d_rev", "REV drop")):
        M = dr[(dr.model == kind) & (dr.base_rate == "sample")].pivot(index="stream", columns="lead_days", values=col)
        M = M.reindex(STREAMS); im = a_.imshow(M.values, cmap="viridis", aspect="auto")
        a_.set_xticks(range(len(LEADS))); a_.set_xticklabels([f"{h}d" for h in LEADS]); a_.set_yticks(range(4))
        a_.set_yticklabels(STREAMS, fontsize=8); a_.set_title(f"{title} when stream removed ({kind})", fontsize=9)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]): a_.text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        plt.colorbar(im, ax=a_)
    plt.tight_layout(); plt.savefig(f"{OUT}/fig_drops_{kind}.png", dpi=150); plt.close()
print(f"\nSaved tables and figures in {OUT}/")
