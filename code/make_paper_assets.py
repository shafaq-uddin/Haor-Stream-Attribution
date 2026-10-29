#!/usr/bin/env python3
"""
make_paper_assets.py  -  turns your result files into LaTeX tables and number-macros for paper.tex.

Why: every number in the paper then comes straight from your CSV files, so the text cannot drift
away from the results. Nothing is typed by hand.

Inputs (all optional; a missing file gives a red TODO box in the PDF instead of a table):
    step5_results.csv                 from step5_baseline.py
    outputs6/stream_drops.csv         from step6_leadtime.py
    outputs6/results_all.csv          from step6_leadtime.py
    outputs6/stream_pvalues_holm.csv  from step8_stats.py
    outputs6/power_analysis.csv       from step8_stats.py
    outputs6/gb_*.csv                 from step9_third_model.py (optional robustness check)
    outputs6/fig_drops_logit.png, fig_drops_rf.png, fig_power.png   (figures)

Run:   python make_paper_assets.py          then compile:  pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
"""
import os, sys, shutil
import numpy as np, pandas as pd

OUT6 = "outputs6"
TAB = "tables"
os.makedirs(TAB, exist_ok=True)

NAMES = {"S1_local_rain": "S1 local rain", "S2_upstream_rain": "S2 upstream rain",
         "S3_soil_moisture": "S3 soil moisture", "S4_upstream_flow": "S4 upstream flow"}
MODEL = {"logit": "Logistic", "rf": "Random forest"}

def todo(name, why):
    with open(f"{TAB}/{name}.tex", "w") as f:
        f.write("% placeholder: " + why + "\n")
    open(f"{TAB}/{name}.todo", "w").close()          # tells paper.tex to show a red TODO box
    print(f"  [missing] {name}: {why}")

def write(name, s):
    if os.path.exists(f"{TAB}/{name}.todo"): os.remove(f"{TAB}/{name}.todo")
    with open(f"{TAB}/{name}.tex", "w") as f: f.write(s)
    print(f"  [ok]      {name}")

def fmt(x, nd=2):
    return "--" if pd.isna(x) else f"{x:.{nd}f}"

macros = {}

# ---------------------------------------------------------------- Table 1: baselines (step 5)
if os.path.exists("step5_results.csv"):
    r = pd.read_csv("step5_results.csv")
    y = r[r.cv == "leave-one-year-out"].copy()
    lines = [r"\begin{tabular}{@{}llcc@{}}", r"\toprule",
             r"Model & Inputs & AUC (all 77) & AUC (hard 52) [95\% CI] \\", r"\midrule"]
    keep = ["A season only", "B streams only (no season)", "C season + 4 streams"]
    label = {"A season only": "season only", "B streams only (no season)": "no season, 4 streams",
             "C season + 4 streams": "season + 4 streams"}
    for m in ["logit", "rf"]:
        for k in keep:
            q = y[(y.model == m) & (y.config == k)].iloc[0]
            lines.append(f"{MODEL[m]} & {label[k]} & {fmt(q.auc_all)} & {fmt(q.auc_hard)} "
                         f"[{fmt(q.auc_hard_lo)}, {fmt(q.auc_hard_hi)}] \\\\")
        if m == "logit": lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_baseline", "\n".join(lines) + "\n")
    s = y[(y.config == "A season only")]
    macros["seasonAUCallLo"] = fmt(s.auc_all.min()); macros["seasonAUCallHi"] = fmt(s.auc_all.max())
    macros["seasonAUChardLo"] = fmt(s.auc_hard.min()); macros["seasonAUChardHi"] = fmt(s.auc_hard.max())
else:
    todo("tab_baseline", "step5_results.csv");

# ---------------------------------------------------------------- Table 2 + macros: ablation (step 6)
p = f"{OUT6}/stream_drops.csv"
if os.path.exists(p):
    d = pd.read_csv(p); s = d[d.base_rate.astype(str) == "sample"].copy()
    leads = sorted(s.lead_days.unique())
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(leads) + r"@{}}", r"\toprule",
             "Model & Stream removed & " + " & ".join(f"{h} d" for h in leads) + r" \\", r"\midrule"]
    for m in ["logit", "rf"]:
        for st in NAMES:
            cells = []
            for h in leads:
                q = s[(s.model == m) & (s.stream == st) & (s.lead_days == h)].iloc[0]
                star = r"$^{*}$" if q.auc_sig else ""
                cells.append(f"{q.d_auc:+.3f}{star}")
            lines.append(f"{MODEL[m]} & {NAMES[st]} & " + " & ".join(cells) + r" \\")
        if m == "logit": lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_ablation", "\n".join(lines) + "\n")
    # numbers used in the text
    n = len(s); pos = int((s.d_auc > 0).sum())
    sp = s[(s.auc_sig) & (s.d_auc > 0)]; sn = s[(s.auc_sig) & (s.d_auc < 0)]
    macros.update(nCells=str(n), nPos=str(pos), pctPos=f"{100*pos/n:.0f}",
                  nSigPos=str(len(sp)), nSigNeg=str(len(sn)),
                  expectedChance=f"{0.05*n:.1f}")
    if len(sp):
        q = sp.iloc[0]
        macros["sigPosText"] = (f"{MODEL[q.model]}, {int(q.lead_days)}-day lead, removing {NAMES[q.stream]} "
                                f"(AUC drop {q.d_auc:+.3f}, 95\\% CI [{q.d_auc_lo:.3f}, {q.d_auc_hi:.3f}])")
    else:
        macros["sigPosText"] = "none"
    rs = s[s.rev_sig]
    macros["nSigRev"] = str(len(rs))
    macros["sigRevText"] = "; ".join(
        f"{MODEL[q.model]}, {int(q.lead_days)}-day lead, removing {NAMES[q.stream]} (REV drop {q.d_rev:+.3f}, 95\\% CI [{q.d_rev_lo:.3f}, {q.d_rev_hi:.3f}])"
        for _, q in rs.iterrows()) or "none"
else:
    todo("tab_ablation", p)
    macros.update(nCells="?", nPos="?", pctPos="?", nSigPos="?", nSigNeg="?", expectedChance="?",
                  sigPosText="?", nSigRev="?", sigRevText="?")

# ---------------------------------------------------------------- Table 3: AUC by lead (step 6)
p = f"{OUT6}/results_all.csv"
if os.path.exists(p):
    r = pd.read_csv(p); leads = sorted(r.lead_days.unique())
    lines = [r"\begin{tabular}{@{}ll" + "r" * len(leads) + r"@{}}", r"\toprule",
             "Model & Inputs & " + " & ".join(f"{h} d" for h in leads) + r" \\", r"\midrule"]
    for m in ["logit", "rf"]:
        for cfg, lab in (("SEASON", "season only"), ("FULL", "season + 4 streams")):
            v = [r[(r.model == m) & (r.config == cfg) & (r.lead_days == h)].auc_hard.iloc[0] for h in leads]
            lines.append(f"{MODEL[m]} & {lab} & " + " & ".join(fmt(x, 3) for x in v) + r" \\")
        if m == "logit": lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_lead_auc", "\n".join(lines) + "\n")
else:
    todo("tab_lead_auc", p)

# ---------------------------------------------------------------- Table 4: Holm (step 8)
p = f"{OUT6}/stream_pvalues_holm.csv"
if os.path.exists(p):
    h = pd.read_csv(p); top = h.sort_values("p_raw").head(5)
    lines = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
             r"Model & Stream (lead) & AUC drop & raw $p$ & Holm $p$ & $<$0.05? \\", r"\midrule"]
    for _, q in top.iterrows():
        lines.append(f"{MODEL[q.model]} & {NAMES[q.stream]} ({int(q.lead_days)} d) & {q.d_auc:+.3f} & "
                     f"{q.p_raw:.3f} & {q.p_holm:.3f} & {'yes' if q.survives_holm_5pct else 'no'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_holm", "\n".join(lines) + "\n")
    macros["nRawSig"] = str(int((h.p_raw < 0.05).sum())); macros["nHolmSig"] = str(int(h.survives_holm_5pct.sum()))
    macros["nTests"] = str(len(h))
else:
    todo("tab_holm", p + " (run step8_stats.py on the real data)")
    macros.update(nRawSig="?", nHolmSig="?", nTests="32")

# ---------------------------------------------------------------- Table 5: power (step 8)
p = f"{OUT6}/power_analysis.csv"
if os.path.exists(p):
    w = pd.read_csv(p); Ns = sorted(w.N_events.unique())
    lines = [r"\begin{tabular}{@{}l" + "r" * len(Ns) + r"@{}}", r"\toprule",
             "Stream effect & " + " & ".join(str(int(n)) for n in Ns) + r" \\", r"\midrule"]
    for e in w.effect.unique():
        v = [w[(w.effect == e) & (w.N_events == n)].power.iloc[0] for n in Ns]
        lines.append(f"{e.replace('(beta', '($\\beta$').replace('=', '{=}')} & " + " & ".join(f"{x:.2f}" for x in v) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_power", "\n".join(lines) + "\n")
    m_ = w[w.effect.str.contains("moderate")]
    macros["powerModAtN"] = f"{100*m_[m_.N_events == 77].power.iloc[0]:.0f}\\%"
    wk = w[w.effect.str.contains("weak")]
    macros["powerWeakAtN"] = f"{100*wk[wk.N_events == 77].power.iloc[0]:.0f}\\%"
else:
    todo("tab_power", p)
    macros.update(powerModAtN="?", powerWeakAtN="?")

# ---------------------------------------------------------------- Table 6: sensitivity to the haor point (optional)
if os.path.exists("site_sensitivity_summary.csv") and os.path.exists("site_sensitivity_agreement.csv"):
    ss = pd.read_csv("site_sensitivity_summary.csv").set_index("site"); ag = pd.read_csv("site_sensitivity_agreement.csv").iloc[0]
    lines = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Haor point & positive drop & clear $+$ & clear $-$ & mean drop \\", r"\midrule"]
    for site, lab in (("sylhet", "Sylhet (main)"), ("sunamganj", "Sunamganj")):
        q = ss.loc[site]
        lines.append(f"{lab} & {int(q.positive_drop)} of {int(q.cells)} & {int(q.sig_positive)} & {int(q.sig_negative)} & {q.mean_drop:+.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_sites", "\n".join(lines) + "\n")
    macros.update(siteCorr=f"{ag['corr']:.2f}", siteSameSign=f"{100*ag['same_sign']:.0f}\\%",
                  nPosSun=str(int(ss.loc['sunamganj'].positive_drop)),
                  pctPosSun=f"{100*ss.loc['sunamganj'].positive_drop/ss.loc['sunamganj'].cells:.0f}")
else:
    todo("tab_sites", "site_sensitivity_*.csv (run compare_sites.py)")
    macros.update(siteCorr="?", siteSameSign="?", nPosSun="?", pctPosSun="?")

# ---------------------------------------------------------------- Table 7: third model, gradient boosting (step 9, optional)
need = [f"{OUT6}/gb_stream_drops.csv", f"{OUT6}/gb_results_all.csv", f"{OUT6}/gb_pvalues_holm.csv"]
if all(os.path.exists(q) for q in need):
    gd = pd.read_csv(need[0], float_precision="round_trip"); gd = gd[gd.base_rate.astype(str) == "sample"]
    gr = pd.read_csv(need[1], float_precision="round_trip"); gp = pd.read_csv(need[2], float_precision="round_trip")
    leads = sorted(gr.lead_days.unique())
    lines = [r"\begin{tabular}{@{}l" + "r" * len(leads) + r"@{}}", r"\toprule",
             "Gradient boosting & " + " & ".join(f"{h} d" for h in leads) + r" \\", r"\midrule"]
    for cfg, lab in (("SEASON", "AUC, season only"), ("FULL", "AUC, season + 4 streams")):
        v = [gr[(gr.config == cfg) & (gr.lead_days == h)].auc_hard.iloc[0] for h in leads]
        lines.append(f"{lab} & " + " & ".join(fmt(x, 3) for x in v) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(len(leads) + 1) + r"}{@{}l}{AUC drop when stream removed} \\")
    for st in NAMES:
        cells = []
        for h in leads:
            q = gd[(gd.stream == st) & (gd.lead_days == h)].iloc[0]
            cells.append(f"{q.d_auc:+.3f}" + (r"$^{*}$" if q.auc_sig else ""))
        lines.append(f"{NAMES[st]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("tab_gb", "\n".join(lines) + "\n")
    se = gr[gr.config == "SEASON"].iloc[0]; fu = gr[gr.config == "FULL"].sort_values("lead_days")
    s2 = gd[gd.stream == "S2_upstream_rain"]
    macros.update(gbSeasonHard=fmt(se.auc_hard, 2), gbSeasonAll=fmt(se.auc_all, 2),
                  gbFullBelow=str(int((fu.auc_hard.values < se.auc_hard).sum())),
                  gbNCells=str(len(gd)), gbPos=str(int((gd.d_auc > 0).sum())),
                  gbSigPos=str(int(((gd.auc_sig) & (gd.d_auc > 0)).sum())), gbSigNeg=str(int(((gd.auc_sig) & (gd.d_auc < 0)).sum())),
                  gbRawSig=str(int((gp.p_raw < 0.05).sum())), gbMinP=f"{gp.p_raw.min():.3f}",
                  gbHolmSixteen=str(int((gp.p_holm_16 < 0.05).sum())), gbHolmJoint=str(int((gp.p_holm_48 < 0.05).sum())),
                  gbRevPos=str(int(((gd.rev_sig) & (gd.d_rev > 0)).sum())), gbRevNeg=str(int(((gd.rev_sig) & (gd.d_rev < 0)).sum())),
                  gbStwoPos=str(int((s2.d_auc > 0).sum())), gbStwoLo=f"{s2.d_auc.min():.3f}", gbStwoHi=f"{s2.d_auc.max():.3f}")
else:
    todo("tab_gb", "outputs6/gb_*.csv (run step9_third_model.py on the real data)")
    macros.update({k: "?" for k in ["gbSeasonHard", "gbSeasonAll", "gbFullBelow", "gbNCells", "gbPos", "gbSigPos", "gbSigNeg", "gbRawSig",
                                    "gbMinP", "gbHolmSixteen", "gbHolmJoint", "gbRevPos", "gbRevNeg", "gbStwoPos", "gbStwoLo", "gbStwoHi"]})

# ---------------------------------------------------------------- figures (all drawn here from your CSV files)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

os.makedirs("figures", exist_ok=True)
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated categorical slots 1-4
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
plt.rcParams.update({"font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "pdf.fonttype": 42})

def save(fig, name):
    fig.savefig(f"figures/{name}.pdf", bbox_inches="tight"); fig.savefig(f"figures/{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig); print(f"  [ok]      figures/{name}.pdf")

# ---- Fig. design: what 'lead time h' means in this study (no data needed)
fig, ax = plt.subplots(figsize=(3.45, 1.7)); ax.set_xlim(-0.3, 10.3); ax.set_ylim(-1.6, 2.1); ax.axis("off")
ax.add_patch(Rectangle((0.2, -0.28), 3.8, 0.56, color=BLUE, alpha=0.18, lw=0))
ax.annotate("", xy=(10.2, 0), xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=0.8))
for x, lab in [(2.2, r"$t_0\!-\!4$"), (4.0, r"$t_0\!-\!3$"), (5.8, r"$t_0\!-\!2$"), (7.6, r"$t_0\!-\!1$")]:
    ax.plot([x, x], [-0.15, 0.15], color=MUTED, lw=0.8); ax.text(x, -0.55, lab, ha="center", va="top", color=MUTED)
ax.plot([9.4, 9.4], [-0.3, 0.3], color=ORANGE, lw=1.6); ax.text(9.4, -0.55, r"$t_0$", ha="center", va="top", color=ORANGE, fontweight="bold")
ax.text(9.4, 0.55, "event:\nflood / no flood\n(report label)", ha="center", va="bottom", color=ORANGE, fontsize=6.5)
ax.annotate("", xy=(4.0, 1.0), xytext=(0.2, 1.0), arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.8))
ax.text(2.1, 1.12, "model may use data\nup to here ($h=3$ days)", ha="center", va="bottom", color=BLUE, fontsize=6.5)
ax.text(2.1, -1.35, "rain, soil moisture, upstream flow", ha="center", va="center", color=MUTED, fontsize=6.3)
save(fig, "fig_design")

# ---- Fig. season: why a calendar baseline is so strong
if os.path.exists("events.csv"):
    ev = pd.read_csv("events.csv", parse_dates=["date"]); ev["m"] = ev.date.dt.month
    c = pd.crosstab(ev.m, ev.flood_label).reindex(range(1, 13), fill_value=0)
    fig, ax = plt.subplots(figsize=(3.45, 2.0))
    ax.bar(c.index, c[0], color=BLUE, width=0.72, label="no flood", edgecolor="white", linewidth=0.8)
    ax.bar(c.index, c[1], bottom=c[0], color=ORANGE, width=0.72, label="flood", edgecolor="white", linewidth=0.8)
    ax.set_xticks(range(1, 13)); ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_xlabel("month of event"); ax.set_ylabel("number of events")
    ax.axvspan(2.5, 9.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    ax.set_ylim(0, 14); ax.text(6, 13.6, "hard subset (Mar-Sep)", ha="center", va="top", color=MUTED, fontsize=6.5)
    ax.legend(frameon=False, loc="upper right", fontsize=6.5); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_season")
else:
    print("  [missing] figures/fig_season (needs events.csv in this folder)")

# ---- Fig. ablation: AUC drop per stream and lead time, with year-cluster bootstrap intervals
p = f"{OUT6}/stream_drops.csv"
if os.path.exists(p):
    d = pd.read_csv(p); s = d[d.base_rate.astype(str) == "sample"]
    leads = sorted(s.lead_days.unique()); cols = [BLUE, ORANGE, AQUA, YELLOW]; mk = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharey=True)
    for ax, m in zip(axes, ["logit", "rf"]):
        for i, st in enumerate(NAMES):
            q = s[(s.model == m) & (s.stream == st)].sort_values("lead_days")
            x = np.array(q.lead_days) + (i - 1.5) * 0.12
            ax.errorbar(x, q.d_auc, yerr=[q.d_auc - q.d_auc_lo, q.d_auc_hi - q.d_auc], fmt=mk[i], ms=3.6,
                        color=cols[i], ecolor=cols[i], elinewidth=0.9, capsize=1.5, mec="white", mew=0.5, label=NAMES[st])
        ax.axhline(0, color=MUTED, lw=0.7); ax.set_xticks(leads); ax.set_xticklabels([f"{h} d" for h in leads])
        ax.set_xlabel("information cut-off before event"); ax.set_title(MODEL[m], fontsize=8, color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("AUC drop when stream removed\n(above 0 = stream helps)")
    h_, l_ = axes[0].get_legend_handles_labels(); fig.legend(h_, l_, frameon=False, ncol=4, fontsize=6.8, loc="lower center", bbox_to_anchor=(0.5, -0.17))
    save(fig, "fig_ablation")
else:
    print(f"  [missing] figures/fig_ablation ({p})")

# ---- Fig. lead: season-only vs full model AUC by lead time
p = f"{OUT6}/results_all.csv"
if os.path.exists(p):
    r = pd.read_csv(p); leads = sorted(r.lead_days.unique())
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for m, col, mkr in (("logit", BLUE, "o"), ("rf", ORANGE, "s")):
        for cfg, ls in (("FULL", "-"), ("SEASON", "--")):
            q = r[(r.model == m) & (r.config == cfg)].sort_values("lead_days")
            ax.plot(q.lead_days, q.auc_hard, ls, color=col, marker=mkr, ms=3.4, lw=1.2,
                    label=f"{MODEL[m]}, " + ("season + streams" if cfg == "FULL" else "season only"))
    ax.set_xticks(leads); ax.set_xticklabels([f"{h} d" for h in leads]); ax.set_xlabel("information cut-off before event")
    ax.set_ylabel("AUC, hard subset"); ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)
    save(fig, "fig_lead_auc")
else:
    print(f"  [missing] figures/fig_lead_auc ({p})")

# ---- Fig. power: chance of detecting a real stream vs number of events
p = f"{OUT6}/power_analysis.csv"
if os.path.exists(p):
    w = pd.read_csv(p); fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for (e, col, mkr) in zip(w.effect.unique(), [BLUE, ORANGE], ["o", "s"]):
        q = w[w.effect == e].sort_values("N_events"); ax.plot(q.N_events, q.power, marker=mkr, ms=3.6, lw=1.3, color=col, label=e.replace("beta", "\u03b2"))
    ax.axhline(0.8, color=MUTED, lw=0.6, ls=":"); ax.text(1250, 0.82, "80%", ha="right", va="bottom", color=MUTED, fontsize=6.5)
    ax.axvline(77, color=MUTED, lw=0.8, ls="--"); ax.text(82, 0.5, "our 77\nevents", color=MUTED, fontsize=6.5, va="bottom")
    ax.set_xscale("log"); ax.set_xticks([40, 77, 150, 300, 600, 1200]); ax.set_xticklabels(["40", "77", "150", "300", "600", "1200"])
    ax.set_xlabel("number of events"); ax.set_ylabel("power (simulated)"); ax.set_ylim(0, 1.03)
    ax.yaxis.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6.3, loc="lower right")
    save(fig, "fig_power")
else:
    print(f"  [missing] figures/fig_power ({p})")

# ---------------------------------------------------------------- macros file
with open(f"{TAB}/results_macros.tex", "w") as f:
    for k, v in macros.items(): f.write(f"\\renewcommand{{\\{k}}}{{{v}}}\n")
print("done. Now compile:  pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper")
