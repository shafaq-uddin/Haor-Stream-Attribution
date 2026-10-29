# Study-area map for the paper.   Run:  python make_study_area_map.py
# Needs: pip install geopandas matplotlib requests   (Colab already has them)
# Data : Natural Earth 1:10m (public domain), downloaded once from GitHub into ./ne_cache/
# Output: figures/fig_study_area.pdf (vector, for LaTeX) and figures/fig_study_area.png (300 dpi preview)
import os, math, requests
import numpy as np, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
FILES = ["ne_10m_admin_0_countries", "ne_10m_admin_1_states_provinces",
         "ne_10m_rivers_lake_centerlines", "ne_10m_lakes"]
os.makedirs("ne_cache", exist_ok=True); os.makedirs("figures", exist_ok=True)
def load(name):
    p = f"ne_cache/{name}.geojson"
    if not os.path.exists(p):
        r = requests.get(BASE + name + ".geojson", timeout=300); r.raise_for_status()
        open(p, "wb").write(r.content)
    return gpd.read_file(p)
countries, states, rivers, lakes = [load(n) for n in FILES]

# ---- the places used in the paper (same coordinates as step6_leadtime.py) --------------------
PTS = {  # key: (lon, lat, label, marker, fill colour)
    "haor":   (91.87, 24.89, "S1/S3 haor point (Sylhet)",         "*", "#eb6834"),
    "sunam":  (91.40, 25.07, "Alternative point (Sunamganj)",      "*", "white"),
    "uprain": (91.72, 25.28, "S2 upstream rain (Meghalaya)",       "^", "#2a78d6"),
    "upriv":  (92.79, 24.83, "S4 upstream discharge (Barak R.)",   "s", "#1baf7a"),
}
BLUE, GREY = "#2a78d6", "#6b7280"
halo = [pe.withStroke(linewidth=2.2, foreground="white")]

def style(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / math.cos(math.radians((ext[2] + ext[3]) / 2)))   # equal-distance look at this latitude
    ax.set_facecolor("#eaf2fb")                                           # sea / no-data
def draw_base(ax, ext, detail=True):
    box = (ext[0], ext[2], ext[1], ext[3])
    c = countries.cx[ext[0]:ext[1], ext[2]:ext[3]]
    s = states.cx[ext[0]:ext[1], ext[2]:ext[3]]
    c.plot(ax=ax, color="#f4f1ea", edgecolor="none", zorder=1)
    bd = s[s.admin == "Bangladesh"]
    if len(bd): bd.plot(ax=ax, color="#fbe3d3", edgecolor="none", zorder=1.5)  # Bangladesh divisions
    if detail: s.boundary.plot(ax=ax, color="#b8b8b8", linewidth=0.4, zorder=2)
    lakes.cx[ext[0]:ext[1], ext[2]:ext[3]].plot(ax=ax, color="#bcd7f2", edgecolor="none", zorder=3)
    rv = rivers.cx[ext[0]:ext[1], ext[2]:ext[3]]
    rv.plot(ax=ax, color=BLUE, linewidth=0.7 if detail else 0.4, zorder=3.5)
    c.boundary.plot(ax=ax, color="#374151", linewidth=0.9 if detail else 0.6, zorder=4)
    style(ax, ext)

fig = plt.figure(figsize=(7.16, 3.9), dpi=300)   # IEEE two-column width
axA = fig.add_axes([0.055, 0.20, 0.29, 0.72])    # (a) regional context
axB = fig.add_axes([0.42, 0.20, 0.56, 0.72])    # (b) study area

# ---- (a) context --------------------------------------------------------------------------------
extA = (85.5, 95.0, 20.0, 28.0)
draw_base(axA, extA, detail=False)
ZB = (90.6, 93.4, 24.1, 25.7)
axA.add_patch(Rectangle((ZB[0], ZB[2]), ZB[1]-ZB[0], ZB[3]-ZB[2], fill=False, edgecolor="#eb6834", linewidth=1.2, zorder=6))
for txt, x, y in [("BANGLADESH", 89.9, 23.2), ("INDIA", 87.2, 26.6), ("MYANMAR", 94.0, 21.6)]:
    axA.text(x, y, txt, fontsize=5.5, color="#374151", ha="center", va="center", style="italic", zorder=7, path_effects=halo)
axA.set_xticks(range(86, 95, 2)); axA.set_yticks(range(20, 29, 2))
axA.set_xticklabels([f"{v}°E" for v in range(86, 95, 2)], fontsize=5.5)
axA.set_yticklabels([f"{v}°N" for v in range(20, 29, 2)], fontsize=5.5)
axA.tick_params(length=2, width=0.5, pad=1.5); axA.grid(color="white", linewidth=0.4)
axA.set_title("(a) Regional context", fontsize=7, loc="left", pad=3)

# ---- (b) study area -----------------------------------------------------------------------------
draw_base(axB, ZB, detail=True)
for k, (lon, lat, lab, m, fc) in PTS.items():
    axB.scatter(lon, lat, marker=m, s=70 if m == "*" else 38, facecolor=fc, edgecolor="black", linewidth=0.7, zorder=8)
for name, x, y, ha in [("Sylhet", 91.8687, 24.8949, "left"), ("Sunamganj", 91.4073, 25.0658, "right"),
                       ("Silchar", 92.7789, 24.8333, "left")]:
    dx = 0.04 if ha == "left" else -0.04
    axB.text(x + dx, y - 0.045, name, fontsize=6, ha=ha, va="top", zorder=9, path_effects=halo)
for nm, x, y in [("MEGHALAYA", 91.45, 25.45), ("ASSAM", 92.75, 25.45), ("BANGLADESH", 91.35, 24.35), ("TRIPURA", 92.0, 24.19)]:
    axB.text(x, y, nm, fontsize=6, color="#374151", ha="center", va="center", style="italic", zorder=7, path_effects=halo)
axB.set_xticks(np.arange(91.0, 93.4, 0.5)); axB.set_yticks(np.arange(24.25, 25.7, 0.25))
axB.set_xticklabels([f"{v:.1f}°E" for v in np.arange(91.0, 93.4, 0.5)], fontsize=6)
axB.set_yticklabels([f"{v:.2f}°N" for v in np.arange(24.25, 25.7, 0.25)], fontsize=6)
axB.tick_params(length=2, width=0.5, pad=1.5); axB.grid(color="white", linewidth=0.4, alpha=0.8)
axB.set_title("(b) Study area and data points", fontsize=7, loc="left", pad=3)

# scale bar (50 km) and north arrow
lat0 = 24.22; km_deg = 111.32 * math.cos(math.radians(lat0)); x0 = 92.45
axB.plot([x0, x0 + 50 / km_deg], [lat0, lat0], color="black", linewidth=2.2, solid_capstyle="butt", zorder=9)
axB.text(x0 + 25 / km_deg, lat0 + 0.03, "50 km", fontsize=6, ha="center", va="bottom", zorder=9, path_effects=halo)
axB.annotate("", xy=(93.2, 25.55), xytext=(93.2, 25.30), zorder=9, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
axB.text(93.2, 25.57, "N", fontsize=7, ha="center", va="bottom", fontweight="bold", zorder=9, path_effects=halo)

handles = [Line2D([], [], marker=m, linestyle="", markerfacecolor=fc, markeredgecolor="black",
                  markersize=7 if m == "*" else 5, label=lab) for (lon, lat, lab, m, fc) in PTS.values()]
handles += [Line2D([], [], color=BLUE, linewidth=1, label="Rivers (Natural Earth)"),
            Line2D([], [], color="#374151", linewidth=0.9, label="Country border")]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=6, frameon=False, columnspacing=1.6,
           handletextpad=0.5, labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
for a in (axA, axB):                       # geopandas adds default axis titles; we use our own tick labels
    a.set_xlabel(""); a.set_ylabel("")

fig.savefig("figures/fig_study_area.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figures/fig_study_area.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("Saved figures/fig_study_area.pdf and .png")
