"""Study-area map for the paper -> figures/fig_map.pdf / .png
Boundaries: Natural Earth (public domain), downloaded once from GitHub into natural_earth/.
Point coordinates are the same ones used in step6_leadtime.py."""
import json, os, urllib.request
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPoly
import numpy as np

BASE = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
FILES = ["ne_10m_admin_0_countries", "ne_10m_admin_1_states_provinces"]
os.makedirs("natural_earth", exist_ok=True)
for f in FILES:
    p = f"natural_earth/{f}.geojson"
    if not os.path.exists(p):
        urllib.request.urlretrieve(BASE + f + ".geojson", p)
ctry = json.load(open("natural_earth/ne_10m_admin_0_countries.geojson"))["features"]
adm1 = json.load(open("natural_earth/ne_10m_admin_1_states_provinces.geojson"))["features"]

# points (lat, lon) - identical to step6_leadtime.py
SYL, SUN = (24.89, 91.87), (25.07, 91.40)
UPR, UPQ = (25.28, 91.72), (24.83, 92.79)

BLUE, ORANGE, TEAL, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#1f2937", "#6b7280"
LAND, WATER, SYLHET_FILL, INDIA_FILL = "#f3f4f6", "#eaf2fb", "#cfe3fa", "#e5e7eb"

def rings(geom):
    if geom["type"] == "Polygon": return [geom["coordinates"][0]]
    return [poly[0] for poly in geom["coordinates"]]

def draw(ax, feats, fc, ec, lw, filt=None, z=1):
    for f in feats:
        if filt and not filt(f["properties"]): continue
        for r in rings(f["geometry"]):
            ax.add_patch(MPoly(np.array(r), closed=True, fc=fc, ec=ec, lw=lw, zorder=z, joinstyle="round"))

def name(p): return p.get("name") or p.get("NAME") or ""
fig = plt.figure(figsize=(3.45, 2.72), dpi=300)
ax = fig.add_axes([0.07, 0.07, 0.915, 0.915])
x0, x1, y0, y1 = 90.2, 93.5, 23.9, 26.1
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_aspect(1 / np.cos(np.deg2rad(25)))
ax.set_facecolor(WATER)
draw(ax, ctry, LAND, GREY, 0.4, lambda p: p.get("ADMIN") in ("India", "Bangladesh", "Myanmar", "Bhutan"))
draw(ax, adm1, INDIA_FILL, "#9ca3af", 0.3, lambda p: p.get("admin") == "India" and name(p) in ("Meghalaya", "Assam", "Tripura", "Mizoram", "Manipur"))
draw(ax, adm1, SYLHET_FILL, BLUE, 0.7, lambda p: p.get("admin") == "Bangladesh" and name(p) == "Sylhet", z=2)
draw(ax, ctry, "none", INK, 1.0, lambda p: p.get("ADMIN") in ("Bangladesh",), z=3)

kw = dict(zorder=6, mec="white", mew=0.8)
ax.plot(SYL[1], SYL[0], "o", ms=7, color=BLUE, **kw)
ax.plot(SUN[1], SUN[0], "o", ms=6.5, mfc="white", mec=BLUE, mew=1.4, zorder=6)
ax.plot(UPR[1], UPR[0], "^", ms=8, color=ORANGE, **kw)
ax.plot(UPQ[1], UPQ[0], "s", ms=7, color=TEAL, **kw)

def lab(x, y, t, **k): ax.text(x, y, t, fontsize=6, color=INK, zorder=7, **k)
lab(SYL[1] - 0.07, SYL[0] - 0.05, "Sylhet point\n(S1 local rain,\nS3 soil moisture)", ha="right", va="top")
lab(SUN[1] - 0.07, SUN[0] + 0.0, "Sunamganj point\n(sensitivity run)", ha="right", va="center")
lab(UPR[1] - 0.05, UPR[0] + 0.06, "S2 upstream rain\n(Meghalaya foothills)", ha="center", va="bottom")
lab(UPQ[1] + 0.0, UPQ[0] + 0.08, "S4 upstream flow\n(GloFAS point\nnear Silchar)", ha="center", va="bottom")
for t, x, y in [("BANGLADESH", 90.65, 24.2), ("INDIA", 92.3, 25.9), ("Sylhet Division", 91.6, 24.25)]:
    ax.text(x, y, t, fontsize=5.5 if t != "Sylhet Division" else 5.2, color=GREY if t != "Sylhet Division" else BLUE,
            style="italic" if t == "Sylhet Division" else "normal", ha="center", zorder=5)
for t, x, y in [("Meghalaya", 91.0, 25.85), ("Assam", 93.1, 25.55), ("Tripura", 91.75, 23.95)]:
    ax.text(x, y, t, fontsize=5, color=GREY, ha="center", zorder=5)
ax.set_xticks([91, 92, 93]); ax.set_yticks([24, 25, 26])
ax.set_xticklabels(["91°E", "92°E", "93°E"], fontsize=5.5); ax.set_yticklabels(["24°N", "25°N", "26°N"], fontsize=5.5)
ax.tick_params(length=2, width=0.4, pad=1.5, colors=GREY)
for s in ax.spines.values(): s.set_linewidth(0.5); s.set_color(GREY)

# locator inset: whole Bangladesh with box
ins = fig.add_axes([0.785, 0.03, 0.195, 0.33])
ins.set_xlim(87.8, 93.7); ins.set_ylim(20.5, 27.0); ins.set_facecolor(WATER)
ins.set_aspect(1 / np.cos(np.deg2rad(24)))
draw(ins, ctry, LAND, GREY, 0.25, lambda p: p.get("ADMIN") in ("India", "Bangladesh", "Myanmar", "Bhutan", "Nepal"))
draw(ins, ctry, "none", INK, 0.6, lambda p: p.get("ADMIN") == "Bangladesh", z=3)
ins.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="none", ec=ORANGE, lw=0.8, zorder=4))
ins.set_xticks([]); ins.set_yticks([])
for s in ins.spines.values(): s.set_linewidth(0.5); s.set_color(GREY)

fig.text(0.985, 0.005, "Boundaries: Natural Earth", fontsize=4, color=GREY, ha="right", va="bottom")
os.makedirs("figures", exist_ok=True)
fig.savefig("figures/fig_map.pdf"); fig.savefig("figures/fig_map.png")
print("saved figures/fig_map.pdf/.png")
