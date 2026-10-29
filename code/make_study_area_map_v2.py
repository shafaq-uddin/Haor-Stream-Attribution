# Study-area map v2: shaded relief (DEM) + the seven haor districts.
# Run in Colab:   !pip -q install geopandas        then      %run make_study_area_map_v2.py
# Data (all downloaded automatically into ./cache/):
#   * Natural Earth 1:10m  borders, rivers, lakes        (public domain)           via GitHub
#   * geoBoundaries BGD ADM2 (districts), CC BY 4.0      Runfola et al. 2020, PLOS ONE 15(4):e0231866
#   * Elevation: AWS "Terrain Tiles" (Terrarium PNG), derived from SRTM in this region.
#       cite SRTM: Farr et al. 2007, Rev. Geophys. 45(2), RG2004, doi:10.1029/2005RG000183
# If the elevation download fails the script still draws the map without relief.
import os, io, math, sys, requests
import numpy as np, geopandas as gpd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap, PowerNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

MOCK = os.environ.get("DEM_MOCK") == "1"      # only for offline testing of the code path
GH = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
GB = "https://media.githubusercontent.com/media/wmgeolab/geoBoundaries/main/releaseData/gbOpen/BGD/ADM2/geoBoundaries-BGD-ADM2_simplified.geojson"
TILE = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
os.makedirs("cache", exist_ok=True); os.makedirs("figures", exist_ok=True)

def get(url, path):
    if not os.path.exists(path):
        r = requests.get(url, timeout=300); r.raise_for_status(); open(path, "wb").write(r.content)
    return path
countries = gpd.read_file(get(GH + "ne_10m_admin_0_countries.geojson", "cache/ne_countries.geojson"))
states    = gpd.read_file(get(GH + "ne_10m_admin_1_states_provinces.geojson", "cache/ne_states.geojson"))
rivers    = gpd.read_file(get(GH + "ne_10m_rivers_lake_centerlines.geojson", "cache/ne_rivers.geojson"))
lakes     = gpd.read_file(get(GH + "ne_10m_lakes.geojson", "cache/ne_lakes.geojson"))
dist      = gpd.read_file(get(GB, "cache/bgd_adm2.geojson"))

# ---- the seven districts usually called "haor districts" (check the spelling in `dist.shapeName`) ----
HAOR = ["Sunamganj", "Sylhet", "Maulvibazar", "Habiganj", "Netrakona", "Kishoreganj", "Brahamanbaria"]
missing = [d for d in HAOR if d not in set(dist.shapeName)]
if missing: sys.exit(f"District name(s) not found: {missing}. Available: {sorted(dist.shapeName)}")
haor = dist[dist.shapeName.isin(HAOR)]

PTS = {"haor":  (91.87, 24.89, "S1/S3 haor point (Sylhet)",       "*", "#eb6834"),
       "sunam": (91.40, 25.07, "Alternative point (Sunamganj)",   "*", "white"),
       "uprain":(91.72, 25.28, "S2 upstream rain (Meghalaya)",    "^", "#2a78d6"),
       "upriv": (92.79, 24.83, "S4 upstream discharge (Barak R.)","s", "#1baf7a")}
BLUE = "#2a78d6"; halo = [pe.withStroke(linewidth=2.2, foreground="white")]
ZB = (90.6, 93.4, 24.1, 25.7)                      # study-area extent (lon0, lon1, lat0, lat1)

# ---- elevation from terrarium tiles, resampled to a regular lon/lat grid -------------------------
def lonlat_to_px(lon, lat, z):
    n = 256 * 2 ** z
    x = (lon + 180) / 360 * n
    y = (1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * n
    return x, y
def get_tile(z, x, y):
    from PIL import Image
    p = f"cache/dem_{z}_{x}_{y}.png"
    if MOCK and not os.path.exists(p):
        n = 2 ** z; lon = (x + np.arange(256) / 256) / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + np.arange(256) / 256) / n))))
        LON, LAT = np.meshgrid(lon, lat)
        elev = 5 + 1700 * np.clip((LAT - 24.9) / 0.8, 0, 1) ** 1.5 * (1 + 0.2 * np.sin(LON * 25))   # fake plateau, test only
        v = (elev + 32768) * 256; R = np.floor(v / 65536); G = np.floor(v / 256) % 256; B = np.floor(v) % 256
        Image.fromarray(np.dstack([R, G, B]).astype("uint8")).save(p)
    else:
        get(TILE.format(z=z, x=x, y=y), p)
    a = np.array(Image.open(p).convert("RGB")).astype(float)
    return a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768            # metres
def dem_grid(ext, z=9, step=0.004):
    lons = np.arange(ext[0], ext[1], step); lats = np.arange(ext[3], ext[2], -step)
    LON, LAT = np.meshgrid(lons, lats)
    X, Y = lonlat_to_px(LON, LAT, z)
    tx0, tx1, ty0, ty1 = int(X.min() // 256), int(X.max() // 256), int(Y.min() // 256), int(Y.max() // 256)
    mosaic = np.vstack([np.hstack([get_tile(z, tx, ty) for tx in range(tx0, tx1 + 1)]) for ty in range(ty0, ty1 + 1)])
    xi = np.clip((X - tx0 * 256).astype(int), 0, mosaic.shape[1] - 1)
    yi = np.clip((Y - ty0 * 256).astype(int), 0, mosaic.shape[0] - 1)
    return lons, lats, mosaic[yi, xi]
try:
    lons, lats, elev = dem_grid(ZB); elev = np.clip(elev, 0, None); HAVE_DEM = True
    print(f"DEM ok: min {elev.min():.0f} m, max {elev.max():.0f} m")
except Exception as e:
    print("DEM download failed -> map without relief:", str(e)[:150]); HAVE_DEM = False

def style(ax, ext):
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_aspect(1 / math.cos(math.radians((ext[2] + ext[3]) / 2))); ax.set_facecolor("#eaf2fb")
fig = plt.figure(figsize=(7.16, 3.5), dpi=300)
axA = fig.add_axes([0.055, 0.22, 0.28, 0.68]); axB = fig.add_axes([0.40, 0.22, 0.50, 0.68]); cax = fig.add_axes([0.915, 0.30, 0.012, 0.52])

# ---- (a) context ------------------------------------------------------------------------------------
extA = (85.5, 95.0, 20.0, 28.0)
countries.cx[extA[0]:extA[1], extA[2]:extA[3]].plot(ax=axA, color="#f4f1ea", edgecolor="none", zorder=1)
dist.plot(ax=axA, color="#f6e3d4", edgecolor="none", zorder=1.5)
haor.plot(ax=axA, color="#eb6834", alpha=0.75, edgecolor="none", zorder=2)
rivers.cx[extA[0]:extA[1], extA[2]:extA[3]].plot(ax=axA, color=BLUE, linewidth=0.4, zorder=3)
countries.cx[extA[0]:extA[1], extA[2]:extA[3]].boundary.plot(ax=axA, color="#374151", linewidth=0.6, zorder=4)
axA.add_patch(Rectangle((ZB[0], ZB[2]), ZB[1]-ZB[0], ZB[3]-ZB[2], fill=False, edgecolor="black", linewidth=1.0, zorder=6))
for t, x, y in [("BANGLADESH", 89.7, 23.2), ("INDIA", 87.2, 26.6), ("MYANMAR", 94.0, 21.6)]:
    axA.text(x, y, t, fontsize=5.5, color="#374151", ha="center", va="center", style="italic", zorder=7, path_effects=halo)
style(axA, extA)
axA.set_xticks(range(86, 95, 2)); axA.set_yticks(range(20, 29, 2))
axA.set_xticklabels([f"{v}°E" for v in range(86, 95, 2)], fontsize=5.5); axA.set_yticklabels([f"{v}°N" for v in range(20, 29, 2)], fontsize=5.5)
axA.tick_params(length=2, width=0.5, pad=1.5); axA.grid(color="white", linewidth=0.4)
axA.set_title("(a) Regional context", fontsize=7, loc="left", pad=3)

# ---- (b) relief + haor districts + points -----------------------------------------------------------
sub = lambda g: g.cx[ZB[0]:ZB[1], ZB[2]:ZB[3]]
if HAVE_DEM:
    ls = LightSource(azdeg=315, altdeg=40)
    dx = 111320 * 0.004 * math.cos(math.radians(24.9)); dy = 111320 * 0.004
    shade = ls.hillshade(elev, vert_exag=6, dx=dx, dy=dy)
    cmap = LinearSegmentedColormap.from_list("hyps", ["#d8e8c4", "#bcd69a", "#e6d9a0", "#d1b27a", "#a98a62", "#8a7360", "#f2efe9"])
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=1800)
    rgb = cmap(norm(elev))[..., :3]
    rgb = rgb * (0.55 + 0.45 * shade[..., None])
    axB.imshow(np.clip(rgb, 0, 1), extent=(lons[0], lons[-1], lats[-1], lats[0]), origin="upper", zorder=1, interpolation="bilinear")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 50, 200, 500, 1000, 1800]); cb.ax.tick_params(labelsize=5.5, length=2, width=0.5)
    cb.set_label("Elevation (m a.s.l.)", fontsize=6, labelpad=2)
else:
    sub(countries).plot(ax=axB, color="#f4f1ea", edgecolor="none", zorder=1); cax.axis("off")
sub(states).boundary.plot(ax=axB, color="#ffffff", linewidth=0.5, alpha=0.7, zorder=2)
sub(dist).boundary.plot(ax=axB, color="#4b5563", linewidth=0.35, zorder=2.5)
haor.dissolve().boundary.plot(ax=axB, color="#eb6834", linewidth=1.5, zorder=3)   # one outline around all 7 districts
sub(lakes).plot(ax=axB, color="#bcd7f2", edgecolor="none", zorder=3.2)
sub(rivers).plot(ax=axB, color=BLUE, linewidth=0.8, zorder=3.5)
sub(countries).boundary.plot(ax=axB, color="black", linewidth=1.0, zorder=4)
for k, (lon, lat, lab, m, fc) in PTS.items():
    axB.scatter(lon, lat, marker=m, s=70 if m == "*" else 38, facecolor=fc, edgecolor="black", linewidth=0.7, zorder=8)
for nm, x, y, ha in [("Sylhet", 91.8687, 24.8949, "left"), ("Sunamganj", 91.4073, 25.0658, "right"), ("Silchar", 92.7789, 24.8333, "left")]:
    axB.text(x + (0.04 if ha == "left" else -0.04), y - 0.045, nm, fontsize=6, ha=ha, va="top", zorder=9, path_effects=halo)
for nm, x, y in [("MEGHALAYA", 91.45, 25.5), ("ASSAM", 92.75, 25.5), ("BANGLADESH", 91.35, 24.32), ("TRIPURA", 92.0, 24.19)]:
    axB.text(x, y, nm, fontsize=6, color="black", ha="center", va="center", style="italic", zorder=9, path_effects=halo)
axB.set_xticks(np.arange(91.0, 93.4, 0.5)); axB.set_yticks(np.arange(24.25, 25.7, 0.25))
axB.set_xticklabels([f"{v:.1f}°E" for v in np.arange(91.0, 93.4, 0.5)], fontsize=6)
axB.set_yticklabels([f"{v:.2f}°N" for v in np.arange(24.25, 25.7, 0.25)], fontsize=6)
axB.tick_params(length=2, width=0.5, pad=1.5)
axB.set_title("(b) Study area, terrain and data points", fontsize=7, loc="left", pad=3)
style(axB, ZB)
lat0 = 24.2; kd = 111.32 * math.cos(math.radians(lat0)); x0 = 92.4
axB.plot([x0, x0 + 50 / kd], [lat0, lat0], color="black", linewidth=2.2, solid_capstyle="butt", zorder=9)
axB.text(x0 + 25 / kd, lat0 + 0.03, "50 km", fontsize=6, ha="center", va="bottom", zorder=9, path_effects=halo)
axB.annotate("", xy=(93.2, 25.55), xytext=(93.2, 25.30), zorder=9, arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0))
axB.text(93.2, 25.57, "N", fontsize=7, ha="center", va="bottom", fontweight="bold", zorder=9, path_effects=halo)

handles = [Line2D([], [], marker=m, linestyle="", markerfacecolor=fc, markeredgecolor="black", markersize=7 if m == "*" else 5, label=lab)
           for (_, _, lab, m, fc) in PTS.values()]
handles += [Line2D([], [], color="#eb6834", linewidth=1.3, label="Haor districts (7)"),
            Line2D([], [], color=BLUE, linewidth=1, label="Rivers (Natural Earth)"),
            Line2D([], [], color="black", linewidth=1, label="Country border")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=6, frameon=False, columnspacing=1.4, handletextpad=0.5,
           labelspacing=0.4, bbox_to_anchor=(0.5, 0.0))
for a in (axA, axB): a.set_xlabel(""); a.set_ylabel("")
fig.savefig("figures/fig_study_area_v2.pdf", bbox_inches="tight", pad_inches=0.03)
fig.savefig("figures/fig_study_area_v2.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
print("Saved figures/fig_study_area_v2.pdf and .png")
