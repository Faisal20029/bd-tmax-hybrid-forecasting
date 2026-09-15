"""Seasonal forecast maps (2026): full-country smooth surface, IDW (bounded by station range), station-coverage
outline shown as dashed line, Natural Earth boundary (public domain).
Run: python seasonal_map.py --shp ne_10m_admin_0_countries.shp --csv table_seasonal_2026_upd.csv [--shared] [--separate]
--shared    one colour scale for all three seasons (colours comparable between seasons)
--separate  one file per season instead of a single three-panel figure"""
import argparse, sys, numpy as np, pandas as pd, geopandas as gpd, matplotlib
import matplotlib.pyplot as plt
from shapely.geometry import Point, MultiPoint
ap = argparse.ArgumentParser(); ap.add_argument("--shp", required=True); ap.add_argument("--csv", default="table_seasonal_2026_upd.csv")
ap.add_argument("--out", default="fig_seasonal_maps_2026"); ap.add_argument("--idw_power", type=float, default=2.0)
ap.add_argument("--shared", action="store_true", help="one colour scale for all three seasons")
ap.add_argument("--separate", action="store_true", help="save one figure per season instead of a 3-panel figure"); a = ap.parse_args()
COORDS = {"Dhaka": (23.81, 90.41), "Chittagong": (22.36, 91.78), "Sylhet": (24.89, 91.86), "Rajshahi": (24.37, 88.60),
          "Rangpur": (25.75, 89.25), "Mymensingh": (24.75, 90.42), "Barisal": (22.70, 90.35), "Khulna": (22.84, 89.54),
          "Jessore": (23.17, 89.21), "Bogra": (24.85, 89.31), "Comilla": (23.46, 91.18), "Srimongal": (24.30, 91.72),
          "Cox's Bazar": (21.45, 92.02), "Faridpur": (23.61, 89.84), "Rangamati": (23.14, 92.14), "Bhola": (22.68, 90.65)}
LABEL = {"Chittagong": "Chattogram", "Barisal": "Barishal", "Jessore": "Jashore", "Bogra": "Bogura", "Comilla": "Cumilla", "Srimongal": "Sreemangal"}
OFF = {"Rangamati": (0.10, -0.25), "Sylhet": (0.08, 0.10), "Barisal": (-0.75, -0.50), "Bhola": (0.22, -0.30), "Cox's Bazar": (0.10, -0.20), "Khulna": (-0.68, -0.08), "Faridpur": (-0.62, 0.08), "Comilla": (-0.78, 0.12), "Rangpur": (0.10, 0.10)}
SEASONS = [("Summer (Mar–Jun)_mean", "Summer (Mar–Jun)"), ("Rainy (Jul–Oct)_mean", "Rainy (Jul–Oct)"), ("Winter (Nov–Feb)_mean", "Winter (Nov–Feb)")]
PANEL = "abc"   # Sci Rep wants (a)/(b)/(c) on multi-panel figures; kept on the single-season files too, for assembly in Word
world = gpd.read_file(a.shp).to_crs("EPSG:4326"); col = next(c for c in ["ADMIN", "NAME", "shapeName"] if c in world.columns)
bang = world[world[col].str.contains("Bangladesh", case=False)]
if len(bang) == 0: sys.exit("Bangladesh polygon not found in shapefile")
poly = bang.union_all() if hasattr(bang, "union_all") else bang.unary_union
minx, miny, maxx, maxy = bang.total_bounds
df = pd.read_csv(a.csv); df["lat"] = df.station.map(lambda s: COORDS[s][0]); df["lon"] = df.station.map(lambda s: COORDS[s][1])
hull = MultiPoint([Point(x, y) for x, y in zip(df.lon, df.lat)]).convex_hull.buffer(0.1).intersection(poly)
nx = ny = 400; gx, gy = np.meshgrid(np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny))
pts = gpd.GeoSeries([Point(x, y) for x, y in zip(gx.ravel(), gy.ravel())], crs="EPSG:4326"); inside = pts.within(poly).values.reshape(gx.shape)
def idw(xs, ys, vs, X, Y, p):
    d = np.sqrt((X[..., None]-xs)**2 + (Y[..., None]-ys)**2); w = 1/np.maximum(d, 1e-6)**p; return (w*vs).sum(-1)/w.sum(-1)
matplotlib.rcParams.update({"font.size": 8})
allv = np.concatenate([df[c].values for c, _ in SEASONS])

def panel(fig, ax, c, title, letter):
    """Draw one season onto ax; colour range is the shared range with --shared, else this season's own."""
    z = idw(df.lon.values, df.lat.values, df[c].values, gx, gy, a.idw_power); z[~inside] = np.nan
    lo, hi = (allv.min(), allv.max()) if a.shared else (df[c].min(), df[c].max())
    levels = np.linspace(lo - 0.05, hi + 0.05, 41)
    cf = ax.contourf(gx, gy, z, levels=levels, cmap="RdYlBu_r", antialiased=True)
    bang.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)
    ax.scatter(df.lon, df.lat, s=28, facecolor="none", edgecolor="white", linewidth=1.2, zorder=5)
    for _, r in df.iterrows():
        dx, dy = OFF.get(r.station, (0.07, 0.06)); ax.text(r.lon+dx, r.lat+dy, f"{LABEL.get(r.station, r.station)} {r[c]:.1f}", fontsize=6.5, zorder=6,
                                                        color="black", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.55))
    ax.set_aspect("equal"); ax.set_xlim(minx-0.15, maxx+0.95); ax.set_ylim(miny-0.15, maxy+0.15)
    ax.set_title(f"({letter}) {title} 2026", fontsize=10); ax.set_xlabel("Longitude (°E)")
    cb = fig.colorbar(cf, ax=ax, shrink=0.8, pad=0.02, ticks=np.round(np.linspace(lo, hi, 7), 1)); cb.set_label("Forecast monthly mean of daily maximum temperature (°C)", fontsize=7)

def save(fig, stem):
    fig.savefig(stem+".png", dpi=300); fig.savefig(stem+".pdf"); print("saved", stem+".png /.pdf")

if a.separate:
    for letter, (c, title) in zip(PANEL, SEASONS):
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        panel(fig, ax, c, title, letter); ax.set_ylabel("Latitude (°N)")
        fig.tight_layout(); save(fig, f"{a.out}_{title.split()[0].lower()}"); plt.close(fig)
else:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
    for ax, letter, (c, title) in zip(axes, PANEL, SEASONS): panel(fig, ax, c, title, letter)
    axes[0].set_ylabel("Latitude (°N)")
    fig.tight_layout(); save(fig, a.out)
