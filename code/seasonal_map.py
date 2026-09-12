"""Seasonal forecast maps (2026): THREE separate figures in the style of the original submission
(shared discrete colour scale across seasons, RdYlBu_r, colour bar at the right, full-country surface),
but with inverse-distance weighting (bounded by the station range — no extrapolation artefacts) and
a public-domain Natural Earth boundary.
Run: python seasonal_map.py --shp ne_10m_admin_0_countries.shp --csv table_seasonal_2026_upd.csv"""
import argparse, numpy as np, pandas as pd, geopandas as gpd, matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from shapely.geometry import Point
ap = argparse.ArgumentParser(); ap.add_argument("--shp", required=True); ap.add_argument("--csv", default="table_seasonal_2026_upd.csv")
ap.add_argument("--prefix", default="fig_map_2026"); ap.add_argument("--idw_power", type=float, default=2.0)
ap.add_argument("--nlevels", type=int, default=16); a = ap.parse_args()
COORDS = {"Dhaka": (23.81, 90.41), "Chittagong": (22.36, 91.78), "Sylhet": (24.89, 91.86), "Rajshahi": (24.37, 88.60),
          "Rangpur": (25.75, 89.25), "Mymensingh": (24.75, 90.42), "Barisal": (22.70, 90.35), "Khulna": (22.84, 89.54),
          "Jessore": (23.17, 89.21), "Bogra": (24.85, 89.31), "Comilla": (23.46, 91.18), "Srimongal": (24.30, 91.72),
          "Cox's Bazar": (21.45, 92.02), "Faridpur": (23.61, 89.84), "Rangamati": (23.14, 92.14), "Bhola": (22.68, 90.65)}
LABEL = {"Chittagong": "Chattogram", "Barisal": "Barishal", "Jessore": "Jashore", "Bogra": "Bogura", "Comilla": "Cumilla", "Srimongal": "Sreemangal"}
OFF = {"Barisal": (-0.28, -0.30), "Bhola": (0.08, -0.22), "Khulna": (-0.58, -0.06), "Faridpur": (-0.55, 0.08), "Cox's Bazar": (0.10, -0.18)}
SEASONS = [("Summer (Mar–Jun)_mean", "Summer (Mar–Jun)", "summer"), ("Rainy (Jul–Oct)_mean", "Rainy (Jul–Oct)", "rainy"), ("Winter (Nov–Feb)_mean", "Winter (Nov–Feb)", "winter")]
world = gpd.read_file(a.shp).to_crs("EPSG:4326"); col = next(c for c in ["ADMIN", "NAME", "shapeName"] if c in world.columns)
bang = world[world[col].str.contains("Bangladesh", case=False)]; poly = bang.union_all() if hasattr(bang, "union_all") else bang.unary_union
minx, miny, maxx, maxy = bang.total_bounds
df = pd.read_csv(a.csv); df["lat"] = df.station.map(lambda s: COORDS[s][0]); df["lon"] = df.station.map(lambda s: COORDS[s][1])
nx = ny = 400; gx, gy = np.meshgrid(np.linspace(minx, maxx, nx), np.linspace(miny, maxy, ny))
pts = gpd.GeoSeries([Point(x, y) for x, y in zip(gx.ravel(), gy.ravel())], crs="EPSG:4326"); inside = pts.within(poly).values.reshape(gx.shape)
def idw(xs, ys, vs, X, Y, p):
    d = np.sqrt((X[..., None]-xs)**2 + (Y[..., None]-ys)**2); w = 1/np.maximum(d, 1e-6)**p; return (w*vs).sum(-1)/w.sum(-1)
# shared discrete scale across the three seasons (as in the original figures)
allv = np.concatenate([df[c].values for c, _, _ in SEASONS]); vmin, vmax = allv.min(), allv.max()
levels = np.linspace(vmin, vmax, a.nlevels); cmap = plt.get_cmap("RdYlBu_r", len(levels)-1); norm = BoundaryNorm(levels, ncolors=len(levels)-1, clip=True)
matplotlib.rcParams.update({"font.size": 9})
for c, title, tag in SEASONS:
    z = idw(df.lon.values, df.lat.values, df[c].values, gx, gy, a.idw_power); z[~inside] = np.nan
    fig, ax = plt.subplots(figsize=(6.4, 8))
    cf = ax.contourf(gx, gy, z, levels=levels, cmap=cmap, norm=norm, extend="both", alpha=0.85, antialiased=True)
    bang.boundary.plot(ax=ax, edgecolor="black", linewidth=1)
    ax.scatter(df.lon, df.lat, s=50, facecolor="none", edgecolor="white", linewidth=1.2, zorder=5)
    for _, r in df.iterrows():
        dx, dy = OFF.get(r.station, (0.10, 0.05)); ax.text(r.lon+dx, r.lat+dy, f"{LABEL.get(r.station, r.station)} ({r[c]:.1f})", fontsize=7, zorder=6)
    ax.set_aspect("equal", adjustable="box"); pad_x, pad_y = (maxx-minx)*0.03, (maxy-miny)*0.03
    ax.set_xlim(minx-pad_x, maxx+pad_x+0.35); ax.set_ylim(miny-pad_y, maxy+pad_y)
    ax.set_title(f"{title} 2026: forecast average maximum temperature (°C)", fontsize=10)
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)")
    cb = fig.colorbar(cf, ax=ax, ticks=levels, spacing="proportional", shrink=0.9, pad=0.03)
    cb.ax.set_yticklabels([f"{v:.1f}" for v in levels]); cb.set_label("Monthly mean of daily maximum temperature (°C)")
    plt.tight_layout(); fig.savefig(f"{a.prefix}_{tag}.png", dpi=300); fig.savefig(f"{a.prefix}_{tag}.pdf"); plt.close(fig); print("saved", f"{a.prefix}_{tag}")
