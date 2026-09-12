"""Figures 1-3 and Supplementary Figs S1-S2 for the revised manuscript.
Run: python make_figs_1_3.py --shp ne_10m_admin_0_countries.shp --data newdata_1972_2025.xlsx"""
import argparse, numpy as np, pandas as pd, matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
ap = argparse.ArgumentParser(); ap.add_argument("--shp", required=True); ap.add_argument("--data", default="newdata_1972_2025.xlsx")
ap.add_argument("--shp1", default="ne_10m_admin_1_states_provinces/ne_10m_admin_1_states_provinces.shp",
                help="Natural Earth admin-1 shapefile; division boundaries are skipped if it is missing"); a = ap.parse_args()
matplotlib.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "pdf.fonttype": 42})
COORDS = {"Dhaka": (23.81, 90.41), "Chittagong": (22.36, 91.78), "Sylhet": (24.89, 91.86), "Rajshahi": (24.37, 88.60),
          "Rangpur": (25.75, 89.25), "Mymensingh": (24.75, 90.42), "Barisal": (22.70, 90.35), "Khulna": (22.84, 89.54),
          "Jessore": (23.17, 89.21), "Bogra": (24.85, 89.31), "Comilla": (23.46, 91.18), "Srimongal": (24.30, 91.72),
          "Cox's Bazar": (21.45, 92.02), "Faridpur": (23.61, 89.84), "Rangamati": (23.14, 92.14), "Bhola": (22.68, 90.65)}
LABEL = {"Chittagong": "Chattogram", "Barisal": "Barishal", "Jessore": "Jashore", "Bogra": "Bogura", "Comilla": "Cumilla", "Srimongal": "Sreemangal"}
COASTAL = {"Chittagong", "Cox's Bazar", "Barisal", "Bhola", "Khulna"}
# (dx, dy, ha): label pushed clear of the country where there is room, joined to its marker by a leader line,
# so crowded stations (Barishal/Bhola, Cumilla/Rangamati) can no longer be read off the wrong dot.
LAB = {"Rangpur": (-0.75, 0.10, "right"), "Bogra": (-0.55, 0.20, "right"), "Rajshahi": (-0.35, 0.15, "right"),
       "Jessore": (-0.42, -0.12, "right"), "Khulna": (-0.55, -0.10, "right"), "Faridpur": (-0.32, 0.24, "right"),
       "Barisal": (-0.55, -0.78, "right"), "Dhaka": (0.18, 0.30, "left"), "Mymensingh": (0.22, 0.32, "left"),
       "Sylhet": (0.30, 0.16, "left"), "Srimongal": (0.58, 0.00, "left"), "Comilla": (0.55, 0.30, "left"),
       "Rangamati": (0.38, 0.12, "left"), "Chittagong": (0.45, -0.22, "left"), "Cox's Bazar": (0.35, -0.14, "left"),
       "Bhola": (0.48, -0.42, "left")}
REP = ["Dhaka", "Cox's Bazar", "Rajshahi", "Sylhet"]

def save(fig, name):
    fig.savefig(name + ".png", dpi=300, bbox_inches="tight"); fig.savefig(name + ".pdf", bbox_inches="tight"); plt.close(fig); print("saved", name)

# ---------------- Fig 1: station map ----------------
import os, geopandas as gpd
world = gpd.read_file(a.shp).to_crs("EPSG:4326"); col = next(c for c in ["ADMIN", "NAME"] if c in world.columns)
bang = world[world[col].str.contains("Bangladesh", case=False)]; minx, miny, maxx, maxy = bang.total_bounds
fig, ax = plt.subplots(figsize=(6.3, 7.2))
ax.set_facecolor("#cfe3f0")                                              # water; land is drawn over it
world.plot(ax=ax, facecolor="#dedcd5", edgecolor="#a8a8a8", linewidth=0.5)   # neighbours, deliberately duller than Bangladesh
if os.path.exists(a.shp1):                                               # division boundaries, if the admin-1 file is there
    adm1 = gpd.read_file(a.shp1).to_crs("EPSG:4326")
    adm1[adm1["admin"].str.contains("Bangladesh", case=False, na=False)].boundary.plot(ax=ax, edgecolor="#9db4c2", linewidth=0.6, zorder=3)
bang.plot(ax=ax, facecolor="#fcfaf2", edgecolor="none", zorder=2)
bang.boundary.plot(ax=ax, edgecolor="black", linewidth=1.1, zorder=4)
for nm, x, y in [("INDIA", 88.05, 26.45), ("INDIA", 92.95, 25.35), ("MYANMAR", 93.15, 22.65)]:
    ax.text(x, y, nm, fontsize=7, color="#6f6f6f", ha="center", va="center", zorder=3)
ax.text(90.0, 21.45, "Bay of Bengal", fontsize=7.5, color="#4a6d87", style="italic", ha="center", va="center", zorder=3)
for st, (lat, lon) in COORDS.items():
    c = "#c62828" if st not in COASTAL else "#1565c0"
    dx, dy, ha = LAB[st]
    ax.annotate(LABEL.get(st, st), xy=(lon, lat), xytext=(lon + dx, lat + dy), fontsize=7, ha=ha, va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#555555", shrinkA=3, shrinkB=5))
    ax.scatter(lon, lat, s=40, color=c, edgecolor="black", linewidth=0.6, zorder=7)
ax.scatter([], [], s=40, color="#c62828", edgecolor="black", label="Inland station (11)")
ax.scatter([], [], s=40, color="#1565c0", edgecolor="black", label="Coastal station (5)")
ax.legend(loc="lower left", fontsize=7, frameon=True, framealpha=0.9)
ax.annotate("N", xy=(0.955, 0.955), xytext=(0.955, 0.885), xycoords="axes fraction", ha="center", va="center",
            fontsize=9, fontweight="bold", arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"))
KM, x0, y0 = 100, 89.55, 20.95                                           # scale bar: 1° lon shrinks as cos(latitude)
deg = KM / (111.320 * np.cos(np.radians(y0)))
ax.plot([x0, x0 + deg], [y0, y0], color="black", lw=1.8, solid_capstyle="butt", zorder=8)
for xe in (x0, x0 + deg): ax.plot([xe, xe], [y0 - 0.07, y0 + 0.07], color="black", lw=1.0, zorder=8)
ax.text(x0 + deg / 2, y0 + 0.11, f"{KM} km", ha="center", va="bottom", fontsize=6.5, zorder=8)
ax.set_aspect("equal"); ax.set_xlim(minx - 0.60, maxx + 1.00); ax.set_ylim(miny - 0.35, maxy + 0.35)
ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.set_title("BMD stations used (monthly maximum temperature, 1972–2025)")
save(fig, "fig1_station_map")

# ---------------- Fig 2: workflow ----------------
fig, ax = plt.subplots(figsize=(9.0, 8.6)); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 13.2)
def box(x, y, w, h, text, fc="#f4f6f8", fs=6.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.15", fc=fc, ec="black", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=0.8, color="black"))
box(1, 12.1, 8, 0.9, "Data: BARC/BMD monthly average maximum temperature, 16 stations\n1972–2023 (model development)  |  2024–2025 (held out for verification)", "#e8f0fe", bold=True)
box(1, 10.9, 8, 0.8, "Screening (completeness, calendar-month outliers) · exploratory analysis\n(decomposition, ACF/PACF, ADF/PP) · station characteristics")
arrow(5, 12.1, 5, 11.7)
box(0.5, 9.2, 2.9, 1.3, "SARIMA\norders by AICc (1972–2011)\ndrift, D = 1, s = 12", "#fff4e5")
box(3.55, 9.2, 2.9, 1.3, "Standalone ML\nANN · SVR · XGBoost · LSTM\n12 lags + calendar month", "#fff4e5")
box(6.6, 9.2, 2.9, 1.3, "Residual hybrids\nSARIMA + ANN / SVR / XGBoost / LSTM\nlearner fitted to SARIMA residuals", "#fff4e5")
for x in (1.95, 5, 8.05): arrow(5, 10.9, x, 10.5)
box(1, 7.4, 8, 1.4, "Rolling-origin (expanding-window) evaluation: origins Dec 2011 / 2014 / 2017 / 2020, 36-month horizon, 2012–2023\n"
    "(i) 36-month recursive setting  ·  (ii) one-step-ahead setting (parameters frozen at origin)\n"
    "everything learned from the training window only (no leakage) · seasonal-naïve benchmark", "#e9f7ef", bold=False)
for x in (1.95, 5, 8.05): arrow(x, 9.2, 5 if x == 5 else x, 8.8)
box(0.5, 5.7, 2.9, 1.3, "Accuracy\nMAE, RMSE, MAPE, MASE, R², ACF1\nby horizon, season, hottest decile", "#f4f6f8")
box(3.55, 5.7, 2.9, 1.3, "Statistical comparison\nDiebold–Mariano (HAC, Harvey)\nFriedman across stations", "#f4f6f8")
box(6.6, 5.7, 2.9, 1.3, "Uncertainty\nsplit-conformal 95 % PIs from\nrolling-origin errors; coverage check", "#f4f6f8")
for x in (1.95, 5, 8.05): arrow(x, 7.4, x, 7.0)
box(1, 4.0, 8, 1.2, "Station-level drivers of accuracy: anomaly variability, latitude, coastal vs inland\n(Spearman, Mann–Whitney)", "#f4f6f8")
arrow(5, 5.7, 5, 5.2)
box(1, 2.3, 8, 1.2, "Independent verification: forecasts issued Dec 2023 (fitted to 1972–2023) vs observed 2024–2025\n(384 station-months): MAE, bias, PI coverage, seasonal means, April 2024 heat-wave", "#fdecea", bold=False)
arrow(5, 4.0, 5, 3.5)
box(1, 0.6, 8, 1.2, "Final forecasts: models re-estimated on 1972–2025 → 2026 monthly and seasonal\nforecasts with 95 % PIs; station tables and maps", "#e8f0fe", bold=True)
arrow(5, 2.3, 5, 1.8)
save(fig, "fig2_workflow")

# ---------------- data ----------------
xl = pd.ExcelFile(a.data); data = {}
for s in xl.sheet_names:
    d = xl.parse(s).dropna(axis=1, how="all"); d = d[["Station", "Year", "Month", "Temperature"]]
    data[d.Station[0]] = pd.Series(d.Temperature.values, index=pd.to_datetime(dict(year=d.Year, month=d.Month, day=1)))

# ---------------- Fig 3: raw series, 4 stations ----------------
fig, axes = plt.subplots(4, 1, figsize=(7.2, 8), sharex=True)
for ax, st in zip(axes, REP):
    y = data[st]; ax.plot(y[:"2023"].index, y[:"2023"].values, color="black", lw=0.6, label="1972–2023 (model development)")
    ax.plot(y["2023-12":].index, y["2023-12":].values, color="tab:red", lw=0.8, label="2024–2025 (verification only)")
    ax.set_ylabel("Max. temperature (°C)"); ax.set_title(LABEL.get(st, st), loc="left"); ax.grid(alpha=0.25)
axes[0].legend(fontsize=7, loc="lower right", ncol=2); axes[-1].set_xlabel("Year")
fig.suptitle("Monthly average maximum temperature at four representative stations", y=0.995)
fig.tight_layout(); save(fig, "fig3_raw_series")

# ---------------- Fig S1: decomposition (additive), 4 stations ----------------
fig, axes = plt.subplots(4, 4, figsize=(11, 8.5))
for j, st in enumerate(REP):
    dec = seasonal_decompose(data[st][:"2023"], model="additive", period=12)
    for i, (comp, name) in enumerate([(dec.observed, "Observed (°C)"), (dec.trend, "Trend (°C)"), (dec.seasonal, "Seasonal (°C)"), (dec.resid, "Remainder (°C)")]):
        ax = axes[i, j]
        if i == 2:   # seasonal component is identical every year: show five years
            c5 = comp["1972":"1976"]; ax.plot(c5.index, c5.values, color="black", lw=0.9, marker="o", ms=1.8)
            ax.set_xlim(pd.Timestamp("1972-01-01"), pd.Timestamp("1976-12-01"))
            for yr in range(1973, 1977): ax.axvline(pd.Timestamp(f"{yr}-01-01"), color="grey", lw=0.4, ls=":")
            ax.set_xticks([pd.Timestamp(f"{yr}-01-01") for yr in range(1972, 1977)]); ax.set_xticklabels([str(yr) for yr in range(1972, 1977)])
        else:
            ax.plot(comp.index, comp.values, color="black", lw=0.5)
        ax.grid(alpha=0.25)
        if j == 0: ax.set_ylabel(name + ("\n(1972–1976 shown; identical each year)" if i == 2 else ""))
        if i == 0: ax.set_title(LABEL.get(st, st))
        if i == 3: ax.set_xlabel("Year")
        if i in (0, 1): ax.tick_params(labelbottom=False)
fig.suptitle("Classical additive decomposition of monthly average maximum temperature, 1972–2023 (exploratory; forecasting models use seasonal differencing)", y=0.995, fontsize=9)
fig.tight_layout(); save(fig, "figS1_decomposition")

# ---------------- Fig S2: ACF / PACF, 4 stations ----------------
fig, axes = plt.subplots(4, 2, figsize=(8, 9))
for i, st in enumerate(REP):
    y = data[st][:"2023"].values
    plot_acf(y, ax=axes[i, 0], lags=48, alpha=0.05, title=None); plot_pacf(y, ax=axes[i, 1], lags=48, alpha=0.05, method="ywm", title=None)
    axes[i, 0].set_ylabel(f"{LABEL.get(st, st)}\nACF"); axes[i, 1].set_ylabel("PACF")
    for ax in axes[i]: ax.grid(alpha=0.25); ax.set_ylim(-1, 1.05)
for ax in axes[-1]: ax.set_xlabel("Lag (months)")
axes[0, 0].set_title("Autocorrelation (48 lags, 95 % band)"); axes[0, 1].set_title("Partial autocorrelation (48 lags, 95 % band)")
fig.suptitle("Sample ACF and PACF of monthly average maximum temperature, 1972–2023", y=0.995, fontsize=9)
fig.tight_layout(); save(fig, "figS2_acf_pacf")
