"""Why does accuracy differ across stations? Relate SARIMA MAE and hybrid gain to series characteristics."""
import numpy as np, pandas as pd
from scipy import stats
from pipeline import load_all
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

coords = {"Dhaka": (23.81, 90.41), "Chittagong": (22.36, 91.78), "Sylhet": (24.89, 91.86), "Rajshahi": (24.37, 88.60),
          "Rangpur": (25.75, 89.25), "Mymensingh": (24.75, 90.42), "Barisal": (22.70, 90.35), "Khulna": (22.84, 89.54),
          "Jessore": (23.17, 89.21), "Bogra": (24.85, 89.31), "Comilla": (23.46, 91.18), "Srimongal": (24.30, 91.72),
          "Cox's Bazar": (21.45, 92.02), "Faridpur": (23.61, 89.84), "Rangamati": (23.14, 92.14), "Bhola": (22.68, 90.65)}
coastal = {"Chittagong", "Cox's Bazar", "Barisal", "Bhola", "Khulna"}   # BMD coastal zone stations in the network
zone = {"Rajshahi": "NW inland", "Bogra": "NW inland", "Rangpur": "NW inland", "Jessore": "SW", "Khulna": "SW/coastal",
        "Faridpur": "Central", "Dhaka": "Central", "Mymensingh": "Central-N", "Comilla": "East", "Sylhet": "NE wet",
        "Srimongal": "NE wet", "Rangamati": "E hilly", "Chittagong": "SE coastal", "Cox's Bazar": "SE coastal",
        "Barisal": "S coastal", "Bhola": "S coastal"}
LAB = {"Chittagong": "Chattogram", "Barisal": "Barishal", "Jessore": "Jashore", "Bogra": "Bogura", "Comilla": "Cumilla", "Srimongal": "Sreemangal"}
data = load_all(); pooled = pd.read_csv("table_pooled_metrics.csv")
rows = []
for st, y in data.items():
    tr = y[:"2011"]
    mm = tr.groupby(tr.index.month).mean(); anom = tr - mm.reindex(tr.index.month).values
    slope = stats.linregress(np.arange(len(tr)), tr.values).slope * 120           # °C per decade
    # interannual variability of summer (Mar-Jun) means
    summer = tr[tr.index.month.isin([3, 4, 5, 6])].groupby(tr[tr.index.month.isin([3, 4, 5, 6])].index.year).mean()
    p = pooled[pooled.station == st].set_index("model")
    best_h = p.loc[[m for m in p.index if m.startswith("SARIMA+")]].MAE.min()
    rows.append(dict(station=st, lat=coords[st][0], lon=coords[st][1], zone=zone[st], coastal=int(st in coastal),
                     mean_T=tr.mean(), seasonal_amplitude=mm.max()-mm.min(), anomaly_sd=anom.std(),
                     anomaly_acf1=anom.autocorr(1), anomaly_acf12=anom.autocorr(12), trend_per_decade=slope,
                     summer_interannual_sd=summer.std(),
                     SARIMA_MAE=p.loc["SARIMA", "MAE"], best_hybrid_MAE=best_h, hybrid_gain=p.loc["SARIMA", "MAE"]-best_h,
                     hybrid_gain_pct=100*(p.loc["SARIMA", "MAE"]-best_h)/p.loc["SARIMA", "MAE"],
                     SNAIVE_MAE=p.loc["SNAIVE", "MAE"], skill_vs_snaive=100*(1-p.loc["SARIMA", "MAE"]/p.loc["SNAIVE", "MAE"])))
d = pd.DataFrame(rows); d.to_csv("table_station_characteristics.csv", index=False)
print(d.round(3).to_string())
print("\nSpearman correlations with SARIMA_MAE / hybrid_gain:")
for c in ["seasonal_amplitude", "anomaly_sd", "anomaly_acf1", "trend_per_decade", "summer_interannual_sd", "lat", "coastal"]:
    r1, p1 = stats.spearmanr(d[c], d.SARIMA_MAE); r2, p2 = stats.spearmanr(d[c], d.hybrid_gain)
    print(f"  {c:24s} MAE: rho={r1:+.2f} (p={p1:.3f})   gain: rho={r2:+.2f} (p={p2:.3f})")
print("\nCoastal vs inland SARIMA MAE:", d.groupby("coastal").SARIMA_MAE.mean().round(3).to_dict(),
      "Mann-Whitney p=%.3f" % stats.mannwhitneyu(d[d.coastal == 1].SARIMA_MAE, d[d.coastal == 0].SARIMA_MAE).pvalue)

fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
for a, xcol, xl in zip(ax, ["anomaly_sd", "seasonal_amplitude"], ["SD of monthly anomalies, 1972–2011 (°C)", "Seasonal amplitude of monthly means (°C)"]):
    a.scatter(d[xcol], d.SARIMA_MAE, c=d.coastal.map({1: "tab:blue", 0: "tab:red"}), s=28)
    for _, r in d.iterrows(): a.annotate(LAB.get(r.station, r.station), (r[xcol], r.SARIMA_MAE), fontsize=6, xytext=(2, 2), textcoords="offset points")
    a.set_xlabel(xl); a.set_ylabel("SARIMA MAE, 36-month recursive (°C)"); a.grid(alpha=.3)
ax[0].scatter([], [], c="tab:blue", label="coastal"); ax[0].scatter([], [], c="tab:red", label="inland"); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_station_drivers.png", dpi=300); plt.savefig("fig_station_drivers.pdf")
