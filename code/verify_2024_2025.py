"""Verify the 2024-2026 forecasts (models fitted on 1972-2023) against observed Jan 2024 - Dec 2025."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pipeline import load_all

new = load_all("/mnt/user-data/uploads/2024-2025_newdata.xlsx") if False else None
xl = pd.ExcelFile("/mnt/user-data/uploads/2024-2025_newdata.xlsx"); obs = {}
for s in xl.sheet_names:
    d = xl.parse(s); d.columns = ["SL", "Station", "Year", "Month", "Temperature"]
    obs[d.Station[0]] = pd.Series(d.Temperature.values, index=pd.to_datetime(dict(year=d.Year, month=d.Month, day=1)))
hist = load_all("/mnt/project/newdata.xlsx")
fc = pd.read_csv("forecast_2024_2026_monthly.csv", parse_dates=["date"])
SEASON = {"Summer (Mar–Jun)": [3, 4, 5, 6], "Rainy (Jul–Oct)": [7, 8, 9, 10], "Winter (Nov–Feb)": [11, 12, 1, 2]}

rows, monthly = [], []
for st in obs:
    y = obs[st]; f = fc[(fc.station == st) & (fc.date <= "2025-12-01")].set_index("date").reindex(y.index)
    snaive = np.array([hist[st].values[-12 + (i % 12)] for i in range(24)])
    e = y.values - f.forecast.values; es = y.values - f.sarima_forecast.values; en = y.values - snaive
    rows.append(dict(station=st, model=f.model.iloc[0],
                     MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)), bias=e.mean(),
                     MAE_2024=np.mean(np.abs(e[:12])), MAE_2025=np.mean(np.abs(e[12:])),
                     MAE_SARIMA=np.mean(np.abs(es)), MAE_snaive=np.mean(np.abs(en)),
                     MASE_vs_snaive=np.mean(np.abs(e))/np.mean(np.abs(en)),
                     cov95_conformal=np.mean((y.values >= f.lower95) & (y.values <= f.upper95)),
                     cov95_sarima=np.mean((y.values >= f.sarima_lower95) & (y.values <= f.sarima_upper95)),
                     n_above_upper=int(np.sum(y.values > f.upper95)), n_below_lower=int(np.sum(y.values < f.lower95)),
                     Apr2024_obs=y["2024-04-01"], Apr2024_fc=f.forecast["2024-04-01"], Apr2024_upper=f.upper95["2024-04-01"],
                     Apr2024_percentile_1972_2023=100*np.mean(hist[st][hist[st].index.month == 4] <= y["2024-04-01"])))
    m = f.assign(observed=y.values, error=e, snaive=snaive, station=st); monthly.append(m.rename_axis("date").reset_index())
ver = pd.DataFrame(rows); ver.to_csv("table_verification_2024_2025.csv", index=False)
mon = pd.concat(monthly); mon.to_csv("verification_monthly_2024_2025.csv", index=False)

# seasonal comparison
srows = []
for st in obs:
    m = mon[mon.station == st].set_index("date")
    for yr in [2024, 2025]:
        r = dict(station=st, year=yr)
        for sn, mons in SEASON.items():
            sel = m[(m.index.year == yr) & (m.index.month.isin(mons))]
            r[sn + " obs"] = sel.observed.mean(); r[sn + " fc"] = sel.forecast.mean()
            r[sn + " PI"] = f"{sel.lower95.mean():.2f}–{sel.upper95.mean():.2f}"
            r[sn + " inPI"] = (sel.lower95.mean() <= sel.observed.mean() <= sel.upper95.mean())
        srows.append(r)
seas = pd.DataFrame(srows); seas.to_csv("table_verification_seasonal.csv", index=False)

# summary
allm = mon.copy()
summary = dict(n_points=len(allm), MAE=np.mean(np.abs(allm.error)), RMSE=np.sqrt(np.mean(allm.error**2)), bias=allm.error.mean(),
               MAE_sarima=np.mean(np.abs(allm.observed-allm.sarima_forecast)), MAE_snaive=np.mean(np.abs(allm.observed-allm.snaive)),
               cov_conformal=np.mean((allm.observed >= allm.lower95) & (allm.observed <= allm.upper95)),
               cov_sarima=np.mean((allm.observed >= allm.sarima_lower95) & (allm.observed <= allm.upper95)),
               MAE_by_year=allm.groupby(allm.date.dt.year).error.apply(lambda e: np.mean(np.abs(e))).round(3).to_dict(),
               bias_by_month=allm.groupby(allm.date.dt.month).error.mean().round(2).to_dict(),
               MAE_by_month=allm.groupby(allm.date.dt.month).error.apply(lambda e: np.mean(np.abs(e))).round(2).to_dict())
for k, v in summary.items(): print(k, v)
print(ver.round(3).to_string())
print(seas.round(2)[["station", "year", "Summer (Mar–Jun) obs", "Summer (Mar–Jun) fc", "Summer (Mar–Jun) inPI", "Winter (Nov–Feb) obs", "Winter (Nov–Feb) fc", "Winter (Nov–Feb) inPI"]].to_string())

# figure 4x4: 2022-2025 observed, forecast + PI
plt.rcParams.update({"font.size": 7.5})
fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True)
for ax, st in zip(axes.ravel(), obs):
    h = hist[st]["2021":]; m = mon[mon.station == st].set_index("date")
    ax.plot(h.index, h.values, color="black", lw=0.9, label="Observed (training period)")
    ax.plot(m.index, m.observed, color="black", lw=0.9, ls="--", marker="o", ms=2, label="Observed 2024–25 (not used in fitting)")
    ax.fill_between(m.index, m.lower95, m.upper95, color="tab:red", alpha=0.2, label="95% PI")
    ax.plot(m.index, m.forecast, color="tab:red", lw=1.0, label="Forecast (fitted to 1972–2023)")
    ax.axvline(pd.Timestamp("2023-12-15"), color="grey", lw=0.6, ls=":")
    r = ver[ver.station == st].iloc[0]
    ax.set_title(f"{st} — {r.model}: MAE {r.MAE:.2f} °C, coverage {100*r.cov95_conformal:.0f}%", fontsize=7.5); ax.grid(alpha=.25)
for ax in axes[:, 0]: ax.set_ylabel("Max. temperature (°C)")
for ax in axes[-1]: ax.set_xlabel("Year")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig("fig_verification_2024_2025.png", dpi=300); plt.savefig("fig_verification_2024_2025.pdf")
