"""Final 2024-2026 forecasts: fit each station's selected model on 1972-2023, produce 36-month forecasts,
empirical horizon-block prediction intervals calibrated on rolling-origin errors, coverage check
(leave-one-origin-out), seasonal means, and a 4x4 publication figure."""
import os, json, warnings, sys
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from pipeline import load_all, fit_sarima, make_xy, fit_ml, recursive_forecast, H, NLAG
warnings.filterwarnings("ignore")
ALPHA = 0.05
BLOCKS = {1: range(1, 13), 2: range(13, 25), 3: range(25, 37)}
SEASON = {"Summer (Mar–Jun)": [3, 4, 5, 6], "Rainy (Jul–Oct)": [7, 8, 9, 10], "Winter (Nov–Feb)": [11, 12, 1, 2]}
FORCE = sys.argv[1] if len(sys.argv) > 1 else None   # e.g. "SARIMA" to force one model everywhere

DATA = os.environ.get("DATA", "/mnt/project/newdata.xlsx"); UTAG = os.environ.get("UTAG", "")
data = load_all(DATA); orders = json.load(open("sarima_orders.json"))
pooled = pd.read_csv("table_pooled_metrics.csv")
f = pd.read_csv("results_forecasts.csv", index_col=0, parse_dates=True); f.index.name = "date"
fl = pd.read_csv("results_forecasts_lstm.csv", index_col=0, parse_dates=True); fl.index.name = "date"
f = f.reset_index().merge(fl.reset_index()[["date", "station", "origin", "LSTM", "SARIMA+LSTM"]], on=["date", "station", "origin"]).set_index("date")
f["h"] = f.groupby(["station", "origin"]).cumcount() + 1

def cq(e, p):
    n = len(e); k = min(int(np.ceil((n + 1) * p)), n); return np.sort(e)[k - 1]
def block_quantiles(err, h):
    """Pooled-horizon, finite-sample-corrected (split-conformal) two-sided quantiles; same for all blocks."""
    q = (cq(err, ALPHA/2), cq(err, 1-ALPHA/2)); return {b: q for b in BLOCKS}

def fit_and_forecast(model, y):
    tr, mo = y.values, y.index.month.values
    order, sorder = tuple(orders[y.name]["order"]), tuple(orders[y.name]["sorder"])
    sar = fit_sarima(tr, order, sorder); sf = sar.get_forecast(H)
    sar_mean, sar_pi = np.asarray(sf.predicted_mean), np.asarray(sf.conf_int(alpha=ALPHA))
    if model == "SARIMA": return sar_mean, sar_mean, sar_pi
    base = model.replace("SARIMA+", ""); hybrid = model.startswith("SARIMA+")
    series, months = (np.asarray(sar.resid)[13:], mo[13:]) if hybrid else (tr, mo)
    if base == "LSTM":
        from lstm_pipeline import fit_lstm
        fc, _ = fit_lstm("RES" if hybrid else "LSTM", series)
    else:
        X, Y = make_xy(series, months); m, sx, sy = fit_ml(base, X, Y)
        fc = recursive_forecast(m, sx, sy, series[-NLAG:], months[-1], H)
    return (sar_mean + fc if hybrid else fc), sar_mean, sar_pi

rows, cov_rows, monthly = [], [], []
future_idx = pd.date_range(max(v.index.max() for v in data.values()) + pd.offsets.MonthBegin(1), periods=H, freq="MS")
for st in data:
    p = pooled[pooled.station == st].sort_values("MAE")
    model = FORCE or p.model.iloc[0]
    g = f[f.station == st]; err = (g.actual - g[model]).values; h = g.h.values
    # coverage check: calibrate on 3 origins, evaluate on the held-out origin
    covs = []
    for o in g.origin.unique():
        cal, tst = g[g.origin != o], g[g.origin == o]
        q = block_quantiles((cal.actual-cal[model]).values, cal.h.values)
        lo = np.array([tst[model].values[i] + q[(tst.h.values[i]-1)//12+1][0] for i in range(len(tst))])
        hi = np.array([tst[model].values[i] + q[(tst.h.values[i]-1)//12+1][1] for i in range(len(tst))])
        covs.append(np.mean((tst.actual.values >= lo) & (tst.actual.values <= hi)))
        if model != "SARIMA":
            covs_s = np.mean((tst.actual.values >= tst.sarima_lo.values) & (tst.actual.values <= tst.sarima_hi.values))
    cov_rows.append(dict(station=st, model=model, empirical_PI_coverage=np.mean(covs),
                         SARIMA_analytic_coverage=np.mean((g.actual >= g.sarima_lo) & (g.actual <= g.sarima_hi))))
    # final forecast
    q = block_quantiles(err, h)
    fc, sar_mean, sar_pi = fit_and_forecast(model, data[st])
    lo = np.array([fc[i] + q[i//12+1][0] for i in range(H)]); hi = np.array([fc[i] + q[i//12+1][1] for i in range(H)])
    mdf = pd.DataFrame(dict(station=st, model=model, date=future_idx, forecast=fc, lower95=lo, upper95=hi,
                            sarima_forecast=sar_mean, sarima_lower95=sar_pi[:, 0], sarima_upper95=sar_pi[:, 1]))
    monthly.append(mdf)
    m26 = mdf[mdf.date.dt.year == (2026 if UTAG == "" else 2026)]
    r = dict(station=st, model=model)
    for sname, mons in SEASON.items():
        sel = m26[m26.date.dt.month.isin(mons)]
        r[sname] = f"{sel.forecast.mean():.2f} ({sel.lower95.mean():.2f}–{sel.upper95.mean():.2f})"
        r[sname + "_mean"] = sel.forecast.mean()
    rows.append(r)
    print(st, model, r["Summer (Mar–Jun)"], f"cov={np.mean(covs):.3f}", flush=True)

tag = (f"_{FORCE}" if FORCE else "") + UTAG
pd.concat(monthly).to_csv(f"forecast_2024_2026_monthly{tag}.csv", index=False)
pd.DataFrame(rows).to_csv(f"table_seasonal_2026{tag}.csv", index=False)
pd.DataFrame(cov_rows).to_csv(f"table_pi_coverage{tag}.csv", index=False)

# ---- 4x4 figure: observed 2008-2023, rolling-origin forecasts, 2024-2026 forecast with 95% PI ----
plt.rcParams.update({"font.size": 7.5})
fig, axes = plt.subplots(4, 4, figsize=(12, 10), sharex=True)
mon = pd.concat(monthly)
for ax, st in zip(axes.ravel(), data):
    y = data[st]["2012":]; g = f[f.station == st]; m = mon[mon.station == st]; model = m.model.iloc[0]
    ax.plot(y.index, y.values, color="black", lw=0.9, label="Observed")
    for o, go in g.groupby("origin"):
        ax.plot(go.index, go[model], color="tab:blue", lw=0.9, alpha=0.9, label="Rolling-origin forecast (36-month)" if o == 2011 else None)
        ax.axvline(pd.Timestamp(f"{o}-12-15"), color="grey", lw=0.5, ls=":")
    ax.fill_between(m.date, m.lower95, m.upper95, color="tab:red", alpha=0.2, label="95% PI (forecast)")
    ax.plot(m.date, m.forecast, color="tab:red", lw=1.0, label="Forecast")
    ax.set_title(f"{st} — {model}", fontsize=8); ax.grid(alpha=0.25)
for ax in axes[:, 0]: ax.set_ylabel("Max. temperature (°C)")
for ax in axes[-1]: ax.set_xlabel("Year")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.005))
plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.savefig(f"fig_station_forecasts{tag}.png", dpi=300); plt.savefig(f"fig_station_forecasts{tag}.pdf")
print(pd.DataFrame(rows)[["station", "model"] + list(SEASON)].to_string())
print(pd.DataFrame(cov_rows).round(3).to_string())
