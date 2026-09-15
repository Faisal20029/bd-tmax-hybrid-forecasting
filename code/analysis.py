import numpy as np, pandas as pd, os, warnings
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")
LAB = {"Chittagong": "Chattogram", "Barisal": "Barishal", "Jessore": "Jashore", "Bogra": "Bogura", "Comilla": "Cumilla", "Srimongal": "Sreemangal"}
H = 36
MODELS = ["SNAIVE", "SARIMA", "ANN", "SVR", "XGB", "LSTM", "SARIMA+ANN", "SARIMA+SVR", "SARIMA+XGB", "SARIMA+LSTM"]

# ---------------- load ----------------
import sys
SET = sys.argv[1] if len(sys.argv) > 1 else "rec"     # "rec" = 36-month recursive, "1s" = one-step-ahead
FMAIN, FLSTM = ("results_forecasts.csv", "results_forecasts_lstm.csv") if SET == "rec" else ("results1s_forecasts_main.csv", "results1s_forecasts_lstm.csv")
TAG = "" if SET == "rec" else "_1s"
f = pd.read_csv(FMAIN, index_col=0, parse_dates=True)
if os.path.exists(FLSTM):
    fl = pd.read_csv(FLSTM, index_col=0, parse_dates=True)
    key = ["station", "origin"]
    fl.index.name = "date"; f.index.name = "date"
    f = f.reset_index().merge(fl.reset_index()[["date"] + key + ["LSTM", "SARIMA+LSTM"]], on=["date"] + key, how="left").set_index("date")
    f = f.drop(columns=[c for c in ["SNAIVE_y"] if c in f.columns])
models = [m for m in MODELS if m in f.columns]
f["h"] = f.groupby(["station", "origin"]).cumcount() + 1
HDM = 36 if SET == "rec" else 1

# ---------------- pooled metrics (all 144 test months per station) ----------------
def snaive_scale():
    from pipeline import load_all
    d = load_all(); return {s: np.mean(np.abs(v.values[12:len(v[:'2011-12-01'])] - v.values[:len(v[:'2011-12-01'])-12])) for s, v in d.items()}
scale = snaive_scale()
rows = []
for (st), g in f.groupby("station"):
    y = g.actual.values
    for m in models:
        e = y - g[m].values
        rows.append(dict(station=st, model=m, MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)),
                         MAPE=100*np.mean(np.abs(e/y)), MASE=np.mean(np.abs(e))/scale[st],
                         R2=1-np.sum(e**2)/np.sum((y-y.mean())**2),
                         ACF1=np.corrcoef(e[:-1], e[1:])[0, 1],
                         MAE_h1_12=np.mean(np.abs(e[g.h.values <= 12])),
                         MAE_h13_36=np.mean(np.abs(e[g.h.values > 12])),
                         MAE_summer=np.mean(np.abs(e[g.index.month.isin([3, 4, 5, 6])])),
                         MAE_top10=np.mean(np.abs(e[y >= np.quantile(y, 0.9)]))))
pooled = pd.DataFrame(rows)
pooled.to_csv("table_pooled_metrics"+TAG+".csv", index=False)

# ---------------- Diebold-Mariano (absolute-error loss, HAC variance, Harvey correction) ----------------
def dm_test(e1, e2, h=None):
    h = h or HDM
    d = np.abs(e1) - np.abs(e2); T = len(d); dbar = d.mean()
    lag = h - 1
    gamma = [np.sum((d[k:]-dbar)*(d[:T-k]-dbar))/T for k in range(lag+1)]
    var = gamma[0] + 2*sum((1-k/(lag+1))*gamma[k] for k in range(1, lag+1))  # Bartlett kernel
    var = max(var, 1e-12)
    dm = dbar/np.sqrt(var/T)
    dm *= np.sqrt((T+1-2*h+h*(h-1)/T)/T)            # Harvey, Leybourne & Newbold (1997)
    p = 2*stats.t.sf(abs(dm), T-1)
    return dm, p

dm_rows = []
for st, g in f.groupby("station"):
    y = g.actual.values
    rank = pooled[pooled.station == st].sort_values("MAE")
    best, second = rank.model.iloc[0], rank.model.iloc[1]
    dm, p = dm_test(y-g[best].values, y-g[second].values)
    dmn, pn = dm_test(y-g[best].values, y-g["SNAIVE"].values)
    dms, ps = dm_test(y-g[best].values, y-g["SARIMA"].values) if best != "SARIMA" else (np.nan, np.nan)
    dm_rows.append(dict(station=st, best=best, best_MAE=rank.MAE.iloc[0], second=second, second_MAE=rank.MAE.iloc[1],
                        dMAE=rank.MAE.iloc[1]-rank.MAE.iloc[0], DM_best_vs_2nd=dm, p_best_vs_2nd=p,
                        DM_best_vs_SNAIVE=dmn, p_best_vs_SNAIVE=pn, DM_best_vs_SARIMA=dms, p_best_vs_SARIMA=ps))
dm_df = pd.DataFrame(dm_rows); dm_df.to_csv("table_dm_tests"+TAG+".csv", index=False)

# full pairwise DM p-value matrix per station (for supplementary)
pw = []
for st, g in f.groupby("station"):
    y = g.actual.values
    for a in models:
        for b in models:
            if a < b:
                dm, p = dm_test(y-g[a].values, y-g[b].values)
                pw.append(dict(station=st, model_a=a, model_b=b, DM=dm, p=p))
pd.DataFrame(pw).to_csv("table_dm_pairwise"+TAG+".csv", index=False)

# ---------------- summary across stations ----------------
summ = pooled.groupby("model")[["MAE", "RMSE", "MASE", "R2", "ACF1", "MAE_h1_12", "MAE_h13_36", "MAE_summer", "MAE_top10"]].mean()
summ["mean_rank"] = pooled.assign(r=pooled.groupby("station").MAE.rank()).groupby("model").r.mean()
summ["n_best"] = pooled.loc[pooled.groupby("station").MAE.idxmin()].model.value_counts()
summ = summ.fillna({"n_best": 0}).sort_values("MAE"); summ.to_csv("table_summary_models"+TAG+".csv")

# Friedman test across stations (blocks = stations)
mat = pooled.pivot(index="station", columns="model", values="MAE")[models]
fr = stats.friedmanchisquare(*[mat[m].values for m in models])

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
hm = pooled.pivot(index="station", columns="model", values="MAE")[models]; hm.index = [LAB.get(i, i) for i in hm.index]
plt.figure(figsize=(8.5, 6))
sns.heatmap(hm, annot=True, fmt=".2f", cmap="viridis_r", cbar_kws={"label": "MAE (°C)"})
plt.title(("36-month recursive" if SET=="rec" else "One-step-ahead") + " out-of-sample MAE (°C), rolling origins 2012–2023")
plt.ylabel(""); plt.xlabel(""); plt.tight_layout(); plt.savefig("fig_mae_heatmap"+TAG+".png", dpi=300); plt.savefig("fig_mae_heatmap"+TAG+".pdf")

# horizon profile
hp = f.melt(id_vars=["station", "origin", "h", "actual"], value_vars=models, var_name="model", value_name="fc")
hp["ae"] = (hp.actual - hp.fc).abs()
hpm = hp.groupby(["h", "model"]).ae.mean().unstack()
plt.figure(figsize=(7, 4))
for m in models: plt.plot(hpm.index, hpm[m], label=m, lw=1.4 if "SARIMA" in m or m == "SNAIVE" else 1)
plt.xlabel("Forecast horizon (months ahead)"); plt.ylabel("MAE (°C), mean over 16 stations × 4 origins")
plt.legend(ncol=2, fontsize=7); plt.grid(alpha=.3); plt.tight_layout(); plt.savefig("fig_horizon_mae"+TAG+".png", dpi=300); plt.savefig("fig_horizon_mae"+TAG+".pdf")

with open("analysis_summary"+TAG+".txt", "w") as fh:
    fh.write("MODEL SUMMARY (mean over 16 stations, pooled 144 test months each)\n"); fh.write(summ.round(3).to_string()+"\n\n")
    fh.write(f"Friedman test across stations: chi2={fr.statistic:.2f}, p={fr.pvalue:.4g}\n\n")
    fh.write(f"STATION-WISE BEST vs SECOND (DM test, HAC lag={HDM-1}, Harvey corr.)\n"); fh.write(dm_df.round(3).to_string()+"\n")
print(open("analysis_summary"+TAG+".txt").read())
