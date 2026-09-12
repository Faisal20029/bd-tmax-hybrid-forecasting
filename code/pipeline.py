"""
Unified rolling-origin evaluation for Bangladesh monthly max-temperature forecasting.
Protocol:
  - expanding window; origins = Dec of ORIGIN_YEARS; horizon H months recursive multi-step
  - NO test-period information used in any model (scalers, order selection, tuning: train only)
  - identical metrics for all models; MASE scaled by seasonal-naive MAE on the training window
Models: SNAIVE, SARIMA, SVR, XGB, ANN, SARIMA+SVR, SARIMA+XGB, SARIMA+ANN  (LSTM handled separately)
"""
import warnings, itertools, json, sys, time
import numpy as np, pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
warnings.filterwarnings("ignore")

SEED = 2026
ORIGIN_YEARS = [2011, 2014, 2017, 2020]
H = 36
NLAG = 12

# ---------- data ----------
def load_all(path="/mnt/project/newdata.xlsx"):
    xl = pd.ExcelFile(path); out = {}
    for s in xl.sheet_names:
        d = xl.parse(s).dropna(axis=1, how="all")[["Station", "Year", "Month", "Temperature"]]
        d = d.sort_values(["Year", "Month"]).reset_index(drop=True)
        d["date"] = pd.to_datetime(dict(year=d.Year, month=d.Month, day=1))
        out[d.Station[0]] = pd.Series(d.Temperature.values, index=d.date, name=d.Station[0])
    return out

# ---------- metrics ----------
def metrics(y, f, train):
    y, f = np.asarray(y, float), np.asarray(f, float); e = y - f
    snaive_mae = np.mean(np.abs(train[12:] - train[:-12]))
    ac1 = np.corrcoef(e[:-1], e[1:])[0, 1] if len(e) > 2 else np.nan
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e**2)),
                MAPE=100*np.mean(np.abs(e/y)), MASE=np.mean(np.abs(e))/snaive_mae,
                R2=1-np.sum(e**2)/np.sum((y-y.mean())**2), ACF1=ac1)

# ---------- SARIMA ----------
def select_order(train, grid=None):
    """AICc grid search on training data only. d=0, D=1, s=12 (as in original study)."""
    best = None
    ps, qs, Ps, Qs = range(3), range(3), range(3), range(2)
    for p, q, P, Q in itertools.product(ps, qs, Ps, Qs):
        try:
            m = SARIMAX(train, order=(p, 0, q), seasonal_order=(P, 1, Q, 12),
                        trend="c" if True else None, enforce_stationarity=True,
                        enforce_invertibility=True).fit(disp=False, maxiter=200)
            k = m.params.size; n = m.nobs
            aicc = m.aic + 2*k*(k+1)/max(n-k-1, 1)
            if best is None or aicc < best[0]:
                best = (aicc, (p, 0, q), (P, 1, Q, 12))
        except Exception:
            pass
    return best[1], best[2]

def fit_sarima(train, order, sorder):
    return SARIMAX(train, order=order, seasonal_order=sorder, trend="c",
                   enforce_stationarity=True, enforce_invertibility=True).fit(disp=False, maxiter=300)

# ---------- ML feature construction (calendar-correct seasonal features) ----------
def make_xy(series, months, nlag=NLAG):
    """series: 1-D array; months: calendar month (1..12) aligned with series."""
    X, y = [], []
    for t in range(nlag, len(series)):
        m = months[t]
        X.append(np.r_[series[t-nlag:t], np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)])
        y.append(series[t])
    return np.array(X), np.array(y)

def recursive_forecast(model, sx, sy, last_vals, last_month, h, nlag=NLAG):
    hist = list(last_vals); out = []; m = last_month
    for i in range(h):
        m = m % 12 + 1
        x = np.r_[hist[-nlag:], np.sin(2*np.pi*m/12), np.cos(2*np.pi*m/12)][None, :]
        p = model.predict(sx.transform(x))
        p = sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()[0]
        out.append(p); hist.append(p)
    return np.array(out)

def fit_ml(kind, X, y, seed=SEED):
    sx, sy = StandardScaler().fit(X), StandardScaler().fit(y.reshape(-1, 1))
    Xs, ys = sx.transform(X), sy.transform(y.reshape(-1, 1)).ravel()
    if kind == "SVR":
        grid = {"C": 10**np.arange(-1, 2.5, 0.5), "epsilon": [0.01, 0.05, 0.1, 0.15, 0.2], "gamma": ["scale"]}
        gs = GridSearchCV(SVR(kernel="rbf"), grid, cv=TimeSeriesSplit(5),
                          scoring="neg_mean_absolute_error", n_jobs=-1).fit(Xs, ys)
        model = gs.best_estimator_
    elif kind == "XGB":
        nval = max(int(0.1*len(Xs)), 24)
        model = xgb.XGBRegressor(objective="reg:squarederror", learning_rate=0.05, max_depth=4,
                                 subsample=0.7, colsample_bytree=0.7, gamma=0.1, n_estimators=500,
                                 early_stopping_rounds=20, random_state=seed, n_jobs=4)
        model.fit(Xs[:-nval], ys[:-nval], eval_set=[(Xs[-nval:], ys[-nval:])], verbose=False)
    elif kind == "ANN":
        model = MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
                             max_iter=500, early_stopping=True, validation_fraction=0.1,
                             n_iter_no_change=15, random_state=seed).fit(Xs, ys)
    return model, sx, sy

# ---------- one fold ----------
def run_fold(y, origin_year, order, sorder, station, log):
    train = y[: f"{origin_year}-12-01"]; test = y[f"{origin_year+1}-01-01":].iloc[:H]
    tr, mo_tr = train.values, train.index.month.values
    res = {}; fc = {}
    # benchmark: seasonal naive
    fc["SNAIVE"] = np.array([tr[len(tr)-12 + (i % 12)] for i in range(H)])
    # SARIMA
    t0 = time.time(); sar = fit_sarima(tr, order, sorder); sf = sar.get_forecast(H)
    fc["SARIMA"] = sf.predicted_mean; sar_pi = sf.conf_int(alpha=0.05)
    log(f"  SARIMA fit {time.time()-t0:.1f}s")
    # standalone ML
    X, Y = make_xy(tr, mo_tr)
    for k in ["SVR", "XGB", "ANN"]:
        t0 = time.time(); m, sx, sy = fit_ml(k, X, Y)
        fc[k] = recursive_forecast(m, sx, sy, tr[-NLAG:], mo_tr[-1], H)
        log(f"  {k} {time.time()-t0:.1f}s")
    # hybrids: SARIMA in-sample residuals (train only) -> ML residual learner -> recursive
    resid = np.asarray(sar.resid)[13:]  # drop initialisation burn-in
    mo_res = mo_tr[13:]
    Xr, Yr = make_xy(resid, mo_res)
    for k in ["SVR", "XGB", "ANN"]:
        m, sx, sy = fit_ml(k, Xr, Yr)
        rf = recursive_forecast(m, sx, sy, resid[-NLAG:], mo_res[-1], H)
        fc[f"SARIMA+{k}"] = fc["SARIMA"] + rf
    rows = []
    for k, f in fc.items():
        mt = metrics(test.values, f, tr)
        rows.append(dict(station=station, origin=origin_year, model=k, **mt))
    fdf = pd.DataFrame({k: np.asarray(v) for k, v in fc.items()}, index=test.index)
    fdf.insert(0, "actual", test.values); fdf.insert(0, "origin", origin_year); fdf.insert(0, "station", station)
    fdf["sarima_lo"], fdf["sarima_hi"] = np.asarray(sar_pi)[:, 0], np.asarray(sar_pi)[:, 1]
    return pd.DataFrame(rows), fdf

def main(stations=None):
    data = load_all(); stations = stations or list(data)
    all_m, all_f, orders = [], [], {}
    log = lambda s: (print(s), sys.stdout.flush())
    for st in stations:
        y = data[st]
        t0 = time.time()
        order, sorder = select_order(y[: f"{ORIGIN_YEARS[0]}-12-01"].values)  # selected on first training window only
        orders[st] = dict(order=order, sorder=sorder)
        log(f"{st}: SARIMA{order}{sorder}  [{time.time()-t0:.0f}s]")
        for oy in ORIGIN_YEARS:
            log(f" origin {oy}")
            m, f = run_fold(y, oy, order, sorder, st, log)
            all_m.append(m); all_f.append(f)
        pd.concat(all_m).to_csv("results_metrics.csv", index=False)
        pd.concat(all_f).to_csv("results_forecasts.csv")
        json.dump({k: dict(order=list(v["order"]), sorder=list(v["sorder"])) for k, v in orders.items()},
                  open("sarima_orders.json", "w"), indent=1)

if __name__ == "__main__":
    main(sys.argv[1:] or None)
