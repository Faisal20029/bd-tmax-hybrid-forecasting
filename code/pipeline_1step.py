"""One-step-ahead setting: models fitted on the training window only (parameters, scalers, tuning);
during the 36-month test window forecasts are made 1 month ahead using observed lags, without refitting.
Same folds, same metrics as the recursive setting."""
import os, sys, json, time, warnings
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np, pandas as pd
from pipeline import load_all, metrics, fit_sarima, select_order, make_xy, fit_ml, ORIGIN_YEARS, H, NLAG
warnings.filterwarnings("ignore")
DO_LSTM = "--lstm" in sys.argv

def ml_onestep(model, sx, sy, full, months, n_test):
    X, _ = make_xy(full, months)                     # features built from observed values (train+test)
    p = model.predict(sx.transform(X[-n_test:]))
    return sy.inverse_transform(np.asarray(p).reshape(-1, 1)).ravel()

def lstm_onestep(kind, train, full, n_test):
    import keras
    from lstm_pipeline import build, seqs, TS, SEED
    keras.utils.set_random_seed(SEED)
    lo, hi = train.min(), train.max(); s_tr = (train-lo)/(hi-lo); s_full = (full-lo)/(hi-lo)
    X, y = seqs(s_tr); nval = max(int(0.1*len(X)), 24); m = build(kind)
    cb = [keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True), keras.callbacks.ReduceLROnPlateau(factor=0.1, patience=5)]
    m.fit(X[:-nval], y[:-nval], validation_data=(X[-nval:], y[-nval:]), epochs=200, batch_size=32, verbose=0, callbacks=cb)
    Xf, _ = seqs(s_full)
    return m.predict(Xf[-n_test:], verbose=0).ravel()*(hi-lo)+lo

def main(stations=None):
    data = load_all(); stations = stations or list(data)
    orders = json.load(open("sarima_orders.json"))
    rows, fcs = [], []
    tag = "lstm" if DO_LSTM else "main"
    for st in stations:
        y = data[st]; order, sorder = tuple(orders[st]["order"]), tuple(orders[st]["sorder"])
        for oy in ORIGIN_YEARS:
            t0 = time.time()
            train = y[: f"{oy}-12-01"]; test = y[f"{oy+1}-01-01":].iloc[:H]
            tr = train.values; full = np.r_[tr, test.values]; months = np.r_[train.index.month.values, test.index.month.values]
            sar = fit_sarima(tr, order, sorder)
            sar_full = sar.apply(full, refit=False)          # same parameters, extended with observed data
            fv = np.asarray(sar_full.fittedvalues)           # one-step-ahead predictions
            resid_full = np.asarray(sar_full.resid)
            fc = {}
            fc["SNAIVE"] = full[-H-12:-12]
            fc["SARIMA"] = fv[-H:]
            if DO_LSTM:
                fc["LSTM"] = lstm_onestep("LSTM", tr, full, H)
                fc["SARIMA+LSTM"] = fc["SARIMA"] + lstm_onestep("RES", resid_full[13:len(tr)], resid_full[13:], H)
            else:
                X, Y = make_xy(tr, train.index.month.values)
                resid_tr = resid_full[13:len(tr)]; mo_res = months[13:len(tr)]
                Xr, Yr = make_xy(resid_tr, mo_res)
                for k in ["SVR", "XGB", "ANN"]:
                    m, sx, sy = fit_ml(k, X, Y); fc[k] = ml_onestep(m, sx, sy, full, months, H)
                    mr, sxr, syr = fit_ml(k, Xr, Yr)
                    fc[f"SARIMA+{k}"] = fc["SARIMA"] + ml_onestep(mr, sxr, syr, resid_full[13:], months[13:], H)
            for k, f in fc.items():
                rows.append(dict(station=st, origin=oy, model=k, **metrics(test.values, f, tr)))
            fdf = pd.DataFrame(fc, index=test.index); fdf.insert(0, "actual", test.values)
            fdf.insert(0, "origin", oy); fdf.insert(0, "station", st); fcs.append(fdf)
            print(f"{st} {oy} " + " ".join(f"{k}={metrics(test.values, f, tr)['MAE']:.3f}" for k, f in fc.items()) + f" [{time.time()-t0:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv(f"results1s_metrics_{tag}.csv", index=False)
            pd.concat(fcs).to_csv(f"results1s_forecasts_{tag}.csv")

if __name__ == "__main__":
    main([a for a in sys.argv[1:] if not a.startswith("--")] or None)
