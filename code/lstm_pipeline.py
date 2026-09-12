import os, sys, json, time, warnings
os.environ["KERAS_BACKEND"] = "jax"
import numpy as np, pandas as pd, keras
from keras import layers, callbacks
from pipeline import load_all, metrics, fit_sarima, select_order, ORIGIN_YEARS, H, SEED
warnings.filterwarnings("ignore")
TS = 12

def seqs(x):
    X = np.array([x[i-TS:i] for i in range(TS, len(x))]); y = x[TS:]
    return X[..., None], y

def build(kind):
    m = keras.Sequential([layers.Input((TS, 1))])
    if kind == "LSTM":
        m.add(layers.LSTM(128, return_sequences=True)); m.add(layers.BatchNormalization())
        m.add(layers.Dropout(0.2)); m.add(layers.LSTM(64)); m.add(layers.Dropout(0.2))
    else:  # residual learner, as in original SARIMA+LSTM script
        m.add(layers.LSTM(50, return_sequences=True)); m.add(layers.Dropout(0.2)); m.add(layers.LSTM(50))
        m.add(layers.Dropout(0.2))
    m.add(layers.Dense(1))
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return m

def fit_lstm(kind, series, seed=SEED):
    keras.utils.set_random_seed(seed)
    lo, hi = series.min(), series.max()            # scaling on training window only
    s = (series - lo) / (hi - lo)
    X, y = seqs(s); nval = max(int(0.1 * len(X)), 24)
    m = build(kind)
    cb = [callbacks.EarlyStopping(patience=15, restore_best_weights=True),
          callbacks.ReduceLROnPlateau(factor=0.1, patience=5)]
    m.fit(X[:-nval], y[:-nval], validation_data=(X[-nval:], y[-nval:]), epochs=200, batch_size=32,
          verbose=0, callbacks=cb)
    hist = list(s[-TS:]); out = []
    for _ in range(H):
        p = float(m.predict(np.array(hist[-TS:])[None, :, None], verbose=0)[0, 0]); out.append(p); hist.append(p)
    return np.array(out) * (hi - lo) + lo, (np.array(hist[:0]))

def main(stations=None):
    data = load_all(); stations = stations or list(data)
    orders = json.load(open("sarima_orders.json")) if os.path.exists("sarima_orders.json") else {}
    rows, fcs = [], []
    if os.path.exists("results_metrics_lstm.csv"):
        rows = pd.read_csv("results_metrics_lstm.csv").to_dict("records")
        fcs = [pd.read_csv("results_forecasts_lstm.csv", index_col=0, parse_dates=True)]
    done = {(r["station"], r["origin"]) for r in rows}
    for st in stations:
        y = data[st]
        if st in orders: order, sorder = tuple(orders[st]["order"]), tuple(orders[st]["sorder"])
        else: order, sorder = select_order(y[: f"{ORIGIN_YEARS[0]}-12-01"].values)
        for oy in ORIGIN_YEARS:
            if (st, oy) in done: continue
            t0 = time.time()
            train = y[: f"{oy}-12-01"]; test = y[f"{oy+1}-01-01":].iloc[:H]; tr = train.values
            f_l, _ = fit_lstm("LSTM", tr)
            sar = fit_sarima(tr, order, sorder); sf = sar.get_forecast(H).predicted_mean
            resid = np.asarray(sar.resid)[13:]
            f_r, _ = fit_lstm("RES", resid)
            fc = {"LSTM": f_l, "SARIMA+LSTM": np.asarray(sf) + f_r}
            for k, f in fc.items():
                rows.append(dict(station=st, origin=oy, model=k, **metrics(test.values, f, tr)))
            fdf = pd.DataFrame(fc, index=test.index); fdf.insert(0, "actual", test.values)
            fdf.insert(0, "origin", oy); fdf.insert(0, "station", st); fcs.append(fdf)
            print(f"{st} {oy} LSTM MAE={rows[-2]['MAE']:.3f} S+LSTM MAE={rows[-1]['MAE']:.3f} [{time.time()-t0:.0f}s]", flush=True)
            pd.DataFrame(rows).to_csv("results_metrics_lstm.csv", index=False)
            pd.concat(fcs).to_csv("results_forecasts_lstm.csv")

if __name__ == "__main__":
    main(sys.argv[1:] or None)
