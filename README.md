# Hybrid SARIMA–machine-learning forecasting of monthly maximum temperature at 16 Bangladesh stations

Code and data for: Rahaman M, Holy FN, Ali MH, Yasmin S (2026) Hybrid SARIMA–Machine Learning Models for
Maximum Temperature Forecasting in Bangladesh. Scientific Reports (in revision). Zenodo DOI: 10.5281/zenodo.22724139

## What is here
- `data/newdata_1972_2025.xlsx` – monthly average maximum temperature (°C), 16 BMD stations, Jan 1972–Dec 2025,
  one sheet per station (source: BARC Climate Database Portal). 1972–2023 were used for model development;
  2024–2025 only for out-of-sample verification.
- `code/pipeline.py` – rolling-origin evaluation (4 origins: Dec 2011/2014/2017/2020; 36-month recursive
  forecasts) for SNAIVE, SARIMA, ANN, SVR, XGBoost and the SARIMA+ANN/SVR/XGBoost hybrids.
- `code/lstm_pipeline.py` – same protocol for LSTM and SARIMA+LSTM (Keras, JAX backend).
- `code/pipeline_1step.py` – one-step-ahead setting on the same folds (`--lstm` for the LSTM models).
- `code/analysis.py` – pooled metrics, Diebold–Mariano (HAC, Harvey correction) and Friedman tests,
  horizon/season/extreme-month errors, heat-map and horizon figures (`rec` or `1s` argument).
- `code/station_drivers.py` – station-characteristic analysis (Section 4.5).
- `code/final_forecast.py` – fit on full record, 36-month forecasts, split-conformal prediction intervals,
  leave-one-origin-out coverage check, seasonal table and station figure (`DATA=... UTAG=_upd` for the 1972–2025 refit).
- `code/verify_2024_2025.py` – verification of the 1972–2023-based forecasts against observed 2024–2025.
- `code/R_original/` – the R scripts used for the originally submitted analysis and for the original maps (`BD_map.R`).
- `code/seasonal_map.py`, `code/make_figs_1_3.py` – Figs 1–3, 9 and Supplementary Figs S1–S2 (Natural Earth boundary in `code/natural_earth_boundary/`, public domain).
- `results/` – all CSV tables reported in the paper and Supplement (rolling-origin forecasts and metrics, DM tests, station characteristics, PI coverage, 2024–2025 verification, 2026 forecasts).

## Reproduce
```
pip install -r requirements.txt
cd code
python pipeline.py            # ~16 min on one core
python lstm_pipeline.py       # ~30 min
python pipeline_1step.py && python pipeline_1step.py --lstm
python analysis.py rec && python analysis.py 1s
python station_drivers.py
python final_forecast.py
python seasonal_map.py --shp natural_earth_boundary/ne_10m_admin_0_countries.shp --csv ../results/table_seasonal_2026_upd.csv
python make_figs_1_3.py --shp natural_earth_boundary/ne_10m_admin_0_countries.shp --data ../data/newdata_1972_2025.xlsx
DATA=../data/newdata_1972_2025.xlsx UTAG=_upd python final_forecast.py
python seasonal_map.py --shp natural_earth_boundary/ne_10m_admin_0_countries.shp --csv ../results/table_seasonal_2026_upd.csv
python make_figs_1_3.py --shp natural_earth_boundary/ne_10m_admin_0_countries.shp --data ../data/newdata_1972_2025.xlsx
python verify_2024_2025.py
```
Seeds are fixed (`SEED = 2026`); results are reproducible to the printed precision on CPU.

## Licence
Code: MIT. Data and figures: CC BY 4.0 (underlying observations © Bangladesh Meteorological Department / BARC).
