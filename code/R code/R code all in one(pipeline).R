# =====================================================================================
# R cross-check pipeline — same protocol as the Python analysis (pipeline.py)
#   expanding-window rolling origin (Dec 2011/2014/2017/2020), 36-month recursive forecasts,
#   no test-period information used, identical metrics for every model, DM tests.
# Models: SNAIVE, SARIMA, ANN, SVR, XGB, SARIMA+ANN, SARIMA+SVR, SARIMA+XGB  (+ LSTM optional, keras)
# Packages: readxl, forecast, e1071, xgboost, nnet, dplyr   (LSTM block: keras)
# NOTE: written to mirror pipeline.py; not executed in the authors' Python environment — run and
#       report any error. Numbers will differ slightly from the Python results (different optimisers/seeds).
# =====================================================================================
suppressPackageStartupMessages({library(readxl); library(forecast); library(e1071); library(xgboost); library(nnet); library(dplyr)})
set.seed(2026); CAP_HITS <- 0
args <- commandArgs(trailingOnly = TRUE)
DATA_FILE <- if (length(args) >= 1) args[1] else {
  cand <- c("data/newdata_1972_2025.xlsx", "../data/newdata_1972_2025.xlsx", "newdata_1972_2025.xlsx"); cand[file.exists(cand)][1] }
if (is.na(DATA_FILE)) stop("Data file not found: pass the path as the first argument, e.g.  Rscript pipeline_R.R ../data/newdata_1972_2025.xlsx")
ORIGIN_YEARS <- c(2011, 2014, 2017, 2020)
H            <- 36
NLAG         <- 12
END_DEV      <- 2023          # development data end (2024-2025 only for verification)

# ---------- data ----------
load_all <- function(path = DATA_FILE) {
  sheets <- excel_sheets(path); out <- list()
  for (s in sheets) {
    d <- read_excel(path, sheet = s); d <- d[, c("Station", "Year", "Month", "Temperature")]
    d <- d[order(d$Year, d$Month), ]
    out[[d$Station[1]]] <- ts(d$Temperature, start = c(d$Year[1], d$Month[1]), frequency = 12)
  }
  out
}

# ---------- metrics (identical to Python: MASE scaled by seasonal-naive MAE on the training window) ----------
metrics <- function(y, f, train) {
  e <- y - f; sn <- mean(abs(train[-(1:12)] - train[1:(length(train) - 12)]))
  c(MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MAPE = 100 * mean(abs(e / y)),
    MASE = mean(abs(e)) / sn, R2 = 1 - sum(e^2) / sum((y - mean(y))^2),
    ACF1 = cor(e[-length(e)], e[-1]))
}

# ---------- SARIMA ----------
select_order <- function(train) {   # AICc grid on the first training window only; d=0, D=1, s=12, drift
  best <- NULL
  for (p in 0:2) for (q in 0:2) for (P in 0:2) for (Q in 0:1) {
    m <- tryCatch(Arima(train, order = c(p, 0, q), seasonal = c(P, 1, Q), include.drift = TRUE, method = "ML"), error = function(e) NULL)
    if (!is.null(m) && (is.null(best) || m$aicc < best$aicc)) best <- list(aicc = m$aicc, order = c(p, 0, q), seasonal = c(P, 1, Q))
  }
  best
}
fit_sarima <- function(train, ord) Arima(train, order = ord$order, seasonal = ord$seasonal, include.drift = TRUE, method = "ML")

# ---------- ML features (calendar-correct seasonal terms) ----------
make_xy <- function(x, months, nlag = NLAG) {
  n <- length(x); X <- matrix(NA, n - nlag, nlag + 2); y <- x[(nlag + 1):n]
  for (t in (nlag + 1):n) X[t - nlag, ] <- c(x[(t - nlag):(t - 1)], sin(2 * pi * months[t] / 12), cos(2 * pi * months[t] / 12))
  list(X = X, y = y)
}
scale_fit <- function(X, y) list(mx = colMeans(X), sx = apply(X, 2, sd) + 1e-9, my = mean(y), sy = sd(y) + 1e-9)
scale_X <- function(X, s) sweep(sweep(X, 2, s$mx), 2, s$sx, "/")

fit_ml <- function(kind, X, y) {
  s <- scale_fit(X, y); Xs <- scale_X(X, s); ys <- (y - s$my) / s$sy
  if (kind == "SVR") {
    tc <- tune.control(cross = 5)   # note: random CV folds in e1071; Python used time-series CV
    tn <- tune(svm, Xs, ys, kernel = "radial", ranges = list(cost = 10^seq(-1, 2, 0.5), epsilon = c(0.01, 0.05, 0.1, 0.15, 0.2)), tunecontrol = tc)
    model <- tn$best.model
  } else if (kind == "XGB") {
    nval <- max(floor(0.1 * nrow(Xs)), 24); ntr <- nrow(Xs) - nval
    dtr <- xgb.DMatrix(Xs[1:ntr, ], label = ys[1:ntr]); dva <- xgb.DMatrix(Xs[(ntr + 1):nrow(Xs), ], label = ys[(ntr + 1):nrow(Xs)])
    model <- xgb.train(params = list(objective = "reg:squarederror", eta = 0.05, max_depth = 4, subsample = 0.7, colsample_bytree = 0.7, gamma = 0.1),
                       data = dtr, nrounds = 500, watchlist = list(val = dva), early_stopping_rounds = 20, verbose = 0)
  } else if (kind == "ANN") {
    # nnet has one hidden layer (Python used 64-32 with early stopping). Choose size/decay on the LAST 10 % of the
    # training window (chronological, as in Python) to avoid the divergence seen with a fixed decay under 36-step recursion.
    nval <- max(floor(0.1 * nrow(Xs)), 24); ntr <- nrow(Xs) - nval; best <- NULL
    rng <- .Random.seed   # keep the global RNG stream unaffected by the ANN's fixed seed
    for (sz in c(8, 16, 32)) for (dc in c(0.01, 0.05, 0.1)) {
      set.seed(2026); m <- nnet(Xs[1:ntr, ], ys[1:ntr], size = sz, linout = TRUE, decay = dc, maxit = 500, trace = FALSE)
      v <- mean(abs(as.numeric(predict(m, Xs[(ntr + 1):nrow(Xs), ])) - ys[(ntr + 1):nrow(Xs)]))
      if (is.null(best) || v < best$v) best <- list(v = v, size = sz, decay = dc)
    }
    set.seed(2026); model <- nnet(Xs, ys, size = best$size, linout = TRUE, decay = best$decay, maxit = 500, trace = FALSE)
    .Random.seed <<- rng
  }
  list(model = model, kind = kind, s = s)
}
predict_ml <- function(m, x) {
  xs <- scale_X(matrix(x, 1), m$s)
  p <- if (m$kind == "XGB") predict(m$model, xgb.DMatrix(xs)) else as.numeric(predict(m$model, xs))
  if (m$kind == "ANN" && abs(p) > 4) { CAP_HITS <<- CAP_HITS + 1; p <- sign(p) * 4 }   # ANN only: cap at ±4 training SDs; count how often it binds
  p * m$s$sy + m$s$my
}
recursive_forecast <- function(m, last_vals, last_month, h, nlag = NLAG) {
  hist <- as.numeric(last_vals); out <- numeric(h); mo <- last_month
  for (i in 1:h) { mo <- mo %% 12 + 1; x <- c(tail(hist, nlag), sin(2 * pi * mo / 12), cos(2 * pi * mo / 12)); out[i] <- predict_ml(m, x); hist <- c(hist, out[i]) }
  out
}

# ---------- one fold ----------
run_fold <- function(y, oy, ord, station) {
  train <- window(y, end = c(oy, 12)); test <- window(y, start = c(oy + 1, 1), end = c(oy + 3, 12))
  tr <- as.numeric(train); mo <- cycle(train); fc <- list()
  fc$SNAIVE <- as.numeric(snaive(train, h = H)$mean)
  sar <- fit_sarima(train, ord); sf <- forecast(sar, h = H); fc$SARIMA <- as.numeric(sf$mean)
  xy <- make_xy(tr, mo)
  for (k in c("SVR", "XGB", "ANN")) { m <- fit_ml(k, xy$X, xy$y); fc[[k]] <- recursive_forecast(m, tail(tr, NLAG), tail(mo, 1), H) }
  res <- as.numeric(residuals(sar))[-(1:13)]; mo_r <- mo[-(1:13)]; xyr <- make_xy(res, mo_r)
  for (k in c("SVR", "XGB", "ANN")) { m <- fit_ml(k, xyr$X, xyr$y); fc[[paste0("SARIMA+", k)]] <- fc$SARIMA + recursive_forecast(m, tail(res, NLAG), tail(mo_r, 1), H) }
  rows <- do.call(rbind, lapply(names(fc), function(k) data.frame(station = station, origin = oy, model = k, t(metrics(as.numeric(test), fc[[k]], tr)))))
  fdf <- data.frame(station = station, origin = oy, date = as.numeric(time(test)), actual = as.numeric(test), as.data.frame(fc, check.names = FALSE),
                    sarima_lo = as.numeric(sf$lower[, "95%"]), sarima_hi = as.numeric(sf$upper[, "95%"]), check.names = FALSE)
  list(metrics = rows, forecasts = fdf)
}

# ---------- main (runs only when executed directly, not when source()d for its functions) ----------
if (sys.nframe() == 0) {
data <- load_all(); data <- lapply(data, function(y) window(y, end = c(END_DEV, 12)))
all_m <- list(); all_f <- list(); orders <- list()
for (st in names(data)) {
  y <- data[[st]]; ord <- select_order(window(y, end = c(ORIGIN_YEARS[1], 12))); orders[[st]] <- ord
  cat(sprintf("%s: SARIMA(%s)(%s)12\n", st, paste(ord$order, collapse = ","), paste(ord$seasonal, collapse = ",")))
  for (oy in ORIGIN_YEARS) { r <- run_fold(y, oy, ord, st); all_m[[length(all_m) + 1]] <- r$metrics; all_f[[length(all_f) + 1]] <- r$forecasts }
}
M <- bind_rows(all_m); Fc <- bind_rows(all_f)
write.csv(M, "results_metrics_R.csv", row.names = FALSE); write.csv(Fc, "results_forecasts_R.csv", row.names = FALSE)
cat(sprintf("ANN ±4 SD cap applied %d times across all recursive steps\n", CAP_HITS))

# ---------- pooled metrics per station and DM tests (best vs runner-up, best vs SARIMA, best vs SNAIVE) ----------
models <- c("SNAIVE", "SARIMA", "ANN", "SVR", "XGB", "SARIMA+ANN", "SARIMA+SVR", "SARIMA+XGB")
pooled <- Fc %>% group_by(station) %>% group_modify(function(g, key) {
  tr_scale <- { y <- data[[key$station]]; tr <- as.numeric(window(y, end = c(ORIGIN_YEARS[1], 12))); mean(abs(tr[-(1:12)] - tr[1:(length(tr) - 12)])) }
  bind_rows(lapply(models, function(m) { e <- g$actual - g[[m]]; data.frame(model = m, MAE = mean(abs(e)), RMSE = sqrt(mean(e^2)), MASE = mean(abs(e)) / tr_scale, R2 = 1 - sum(e^2) / sum((g$actual - mean(g$actual))^2)) }))
}) %>% ungroup()
write.csv(pooled, "table_pooled_metrics_R.csv", row.names = FALSE)

dm_rows <- Fc %>% group_by(station) %>% group_map(function(g, key) {
  p <- pooled %>% filter(station == key$station) %>% arrange(MAE); best <- p$model[1]; second <- p$model[2]
  e_b <- g$actual - g[[best]]; e_2 <- g$actual - g[[second]]; e_s <- g$actual - g$SARIMA; e_n <- g$actual - g$SNAIVE
  dm <- function(e1, e2) if (all(e1 == e2)) NA else dm.test(e1, e2, h = H, power = 1, alternative = "two.sided", varestimator = "bartlett")$p.value   # Bartlett HAC: never negative, no silent fallback to h=1
  data.frame(station = key$station, best = best, best_MAE = p$MAE[1], second = second, p_best_vs_2nd = dm(e_b, e_2),
             p_best_vs_SARIMA = dm(e_b, e_s), p_best_vs_SNAIVE = dm(e_b, e_n))
}) %>% bind_rows(); rownames(dm_rows) <- NULL
write.csv(dm_rows, "table_dm_tests_R.csv", row.names = FALSE); print(dm_rows)

# Friedman test across stations
mat <- pooled %>% select(station, model, MAE) %>% tidyr::pivot_wider(names_from = model, values_from = MAE)
print(friedman.test(as.matrix(mat[, models])))
}  # end main

# ---------- Split-conformal 95% PI for a station's chosen model (pooled-horizon, finite-sample corrected) ----------
conformal_pi <- function(err, alpha = 0.05) { n <- length(err); q <- function(p) sort(err)[min(ceiling((n + 1) * p), n)]; c(lo = q(alpha / 2), hi = q(1 - alpha / 2)) }
# usage: e <- Fc$actual[Fc$station == "Dhaka"] - Fc$`SARIMA+SVR`[Fc$station == "Dhaka"]; conformal_pi(e)

# ---------- OPTIONAL: LSTM / SARIMA+LSTM with keras (same architecture as the Python run) ----------
# library(keras)
# fit_lstm <- function(kind, series) {
#   lo <- min(series); hi <- max(series); s <- (series - lo) / (hi - lo); TS <- 12
#   X <- t(sapply((TS + 1):length(s), function(i) s[(i - TS):(i - 1)])); dim(X) <- c(nrow(X), TS, 1); y <- s[(TS + 1):length(s)]
#   nval <- max(floor(0.1 * nrow(X)), 24); ntr <- nrow(X) - nval
#   m <- keras_model_sequential()
#   if (kind == "LSTM") m %>% layer_lstm(128, return_sequences = TRUE, input_shape = c(TS, 1)) %>% layer_batch_normalization() %>% layer_dropout(0.2) %>% layer_lstm(64) %>% layer_dropout(0.2)
#   else               m %>% layer_lstm(50, return_sequences = TRUE, input_shape = c(TS, 1)) %>% layer_dropout(0.2) %>% layer_lstm(50) %>% layer_dropout(0.2)
#   m %>% layer_dense(1) %>% compile(optimizer = optimizer_adam(1e-3), loss = "mse")
#   m %>% fit(X[1:ntr, , , drop = FALSE], y[1:ntr], validation_data = list(X[(ntr + 1):nrow(X), , , drop = FALSE], y[(ntr + 1):nrow(X)]), epochs = 200, batch_size = 32, verbose = 0,
#             callbacks = list(callback_early_stopping(patience = 15, restore_best_weights = TRUE), callback_reduce_lr_on_plateau(factor = 0.1, patience = 5)))
#   hist <- tail(s, TS); out <- numeric(H)
#   for (i in 1:H) { p <- as.numeric(predict(m, array(tail(hist, TS), c(1, TS, 1)), verbose = 0)); out[i] <- p; hist <- c(hist, p) }
#   out * (hi - lo) + lo
# }
