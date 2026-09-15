# =====================================================================================
# figures_R.R — all figures in R (ggplot2 + forecast), for future use.
# Covers (a) the plots that were in each of the original 9 model scripts and (b) the figures of the revised paper.
# Inputs: data/newdata_1972_2025.xlsx ; outputs of pipeline_R.R (results_forecasts_R.csv, table_pooled_metrics_R.csv)
# Packages: readxl, forecast, ggplot2, dplyr, tidyr, patchwork
# Usage (after running pipeline_R.R):
#   source("figures_R.R")
#   fig_raw_series(c("Dhaka","Cox's Bazar","Rajshahi","Sylhet"))                  # Fig 3
#   fig_decomposition("Dhaka"); fig_acf_pacf("Dhaka")                             # Fig S1 / S2 style
#   fig_train_test_forecast("Dhaka", "SARIMA+SVR", origin = 2020)                 # old "Temperature Forecast" plot
#   fig_actual_vs_predicted("Dhaka", "SARIMA+SVR")                                 # old "Actual vs Predicted (Test Set)"
#   fig_residual_diagnostics("Dhaka", "SARIMA+SVR")                                # old hist + Q-Q + ACF + PACF of residuals
#   fig_mae_heatmap(); fig_horizon_mae()                                           # Fig 4 / Fig 5
#   fig_future_forecast("Dhaka", "SARIMA+SVR")                                     # old "36-Month Forecast (2024–2026)" with PI (refit on full record)
# =====================================================================================
suppressPackageStartupMessages({library(readxl); library(forecast); library(ggplot2); library(dplyr); library(tidyr); library(patchwork); library(zoo)})
theme_set(theme_minimal(base_size = 9) + theme(panel.grid.minor = element_blank()))
DATA_FILE <- if (file.exists("data/newdata_1972_2025.xlsx")) "data/newdata_1972_2025.xlsx" else "../data/newdata_1972_2025.xlsx"
H <- 36; NLAG <- 12
LABEL <- c(Chittagong = "Chattogram", Barisal = "Barishal", Jessore = "Jashore", Bogra = "Bogura", Comilla = "Cumilla", Srimongal = "Sreemangal")
lab <- function(s) ifelse(s %in% names(LABEL), LABEL[s], s)

load_all <- function(path = DATA_FILE) {
  out <- list()
  for (s in excel_sheets(path)) { d <- read_excel(path, sheet = s); d <- d[order(d$Year, d$Month), ]
    out[[d$Station[1]]] <- ts(d$Temperature, start = c(d$Year[1], d$Month[1]), frequency = 12) }
  out
}
DATA <- load_all()
ts_df <- function(x, name = "value") data.frame(date = as.Date(as.yearmon(time(x))), v = as.numeric(x)) %>% setNames(c("date", name))
save_fig <- function(p, name, w = 7, h = 4.5) { ggsave(paste0(name, ".png"), p, width = w, height = h, dpi = 300); ggsave(paste0(name, ".pdf"), p, width = w, height = h); invisible(p) }

read_fc <- function() { f <- read.csv("results_forecasts_R.csv", check.names = FALSE); f$date <- as.Date(as.yearmon(f$date)); f }

# ---------- (1) raw series, several stations (Fig 3) ----------
fig_raw_series <- function(stations, end_dev = 2023) {
  d <- bind_rows(lapply(stations, function(s) ts_df(DATA[[s]]) %>% mutate(station = lab(s), period = ifelse(as.numeric(format(date, "%Y")) <= end_dev, "1972–2023 (model development)", "2024–2025 (verification only)"))))
  p <- ggplot(d, aes(date, value, colour = period)) + geom_line(linewidth = 0.3) + facet_wrap(~station, ncol = 1, scales = "free_y") +
    scale_colour_manual(values = c("1972–2023 (model development)" = "black", "2024–2025 (verification only)" = "red"), name = NULL) +
    labs(x = "Year", y = "Monthly average maximum temperature (°C)") + theme(legend.position = "top")
  save_fig(p, "fig3_raw_series_R", 7, 8)
}

# ---------- (2) decomposition (Fig S1 style) ----------
fig_decomposition <- function(station, end_dev = 2023) {
  y <- window(DATA[[station]], end = c(end_dev, 12)); dc <- decompose(y, type = "additive")
  d <- bind_rows(ts_df(dc$x, "v") %>% mutate(comp = "Observed (°C)"), ts_df(dc$trend, "v") %>% mutate(comp = "Trend (°C)"),
                 ts_df(window(dc$seasonal, end = c(1976, 12)), "v") %>% mutate(comp = "Seasonal (°C), 1972–1976 shown (identical each year)"),
                 ts_df(dc$random, "v") %>% mutate(comp = "Remainder (°C)"))
  d$comp <- factor(d$comp, levels = unique(d$comp))
  p <- ggplot(d, aes(date, v)) + geom_line(linewidth = 0.3) + facet_wrap(~comp, ncol = 1, scales = "free") + labs(x = "Year", y = NULL, title = paste0(lab(station), ": classical additive decomposition, 1972–", end_dev))
  save_fig(p, paste0("figS1_decomposition_", gsub("[^A-Za-z]", "", station), "_R"), 7, 8)
}

# ---------- (3) ACF / PACF of the observed series (Fig S2 style) ----------
fig_acf_pacf <- function(station, end_dev = 2023, lags = 48) {
  y <- window(DATA[[station]], end = c(end_dev, 12))
  p <- (ggAcf(y, lag.max = lags) + ggtitle(paste0(lab(station), ": ACF (", lags, " lags, 95 % band)")) + labs(x = "Lag (months)")) /
       (ggPacf(y, lag.max = lags) + ggtitle("PACF") + labs(x = "Lag (months)"))
  save_fig(p, paste0("figS2_acf_pacf_", gsub("[^A-Za-z]", "", station), "_R"), 7, 6)
}

# ---------- (4) train / test / forecast plot for one station, model, origin (old "Temperature Forecast") ----------
fig_train_test_forecast <- function(station, model, origin = 2020, show_from = 2005) {
  f <- read_fc() %>% filter(station == !!station, origin == !!origin)
  tr <- ts_df(window(DATA[[station]], start = c(show_from, 1), end = c(origin, 12)))
  d <- bind_rows(tr %>% mutate(series = "Training (observed)"), data.frame(date = f$date, value = f$actual, series = "Test (observed)"), data.frame(date = f$date, value = f[[model]], series = paste0(model, " forecast")))
  mae <- mean(abs(f$actual - f[[model]]))
  cols <- setNames(c("grey40", "black", "red"), c("Training (observed)", "Test (observed)", paste0(model, " forecast")))
  d$series <- factor(d$series, levels = names(cols))
  p <- ggplot(d, aes(date, value, colour = series)) + geom_line(linewidth = 0.4) + scale_colour_manual(values = cols, name = NULL) +
    labs(title = paste0(lab(station), ": ", model, " — 36-month forecast from December ", origin), subtitle = sprintf("Test MAE = %.3f °C", mae), x = "Year", y = "Maximum temperature (°C)") + theme(legend.position = "top")
  save_fig(p, paste0("fig_fc_", gsub("[^A-Za-z]", "", station), "_", gsub("[^A-Za-z]", "", model), "_", origin, "_R"))
}

# ---------- (5) actual vs predicted on the pooled rolling-origin test months (old "Actual vs Predicted") ----------
fig_actual_vs_predicted <- function(station, model) {
  f <- read_fc() %>% filter(station == !!station); e <- f$actual - f[[model]]
  p1 <- ggplot(f, aes(date, actual)) + geom_line(colour = "black", linewidth = 0.4) + geom_line(aes(y = .data[[model]]), colour = "red", linewidth = 0.4) +
    labs(title = paste0(lab(station), ": observed (black) vs ", model, " (red), rolling-origin test months 2012–2023"), x = "Year", y = "Maximum temperature (°C)")
  p2 <- ggplot(f, aes(.data[[model]], actual)) + geom_point(size = 0.8, alpha = 0.7) + geom_abline(slope = 1, intercept = 0, colour = "red", linetype = 2) +
    labs(x = "Forecast (°C)", y = "Observed (°C)", subtitle = sprintf("MAE %.3f  RMSE %.3f  R² %.3f", mean(abs(e)), sqrt(mean(e^2)), 1 - sum(e^2) / sum((f$actual - mean(f$actual))^2))) + coord_equal()
  save_fig(p1 / p2 + plot_layout(heights = c(1, 1.2)), paste0("fig_avp_", gsub("[^A-Za-z]", "", station), "_", gsub("[^A-Za-z]", "", model), "_R"), 7, 8)
}

# ---------- (6) residual diagnostics of forecast errors: histogram, Q-Q, ACF, PACF (old residual plots) ----------
fig_residual_diagnostics <- function(station, model) {
  f <- read_fc() %>% filter(station == !!station); e <- f$actual - f[[model]]; d <- data.frame(e = e)
  p1 <- ggplot(d, aes(e)) + geom_histogram(bins = 30, fill = "steelblue", colour = "white") + labs(title = "Forecast-error distribution", x = "Error (°C)", y = "Count")
  p2 <- ggplot(d, aes(sample = e)) + stat_qq(size = 0.8) + stat_qq_line(colour = "red") + labs(title = "Q-Q plot of forecast errors", x = "Theoretical quantiles", y = "Sample quantiles")
  p3 <- ggAcf(e, lag.max = 36) + ggtitle("ACF of forecast errors (multi-step errors are serially correlated by construction)") + labs(x = "Lag (months)")
  p4 <- ggPacf(e, lag.max = 36) + ggtitle("PACF of forecast errors") + labs(x = "Lag (months)")
  p <- (p1 | p2) / (p3 | p4) + plot_annotation(title = paste0(lab(station), ": ", model, " — rolling-origin forecast errors, 2012–2023"))
  save_fig(p, paste0("fig_resid_", gsub("[^A-Za-z]", "", station), "_", gsub("[^A-Za-z]", "", model), "_R"), 9, 7)
}

# ---------- (7) station × model MAE heat-map (Fig 4) ----------
fig_mae_heatmap <- function() {
  p <- read.csv("table_pooled_metrics_R.csv", check.names = FALSE); p$station <- lab(p$station)
  p$model <- factor(p$model, levels = c("SNAIVE", "SARIMA", "ANN", "SVR", "XGB", "SARIMA+ANN", "SARIMA+SVR", "SARIMA+XGB"))
  g <- ggplot(p, aes(model, station, fill = MAE)) + geom_tile() + geom_text(aes(label = sprintf("%.2f", MAE)), size = 2.6) +
    scale_fill_viridis_c(direction = -1, name = "MAE (°C)") + labs(x = NULL, y = NULL, title = "Rolling-origin out-of-sample MAE (°C), 36-month recursive setting, 2012–2023") + theme(axis.text.x = element_text(angle = 30, hjust = 1))
  save_fig(g, "fig4_mae_heatmap_R", 8, 6)
}

# ---------- (8) MAE by forecast horizon (Fig 5) ----------
fig_horizon_mae <- function() {
  f <- read_fc() %>% mutate(h = 12 * (as.numeric(format(date, "%Y")) - origin - 1) + as.numeric(format(date, "%m")))   # horizon from the date, not row order
  models <- c("SNAIVE", "SARIMA", "ANN", "SVR", "XGB", "SARIMA+ANN", "SARIMA+SVR", "SARIMA+XGB")
  d <- f %>% pivot_longer(all_of(models), names_to = "model", values_to = "fc") %>% mutate(ae = abs(actual - fc)) %>% group_by(h, model) %>% summarise(MAE = mean(ae), .groups = "drop")
  g <- ggplot(d, aes(h, MAE, colour = model)) + geom_line() + labs(x = "Forecast horizon (months ahead)", y = "MAE (°C), mean over stations × origins", colour = NULL) + theme(legend.position = "right")
  save_fig(g, "fig5_horizon_mae_R", 7, 4)
}

# ---------- (9) future 36-month forecast with 95 % PI (old "36-Month Forecast (2024–2026)") ----------
# Refits the chosen model on the full record and adds split-conformal PIs from the rolling-origin errors (needs pipeline_R.R functions).
fig_future_forecast <- function(station, model, show_from = 2015) {
  if (!exists("fit_sarima")) source("pipeline_R.R", local = FALSE)   # main section of pipeline_R.R is guarded, so only the functions load
  y <- DATA[[station]]; ord <- select_order(window(y, end = c(2011, 12)))
  sar <- fit_sarima(y, ord); sf <- forecast(sar, h = H); base <- as.numeric(sf$mean); tr <- as.numeric(y); mo <- cycle(y)
  fc <- if (model == "SARIMA") base else {
    k <- sub("SARIMA\\+", "", model)
    if (grepl("^SARIMA\\+", model)) { res <- as.numeric(residuals(sar))[-(1:13)]; xy <- make_xy(res, mo[-(1:13)]); m <- fit_ml(k, xy$X, xy$y); base + recursive_forecast(m, tail(res, NLAG), tail(mo[-(1:13)], 1), H) }
    else { xy <- make_xy(tr, mo); m <- fit_ml(k, xy$X, xy$y); recursive_forecast(m, tail(tr, NLAG), tail(mo, 1), H) } }
  f <- read_fc() %>% filter(station == !!station); e <- f$actual - f[[model]]; q <- conformal_pi(e)
  last <- as.Date(as.yearmon(tail(time(y), 1))); fdates <- seq(last, by = "month", length.out = H + 1)[-1]
  d_obs <- ts_df(window(y, start = c(show_from, 1))); d_fc <- data.frame(date = fdates, fc = fc, lo = fc + q["lo"], hi = fc + q["hi"])
  g <- ggplot() + geom_ribbon(data = d_fc, aes(date, ymin = lo, ymax = hi), fill = "red", alpha = 0.2) + geom_line(data = d_obs, aes(date, value), linewidth = 0.4) +
    geom_line(data = d_fc, aes(date, fc), colour = "red", linewidth = 0.6) + labs(title = paste0(lab(station), ": ", model, " — 36-month forecast with 95 % split-conformal PI"), x = "Year", y = "Maximum temperature (°C)")
  save_fig(g, paste0("fig_future_", gsub("[^A-Za-z]", "", station), "_", gsub("[^A-Za-z]", "", model), "_R"))
  invisible(d_fc)
}
