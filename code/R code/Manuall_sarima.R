# ============================================================
# Manual SARIMA Model: Evaluation + Final Forecast Combined
# You can manually change (p,d,q)(P,D,Q)[12]
# ============================================================

library(forecast)
library(ggplot2)
library(gridExtra)
library(readxl)

# ------------------------------------------------------------
# 1. Load full data
# ------------------------------------------------------------
data <- read_excel(
  "C:/Users/hp/OneDrive/Desktop/Research paper/data/newdata.xlsx",
  sheet = "Bogra"
)

temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
temperature_ts <- window(temperature_ts, end = c(2023, 12))

# ------------------------------------------------------------
# 2. Manually set SARIMA order here
# Example: SARIMA(0,0,3)(0,1,1)[12]
# Change these values according to your selected model
# ------------------------------------------------------------
p <- 1
d <- 0
q <- 0

P <- 2
D <- 1
Q <- 1

s <- 12

# ------------------------------------------------------------
# 3. Train-test split: 80/20
# ------------------------------------------------------------
train_size <- floor(0.8 * length(temperature_ts))

train_ts <- window(
  temperature_ts,
  end = time(temperature_ts)[train_size]
)

test_ts <- window(
  temperature_ts,
  start = time(temperature_ts)[train_size + 1]
)

# ------------------------------------------------------------
# 4. Fit manual SARIMA on training set
# ------------------------------------------------------------
sarima_model_eval <- Arima(
  train_ts,
  order = c(p, d, q),
  seasonal = list(order = c(P, D, Q), period = s),
  include.mean = FALSE
)

cat("\nManual SARIMA model fitted on training data:\n")
print(sarima_model_eval)

cat("\nInformation criteria for evaluation model:\n")
cat("AIC  =", AIC(sarima_model_eval), "\n")
cat("BIC  =", BIC(sarima_model_eval), "\n")
cat("AICc =", sarima_model_eval$aicc, "\n")

# ------------------------------------------------------------
# 5. Forecast on test set
# ------------------------------------------------------------
forecast_eval <- forecast(sarima_model_eval, h = length(test_ts))

preds <- forecast_eval$mean
residuals_test <- as.numeric(test_ts) - as.numeric(preds)

# ------------------------------------------------------------
# 6. Evaluation metrics
# ------------------------------------------------------------
metrics <- data.frame(
  RMSE = round(sqrt(mean(residuals_test^2, na.rm = TRUE)), 4),
  MAE  = round(mean(abs(residuals_test), na.rm = TRUE), 4),
  MAPE = round(mean(abs(residuals_test / as.numeric(test_ts)), na.rm = TRUE) * 100, 4),
  MASE = round(
    mean(abs(residuals_test), na.rm = TRUE) /
      mean(abs(diff(train_ts, lag = 12)), na.rm = TRUE),
    4
  ),
  R2 = round(
    1 - (
      sum(residuals_test^2, na.rm = TRUE) /
        sum((as.numeric(test_ts) - mean(as.numeric(test_ts), na.rm = TRUE))^2, na.rm = TRUE)
    ),
    4
  ),
  ACF1 = round(Acf(residuals_test, plot = FALSE)$acf[2], 4)
)

cat("\nManual SARIMA Test Set Metrics:\n")
print(metrics)

# ------------------------------------------------------------
# 7. Residual diagnostic check
# ------------------------------------------------------------
cat("\nLjung-Box test for residual autocorrelation:\n")
print(Box.test(residuals_test, lag = 24, type = "Ljung-Box"))

# Optional visual residual diagnostics
qq_plot <- ggplot(data.frame(residuals = residuals_test), aes(sample = residuals)) +
  stat_qq(color = "blue") +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals") +
  theme_minimal()

acf_plot <- ggAcf(residuals_test) +
  ggtitle("ACF of Residuals") +
  theme_minimal()

pacf_plot <- ggPacf(residuals_test) +
  ggtitle("PACF of Residuals") +
  theme_minimal()

resid_hist <- ggplot(data.frame(residuals = residuals_test), aes(x = residuals)) +
  geom_histogram(aes(y = after_stat(density)), bins = 30,
                 fill = "skyblue", color = "black") +
  geom_density(color = "red", linewidth = 1) +
  ggtitle("Residual Distribution") +
  theme_minimal()

grid.arrange(acf_plot, pacf_plot, qq_plot, resid_hist, ncol = 2)

# ------------------------------------------------------------
# 8. Prepare test forecast time series for plotting
# ------------------------------------------------------------
test_forecast_ts <- ts(
  preds,
  start = time(test_ts)[1],
  frequency = frequency(temperature_ts)
)

# ------------------------------------------------------------
# 9. Actual vs Predicted plot for test set
# ------------------------------------------------------------
results <- data.frame(
  Time = (train_size + 1):length(temperature_ts),
  Actual = as.numeric(test_ts),
  Predicted = as.numeric(preds)
)

actual_vs_predicted <- ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.9) +
  geom_line(aes(y = Predicted, color = "Predicted"),
            linetype = "dashed", linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(
    title = "Manual SARIMA: Actual vs Predicted (Test Set)",
    x = "Time Index",
    y = "Temperature (°C)",
    color = ""
  ) +
  theme_minimal() +
  theme(legend.position = "top")

print(actual_vs_predicted)

# ------------------------------------------------------------
# 10. Fit the same manual SARIMA order on full data
#     for 36-month final forecast
# ------------------------------------------------------------
sarima_model_final <- Arima(
  temperature_ts,
  order = c(p, d, q),
  seasonal = list(order = c(P, D, Q), period = s),
  include.mean = FALSE
)

cat("\nManual SARIMA model fitted on full data:\n")
print(sarima_model_final)

# ------------------------------------------------------------
# 11. Forecast 36 months: 2024–2026
# ------------------------------------------------------------
forecast_future <- forecast(sarima_model_final, h = 36, level = c(80, 95))

future_forecast_ts <- ts(
  forecast_future$mean,
  start = c(2024, 1),
  frequency = 12
)

# ------------------------------------------------------------
# 12. Combined historical + test forecast + future forecast plot
# ------------------------------------------------------------
forecast_plot <- autoplot(temperature_ts) +
  autolayer(test_forecast_ts, series = "Test Forecast", color = "blue") +
  autolayer(future_forecast_ts, series = "36-Month Forecast", color = "red") +
  labs(
    title = paste0(
      "Manual SARIMA(",
      p, ",", d, ",", q, ")(",
      P, ",", D, ",", Q, ")[", s, "] Temperature Forecast"
    ),
    subtitle = paste(
      "Test Metrics:",
      "RMSE =", round(metrics$RMSE, 3),
      "| MAE =", round(metrics$MAE, 3),
      "| MAPE =", round(metrics$MAPE, 3), "%"
    ),
    y = "Temperature (°C)"
  ) +
  scale_color_manual(
    values = c(
      "Test Forecast" = "blue",
      "36-Month Forecast" = "red"
    ),
    name = "Forecast"
  ) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(forecast_plot)

# ------------------------------------------------------------
# 13. Future forecast table
# ------------------------------------------------------------
forecast_table <- data.frame(
  Month = seq(as.Date("2024-01-01"), by = "month", length.out = 36),
  Forecast = round(as.numeric(forecast_future$mean), 4),
  Lower80 = round(forecast_future$lower[, "80%"], 4),
  Upper80 = round(forecast_future$upper[, "80%"], 4),
  Lower95 = round(forecast_future$lower[, "95%"], 4),
  Upper95 = round(forecast_future$upper[, "95%"], 4)
)

cat("\n36-Month Manual SARIMA Forecast Table:\n")
print(forecast_table)

# Optional: save forecast table
write.csv(
  forecast_table,
  "Manual_SARIMA_36_Month_Forecast.csv",
  row.names = FALSE
)

# ------------------------------------------------------------
# 14. Future forecast only plot
# ------------------------------------------------------------
future_only_plot <- autoplot(temperature_ts) +
  autolayer(future_forecast_ts, series = "36-Month Forecast", color = "red") +
  labs(
    title = paste0(
      "SARIMA(",
      p, ",", d, ",", q, ")(",
      P, ",", D, ",", Q, ")[", s, "] Forecast for 2024–2026"
    ),
    y = "Temperature (°C)"
  ) +
  scale_color_manual(values = c("36-Month Forecast" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")

print(future_only_plot)