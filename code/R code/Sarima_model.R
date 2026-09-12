# ✅ SARIMA Model: Evaluation + Final Forecast Combined

library(forecast)
library(ggplot2)
library(gridExtra)
library(readxl)

# 1. Load full data
data <- read_excel("C:/Users/hp/OneDrive/Desktop/Research paper/data/newdata.xlsx", 
                   sheet = "dhaka")
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
temperature_ts <- window(temperature_ts, end = c(2023, 12))  # ensure data ends at 2023

# ============================================
# ✅ PART A: MODEL EVALUATION (80/20 Split)
# ============================================

# 2. Train-test split
train_size <- floor(0.8 * length(temperature_ts))
train_ts <- window(temperature_ts, end = time(temperature_ts)[train_size])
test_ts <- window(temperature_ts, start = time(temperature_ts)[train_size + 1])

# 3. Fit SARIMA on training set
sarima_model_eval <- auto.arima(train_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
ord <- arimaorder(sarima_model_eval)

cat(
  paste0(
    "ARIMA(",
    ord["p"], ",", ord["d"], ",", ord["q"], ")(",
    ord["P"], ",", ord["D"], ",", ord["Q"], ")[12]"
  )
)
# 4. Forecast on test set
forecast_eval <- forecast(sarima_model_eval, h = length(test_ts))
preds <- forecast_eval$mean
residuals_test <- as.numeric(test_ts) - as.numeric(preds)

# 5. Evaluation metrics
metrics <- data.frame(
  RMSE = round(sqrt(mean(residuals_test^2)), 4),
  MAE = round(mean(abs(residuals_test)), 4),
  MAPE = round(mean(abs(residuals_test / as.numeric(test_ts)), na.rm = TRUE) * 100, 4),
  MASE = round(mean(abs(residuals_test)) / mean(abs(diff(train_ts, lag = 12))), 4),
  R2 = round(1 - (sum(residuals_test^2) / sum((as.numeric(test_ts) - mean(as.numeric(test_ts)))^2)), 4),
  ACF1 = round(Acf(residuals_test, plot = FALSE)$acf[2], 4)
)

cat("\nSARIMA Test Set Metrics:\n")
print(metrics)

# 6. Prepare ts objects for plot

# For plotting test forecast
test_forecast_ts <- ts(preds,
                       start = time(test_ts)[1],
                       frequency = frequency(temperature_ts))

# ============================================
# ✅ PART B: FINAL FORECAST (2024–2026)
# ============================================

# 7. Fit SARIMA on full dataset
sarima_model_final <- auto.arima(temperature_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)

# 8. Forecast 36 months
forecast_future <- forecast(sarima_model_final, h = 36)
future_forecast_ts <- ts(forecast_future$mean, start = c(2024, 1), frequency = 12)

# 9. Combined Forecast Plot (like ANN/LSTM style)
forecast_plot <- autoplot(temperature_ts) +
  autolayer(test_forecast_ts, series = "Test Forecast", color = "blue") +
  autolayer(future_forecast_ts, series = "36-Month Forecast", color = "red") +
  labs(title = "SARIMA Temperature Forecast",
       subtitle = paste("Test Metrics:",
                        "RMSE =", round(metrics$RMSE, 3),
                        "| MAE =", round(metrics$MAE, 3),
                        "| MAPE =", round(metrics$MAPE, 3), "%"),
       y = "Temperature (\u00B0C)") +
  scale_color_manual(values = c("Test Forecast" = "blue", 
                                "36-Month Forecast" = "red"),
                     name = "Forecast") +
  theme_minimal() +
  theme(legend.position = "bottom")
forecast_plot
# 10. Actual vs Predicted (Test)
results <- data.frame(
  Time = (train_size + 1):length(temperature_ts),
  Actual = as.numeric(test_ts),
  Predicted = as.numeric(preds)
)

actual_vs_predicted <- ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", size = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "SARIMA: Actual vs Predicted (Test Set)",
       x = "Time Index", y = "Temperature (\u00B0C)") +
  theme_minimal()
actual_vs_predicted
# 11. Residual diagnostics
qq_plot <- ggplot(data.frame(residuals = residuals_test), aes(sample = residuals)) +
  stat_qq(color = "blue") +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals") +
  theme_minimal()
qq_plot
acf_plot <- ggAcf(residuals_test) + ggtitle("ACF of Residuals") + theme_minimal()
acf_plot
pacf_plot <- ggPacf(residuals_test) + ggtitle("PACF of Residuals") + theme_minimal()
pacf_plot
resid_hist <- ggplot(data.frame(residuals = residuals_test), aes(x = residuals)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Residual Distribution") +
  theme_minimal()

# 12.

# 13. Optional: Print future forecast values
forecast_table <- data.frame(
  Month = seq(as.Date("2024-01-01"), by = "month", length.out = 36),
  Forecast = as.numeric(forecast_future$mean),
  Lower95 = forecast_future$lower[, "95%"],
  Upper95 = forecast_future$upper[, "95%"]
)

print(forecast_table)
forecast_plot <- autoplot(temperature_ts) +
  autolayer(future_forecast_ts, series = "36-Month Forecast", color = "red") +
  labs(title = "SARIMA: 36-Month Forecast (2024–2026)",
       subtitle = paste("SARIMA Model"),
       y = "Temperature (°C)") +
  scale_color_manual(values = c("36-Month Forecast" = "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")
forecast_plot


