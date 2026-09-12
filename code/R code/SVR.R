# ✅ SVR Model: Forecasting + Evaluation (like ANN/LSTM)

library(MLmetrics)
library(forecast)
library(e1071)
library(ggplot2)
library(gridExtra)

# 1. Load data and setup
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
temperature <- as.numeric(temperature_ts)  

# 2. Train-Test split
train_size <- floor(0.8 * length(temperature))
train_data <- temperature[1:train_size]
test_data <- temperature[(train_size + 1):length(temperature)]

# 3. Scaling
min_temp <- min(train_data)
max_temp <- max(train_data)
scale01 <- function(x) { (x - min_temp) / (max_temp - min_temp) }
inverse_scale <- function(x) { x * (max_temp - min_temp) + min_temp }

train_scaled <- scale01(train_data)
test_scaled <- scale01(test_data)
temperature_scaled <- scale01(temperature)

# 4. Lag features
lag_features <- function(data, timesteps) {
  X <- NULL
  y <- NULL
  for (i in (timesteps + 1):length(data)) {
    X <- rbind(X, data[(i - timesteps):(i - 1)])
    y <- c(y, data[i])
  }
  list(X = X, y = y)
}

timesteps <- 12
total_seq <- lag_features(temperature_scaled, timesteps)
X_full <- total_seq$X
y_full <- total_seq$y

# Train-test sequence split
X_train <- X_full[1:(train_size - timesteps),]
y_train <- y_full[1:(train_size - timesteps)]
X_test <- X_full[(train_size - timesteps + 1):nrow(X_full), ]
y_test <- y_full[(train_size - timesteps + 1):length(y_full)]

# 5. SVR training
set.seed(155)
tune_result <- tune.svm(
  x = X_train, y = y_train,
  type = "eps-regression",
  kernel = "radial",
  cost = 10^seq(-1, 2, by = 0.5),
  epsilon = seq(0.01, 0.2, by = 0.05),
  tunecontrol = tune.control(cross = 5)
)

svr_model <- tune_result$best.model
cat("\nOptimal Parameters:\n")
cat("Cost:", svr_model$cost, "\n")
cat("Epsilon:", svr_model$epsilon, "\n")

# 6. Prediction and metrics
predicted_test <- predict(svr_model, X_test)
predicted_test_rescaled <- inverse_scale(predicted_test)
actual_test_rescaled <- inverse_scale(y_test)

# MASE function
mase <- function(actual, predicted, seasonal_period=12) {
  naive_error <- mean(abs(diff(actual, lag=seasonal_period)))
  mean(abs(actual - predicted)) / naive_error
}
calculate_mape <- function(actual, predicted) {
  non_zero <- actual != 0 & !is.na(actual) & !is.na(predicted)
  actual <- actual[non_zero]
  predicted <- predicted[non_zero]
  if(length(actual) == 0) return(NA)
  mean(abs((actual - predicted)/actual)) * 100
}

# Metrics
residuals <- actual_test_rescaled - predicted_test_rescaled
test_metrics <- data.frame(
  RMSE = round(RMSE(predicted_test_rescaled, actual_test_rescaled), 4),
  MAE  = round(MAE(predicted_test_rescaled, actual_test_rescaled), 4),
  MAPE = round(calculate_mape(actual_test_rescaled, predicted_test_rescaled), 4),
  MASE = round(mase(actual_test_rescaled, predicted_test_rescaled), 4),
  R2   = round(cor(actual_test_rescaled, predicted_test_rescaled)^2, 4),
  ACF1 = round(Acf(residuals, plot = FALSE)$acf[2], 4)
)

cat("\nTest Set Metrics SVR:\n")
print(test_metrics)

# 7. Actual vs Predicted Plot
results <- data.frame(
  Time = (train_size + 1):length(temperature_ts),
  Actual = actual_test_rescaled,
  Predicted = predicted_test_rescaled
)

actual_vs_predicted <- ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", size = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "SVR: Actual vs Predicted (Test Set)",
       x = "Time Index", y = "Temperature (\u00B0C)") +
  theme_minimal()

# 8. Residual Diagnostics
qq_plot <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
  stat_qq(color = "blue") +
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals") +
  theme_minimal()

acf_plot <- ggAcf(residuals) + ggtitle("ACF of Residuals") + theme_minimal()
pacf_plot <- ggPacf(residuals) + ggtitle("PACF of Residuals") + theme_minimal()

resid_hist <- ggplot(data.frame(residuals = residuals), aes(x = residuals)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Residual Distribution") +
  theme_minimal()

# 9. Combined Diagnostic Plots
grid.arrange(
  actual_vs_predicted,
  qq_plot,
  grid.arrange(acf_plot, pacf_plot, resid_hist, ncol = 3),
  ncol = 1,
  heights = c(1.5, 1, 1)
)
