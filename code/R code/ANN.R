####ANN(1)
library(keras)
library(tidyverse)
library(Metrics)
library(ggplot2)
library(MLmetrics)
set.seed(129)
tensorflow::tf$random$set_seed(129)
temperature <- as.numeric(temperature_ts)
# Custom MASE function for time series
mase <- function(actual, predicted, seasonal_period=12) {
  naive_error <- mean(abs(diff(actual, lag=seasonal_period)))
  mean(abs(actual - predicted)) / naive_error
}
# ২. Feature creation function
create_features <- function(data, timesteps=15) {
  X <- matrix(NA, nrow=length(data)-timesteps, ncol=timesteps+2)
  y <- numeric(length(data)-timesteps)
  
  for(i in (timesteps+1):length(data)) {
    X[i-timesteps, 1:timesteps] <- data[(i-timesteps):(i-1)]
    month <- (i %% 12) + 1
    season <- ceiling(month / 3)
    X[i-timesteps, timesteps+1] <- month
    X[i-timesteps, timesteps+2] <- season
    y[i-timesteps] <- data[i]
  }
  
  list(X=X, y=y)
}

# ৩. ফিচার তৈরী
timesteps <- 15
features <- create_features(temperature, timesteps)

# ৪. ট্রেন-টেস্ট ভাগ
train_size <- floor(0.8 * nrow(features$X))
train_X <- features$X[1:train_size,]
train_y <- features$y[1:train_size]
test_X <- features$X[(train_size+1):nrow(features$X),]
test_y <- features$y[(train_size+1):nrow(features$X)]

# ৫. ✅ Scaling শুধুমাত্র ট্রেন ডেটা থেকে
min_temp <- min(train_y)
max_temp <- max(train_y)

scale_temp <- function(x) (x - min_temp) / (max_temp - min_temp)
inv_scale <- function(x) x * (max_temp - min_temp) + min_temp

train_y_scaled <- scale_temp(train_y)
test_y_scaled <- scale_temp(test_y)

# ANN-এর জন্য input ফিচারকেও scale করা (সব ভেরিয়েবল না হলে চলবে)
train_X_scaled <- apply(train_X, 2, scale_temp)
test_X_scaled <- apply(test_X, 2, scale_temp)

# ৬. Keras মডেল তৈরি
model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(train_X_scaled)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = "adam",
  loss = "mse"
)

history <- model %>% fit(
  train_X_scaled, train_y_scaled,
  epochs = 100,
  batch_size = 16,
  validation_split = 0.1,
  verbose = 1
)

# ৭. Test Evaluation
preds_scaled <- model %>% predict(test_X_scaled)
preds <- inv_scale(preds_scaled)

metrics <- data.frame(
  RMSE = round(rmse(test_y, preds),4),
  MAE = round(mae(test_y, preds),4),
  MAPE = round(MAPE(preds, test_y)*100,4),  # Percentage format
  MASE = round(mase(test_y, preds),4),
  R2 = round(cor(test_y, preds)^2,4),
  ACF1 = round(Acf(residuals, plot=FALSE)$acf[2],4)
)
cat("\nTest Set Metrics ANN:\n")
print(metrics)

# ৮. Visualization
results <- data.frame(
  Time = (timesteps + train_size + 1):length(temperature),
  Actual = test_y,
  Predicted = preds
)

ggplot(results, aes(x=Time)) +
  geom_line(aes(y=Actual, color="Actual"),size=1) +
  geom_line(aes(y=Predicted, color="Predicted"), linetype="dashed",size=0.8) +
  scale_color_manual(values=c("Actual"="blue", "Predicted"="red")) +
  labs(title="ANN: Actual vs Predicted", x="month", y="Temperature(°C)") +
  theme_minimal()

# ৯. Future Forecast (36 মাস)
last_data <- tail(temperature, timesteps)
future_preds <- numeric(36)

for(i in 1:36) {
  month <- (length(temperature) + i) %% 12 + 1
  season <- ceiling(month / 3)
  input <- c(last_data, month, season)
  input_scaled <- matrix(scale_temp(input), nrow = 1)  
  pred_scaled <- model %>% predict(input_scaled)
  pred <- inv_scale(pred_scaled)
  future_preds[i] <- pred
  last_data <- c(last_data[-1], pred)
}


forecast_df <- data.frame(
  Month = 1:36,
  Temperature = future_preds,
  Lower = future_preds - 1.96 * sd(test_y - preds),  # 95% CI
  Upper = future_preds + 1.96 * sd(test_y - preds)
)
print(forecast_df)

ggplot(forecast_df, aes(x=Month, y=Temperature)) +
  geom_line(color="darkgreen") +
  labs(title="ANN: ৩৬ মাসের তাপমাত্রা পূর্বাভাস", x="মাস", y="তাপমাত্রা (°C)") +
  theme_minimal()
# Residual analysis
residuals <- test_y - preds
ggAcf(residuals) + 
  ggtitle("ACF of residuals") +
  theme_minimal()

ggPacf(residuals) + 
  ggtitle("PACF of residuals") +
  theme_minimal()

autoplot(temperature_ts) +
  autolayer(ts(forecast_df$Temperature, 
               start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", color = "red") +
  labs(title = "ANN Forecast", y = "Temperature")
###############
# 10. Enhanced Visualization with QQ Plot and Combined Forecasts -----------------

# A. QQ Plot for Residual Analysis
qq_plot <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
  stat_qq(color = "blue") + 
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_minimal()

# B. Create proper time series objects for plotting
test_forecast_ts <- ts(preds, 
                       start = time(temperature_ts)[train_size + timesteps + 1],
                       frequency = frequency(temperature_ts))

future_forecast_ts <- ts(forecast_df$Temperature,
                         start = end(temperature_ts)[1] + 1/12,
                         frequency = frequency(temperature_ts))

# C. Create combined forecast plot with CI
forecast_plot <- autoplot(temperature_ts) +
  autolayer(test_forecast_ts, series = "Test Forecast", color = "blue") +
  autolayer(future_forecast_ts, series = "36-Month Forecast", color = "red") +
  labs(title = "ANN Temperature Forecast",
       subtitle = paste("Test Metrics:",
                        "RMSE =", round(metrics$RMSE, 3),
                        "| MAE =", round(metrics$MAE, 3),
                        "| MAPE =", round(metrics$MAPE, 3), "%"),
       y = "Temperature (°C)") +
  scale_color_manual(values = c("Test Forecast" = "blue", 
                                "36-Month Forecast" = "red"),
                     name = "Forecast") +
  theme_minimal() +
  theme(legend.position = "bottom")

# D. Actual vs Predicted Plot (Enhanced)
actual_vs_predicted <- ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", size = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "ANN: Actual vs Predicted Values",
       x = "Time Index",
       y = "Temperature (°C)") +
  theme_minimal()

# E. Residual Diagnostics Plots
acf_plot <- ggAcf(residuals) + 
  ggtitle("ACF of Residuals") +
  theme_minimal()

pacf_plot <- ggPacf(residuals) + 
  ggtitle("PACF of Residuals") +
  theme_minimal()

resid_hist <- ggplot(data.frame(residuals = residuals), aes(x = residuals)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Residual Distribution") +
  theme_minimal()

# F. Combine all diagnostic plots
library(gridExtra)
diagnostic_plots <- grid.arrange(
  actual_vs_predicted,
  qq_plot,
  forecast_plot,
  grid.arrange(acf_plot, pacf_plot, resid_hist, ncol = 3),
  ncol = 1,
  heights = c(1, 1, 1.5, 1)
)

# Print all plots
print(diagnostic_plots)


##########################SVR
# 1.
library(MLmetrics)
library(forecast)
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
temperature <- as.numeric(temperature_ts) 
# ট্রেন-টেস্ট স্প্লিট (80% ট্রেনিং, 20% টেস্ট)
train_size <- floor(0.8 * length(temperature))
train_data <- temperature[1:train_size]
test_data <- temperature[(train_size + 1):length(temperature)]  # টেস্ট ডেটা

# 2. স্কেলিং (Min-Max Scaling)
min_temp <- min(train_data)
max_temp <- max(train_data)
scale01 <- function(x) { (x - min_temp) / (max_temp - min_temp) }
inverse_scale <- function(x) { x * (max_temp - min_temp) + min_temp }

train_scaled <- scale01(train_data)  # ট্রেন ডেটা স্কেল করা হচ্ছে
test_scaled <- scale01(test_data)   # টেস্ট ডেটা স্কেল করা হচ্ছে

# 3. Lag Features তৈরি করা
lag_features <- function(data, timesteps) {
  X <- NULL
  y <- NULL
  for (i in (timesteps + 1):length(data)) {
    X <- rbind(X, data[(i - timesteps):(i - 1)])  # Lag features
    y <- c(y, data[i])  # Actual value to predict
  }
  list(X = X, y = y)
}

timesteps <- 12  # 12 মাসের হিস্ট্রি
train_seq <- lag_features(train_scaled, timesteps)
test_seq <- lag_features(test_scaled, timesteps)

# 4. SVR ইনপুট তৈরি করা
X_train <- train_seq$X
y_train <- train_seq$y
X_test <- test_seq$X
y_test <- test_seq$y

# 5. Hyperparameter Tuning with Grid Search
library(e1071)
set.seed(155)  # রিপ্রোডিউসিবিলিটির জন্য

tune_grid <- list(
  cost = 10^seq(-1, 2, by = 0.5),    # 0.1 থেকে 100 পর্যন্ত
  epsilon = seq(0.01, 0.2, by = 0.05) # 0.01 থেকে 0.2 পর্যন্ত
)
tune_result <- tune.svm(
  x = X_train,
  y = y_train,
  type = "eps-regression",
  kernel = "radial",
  cost = 10^seq(-1, 2, by = 0.5),    # সরাসরি cost পাস করুন
  epsilon = seq(0.01, 0.2, by = 0.05), # সরাসরি epsilon পাস করুন
  tunecontrol = tune.control(cross = 5)
)

svr_model <- tune_result$best.model

# অপ্টিমাল প্যারামিটার প্রিন্ট করুন
cat("Optimal Parameters Found:\n")
cat("Cost (C):", svr_model$cost, "\n")
cat("Epsilon (ε):", svr_model$epsilon, "\n")

# 6. টেস্ট ডেটার উপর পূর্বাভাস
predicted_test <- predict(svr_model, X_test)

# 7. মডেল ইভ্যালুয়েশন ------------------------------------------------------
predicted_test_rescaled <- inverse_scale(predicted_test)
actual_test_rescaled <- inverse_scale(y_test)

# Custom MASE function
mase <- function(actual, predicted, seasonal_period=12) {
  naive_error <- mean(abs(diff(actual, lag=seasonal_period)))
  mean(abs(actual - predicted)) / naive_error
}
# Best practice implementation
calculate_mape <- function(actual, predicted) {
  # Handle division by zero and NA cases
  non_zero <- actual != 0 & !is.na(actual) & !is.na(predicted)
  actual <- actual[non_zero]
  predicted <- predicted[non_zero]
  
  if(length(actual) == 0) return(NA)  # All zero actuals
  
  mean(abs((actual - predicted)/actual)) * 100
}

# Usage
MAPE_value <- calculate_mape(actual_test_rescaled, predicted_test_rescaled)
# Final metrics calculation
test_metrics <- data.frame(
  RMSE = RMSE(predicted_test_rescaled, actual_test_rescaled),
  MAE = MAE(predicted_test_rescaled, actual_test_rescaled),
  MAPE = calculate_mape(actual_test_rescaled, predicted_test_rescaled), # Using safe function
  MASE = mase(actual_test_rescaled, predicted_test_rescaled),
  R2 = cor(actual_test_rescaled, predicted_test_rescaled)^2,
  ACF1 = Acf(actual_test_rescaled - predicted_test_rescaled, plot=FALSE)$acf[2]
)

# Print with percentage sign
cat("\nTest Set Metrics SVR:\n")
print(test_metrics)

# Residual analysis
residuals <- actual_test_rescaled - predicted_test_rescaled
ggAcf(residuals) + 
  ggtitle("ACF of Residuals") +
  theme_minimal()

# Actual vs Predicted প্লট করা
ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), 
            linetype = "dashed", size = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "Actual vs Predicted Temperature (Test Set)",
       x = "Time (Months)", y = "Temperature", color = "") +
  theme_minimal() +
  theme(legend.position = "top")

# 9. Future Forecasting (36 months) ---------------------------------------

last_sequence <- tail(test_scaled, timesteps)  # স্কেল করা test ডেটা থেকে শেষ সিকোয়েন্স
future_preds <- numeric(36)  # 36 months forecast

for (i in 1:36) {
  input_array <- matrix(last_sequence, nrow = 1)
  next_pred <- predict(svr_model, input_array)
  future_preds[i] <- next_pred
  last_sequence <- c(last_sequence[-1], next_pred)
}

# স্কেল ইনভার্স করে আসল ইউনিটে আনা
future_preds_rescaled <- inverse_scale(future_preds)

# Forecast dataframe
forecast_df <- data.frame(
  Month = (length(temperature) + 1):(length(temperature) + 36),
  Forecast = future_preds_rescaled
)

# Plotting Forecast
ggplot() +
  geom_line(data = data.frame(Time = 1:length(temperature), Temp = temperature), 
            aes(x = Time, y = Temp, color = "Historical")) +
  geom_line(data = forecast_df, aes(x = Month, y = Forecast, color = "Forecast"), 
            linetype = "dashed") +
  scale_color_manual(values = c("Historical" = "blue", "Forecast" = "red")) +
  labs(title = "36-Month Temperature Forecast", 
       x = "Time (Months)", y = "Temperature (°C)") +
  theme_minimal()

# Print the forecasted values (next 36 months)
print(forecast_df)


