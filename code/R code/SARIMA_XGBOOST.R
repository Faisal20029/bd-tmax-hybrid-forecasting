##XGBOOOST+SARIMA this model is used
set.seed(661)
library(forecast)
library(xgboost)
library(ggplot2)
library(Metrics)
library(gridExtra)
data<-read_excel("C:/Users/hp/Desktop/Research paper/data/newdata.xlsx", 
                 sheet = "Khulna")
# Step 1: Load example temperature data (or use your own)
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)

# 1. Data Preparation - Ensure proper time series object
if(!exists("temperature_ts") || !is.ts(temperature_ts)) {
  stop("temperature_ts must be a valid time series object")
}

# Train-Test Split (80-20) with proper time series handling
train_size <- floor(0.8 * length(temperature_ts))
train <- window(temperature_ts, end = time(temperature_ts)[train_size])
test <- window(temperature_ts, start = time(temperature_ts)[train_size + 1])

# 2. SARIMA Model with error handling
sarima_model <- tryCatch(
  {
    auto.arima(train, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
  },
  error = function(e) {
    warning("SARIMA fitting failed")
    NULL
  }
)

# Check if SARIMA produced residuals
if(length(residuals(sarima_model)) == 0) {
  stop("SARIMA model failed to produce residuals")
}
sarima_residuals <- residuals(sarima_model)

# 3. Improved XGBoost Data Preparation with feature engineering
create_xgb_data <- function(residuals, timesteps = 12) {
  if(length(residuals) < timesteps + 1) {
    stop("Not enough residuals to create features")
  }
  
  X <- matrix(NA, nrow = length(residuals) - timesteps, ncol = timesteps)
  y <- numeric(length(residuals) - timesteps)
  
  for(i in (timesteps + 1):length(residuals)) {
    X[i - timesteps, ] <- residuals[(i - timesteps):(i - 1)]
    y[i - timesteps] <- residuals[i]
  }
  
  # Add seasonal features
  month_feature <- (1:nrow(X) %% 12) + 1
  X <- cbind(X, 
             sin(2*pi*month_feature/12),
             cos(2*pi*month_feature/12))
  
  list(X = X, y = y)
}

xgb_data <- create_xgb_data(sarima_residuals)

# 4. Proper Scaling Implementation
scaler <- function(x) {
  (x - mean(x)) / sd(x)
}

X_train <- scaler(xgb_data$X)
y_train <- scaler(xgb_data$y)

# 5. XGBoost Model with improved parameters
dtrain <- xgb.DMatrix(data = X_train, label = y_train)

params <- list(
  objective = "reg:squarederror",
  eta = 0.05,  # Lower learning rate
  max_depth = 4,  # Shallower trees
  subsample = 0.7,
  colsample_bytree = 0.7,
  gamma = 0.1
)

xgb_model <- xgb.cv(
  params = params,
  data = dtrain,
  nrounds = 500,
  nfold = 5,
  early_stopping_rounds = 20,
  verbose = 0
)

best_model <- xgb.train(params, dtrain, nrounds = xgb_model$best_iteration)

# 6. Robust Hybrid Prediction Function
hybrid_predict_xgb <- function(model, sarima_model, newdata = NULL, h = NULL, timesteps = 12) {
  # Error checking
  if(is.null(newdata) && is.null(h)) {
    stop("Must provide either newdata or h")
  }
  
  # Get residuals
  model_residuals <- residuals(sarima_model)
  if(length(model_residuals) < timesteps) {
    stop("Not enough residuals available")
  }
  
  if (!is.null(newdata)) {
    # Test data prediction
    h <- length(newdata)
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(model_residuals, timesteps)
    xgb_fc <- numeric(h)
    
    for(i in 1:h) {
      input <- c(last_residuals, sin(2*pi*i/12), cos(2*pi*i/12))
      input <- matrix(input, nrow = 1)
      pred <- predict(model, newdata = input)
      xgb_fc[i] <- pred * sd(xgb_data$y) + mean(xgb_data$y)  # Reverse scaling
      last_residuals <- c(last_residuals[-1], pred)
    }
    
    return(list(
      sarima = sarima_fc$mean,
      xgb_residuals = xgb_fc,
      hybrid = sarima_fc$mean + xgb_fc,
      actual = newdata
    ))
  } else {
    # Future forecast
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(model_residuals, timesteps)
    xgb_fc <- numeric(h)
    
    for(i in 1:h) {
      input <- c(last_residuals, sin(2*pi*i/12), cos(2*pi*i/12))
      input <- matrix(input, nrow = 1)
      pred <- predict(model, newdata = input)
      xgb_fc[i] <- pred * sd(xgb_data$y) + mean(xgb_data$y)  # Reverse scaling
      last_residuals <- c(last_residuals[-1], pred)
    }
    
    return(list(
      sarima = sarima_fc$mean,
      xgb_residuals = xgb_fc,
      hybrid = sarima_fc$mean + xgb_fc
    ))
  }
}

# 7. Test Data Prediction with error handling
test_predictions <- tryCatch(
  hybrid_predict_xgb(best_model, sarima_model, newdata = test),
  error = function(e) {
    warning("Hybrid prediction failed: ", e$message)
    NULL
  }
)

if(is.null(test_predictions)) {
  stop("Failed to generate test predictions")
}

# 8. Calculate Metrics with protection against errors
calculate_metrics <- function(actual, predicted) {
  if(length(actual) != length(predicted)) {
    stop("Actual and predicted lengths differ")
  }
  
  errors <- actual - predicted
  abs_errors <- abs(errors)
  naive_err <- mean(abs(diff(actual)), na.rm = TRUE)
  
  # Handle potential division by zero in MAPE
  mape <- ifelse(all(actual == 0), NA, 
                 mean(abs_errors/abs(actual[actual != 0]), na.rm = TRUE) * 100)
  
  list(
    RMSE = sqrt(mean(errors^2, na.rm = TRUE)),
    MAE = mean(abs_errors, na.rm = TRUE),
    MAPE = mape,
    MASE = mean(abs_errors, na.rm = TRUE) / naive_err,
    R2 = 1 - (sum(errors^2, na.rm = TRUE)/sum((actual - mean(actual, na.rm = TRUE))^2, na.rm = TRUE)),
    ACF1 = tryCatch(acf(errors, plot = FALSE)$acf[2], error = function(e) NA)
  )
}

metrics <- calculate_metrics(as.numeric(test), as.numeric(test_predictions$hybrid))

cat("\nHybrid Model Metrics:\n")
cat("--------------------\n")
metrics_vector2 <- c(
  RMSE  = round(metrics$RMSE, 4),
  MAE   = round(metrics$MAE, 4),
  MAPE  = round(metrics$MAPE, 4),
  MASE  = round(metrics$MASE, 4),
  R2    = round(metrics$R2, 4),
  ACF1  = round(metrics$ACF1, 4)
)


print(metrics_vector2)

# 9. Future Forecast (36 months)
final_sarima <- Arima(temperature_ts, model = sarima_model)
future_36 <- hybrid_predict_xgb(best_model, final_sarima, h = 36)
print(future_36$hybrid)
###
# 10. Calculate 95% Confidence Intervals --------------------------------------

# Calculate residuals from test predictions
hybrid_residuals <- test - test_predictions$hybrid
residual_sd <- sd(hybrid_residuals, na.rm = TRUE)

# For future forecast
future_ci_width <- 1.96 * residual_sd  # 95% CI

# Create forecast data frame with CI
forecast_df <- data.frame(
  Date = seq.Date(
    from = as.Date(paste(end(temperature_ts)[1], end(temperature_ts)[2], "01", sep = "-")),
    by = "month",
    length.out = 36
  ),
  Forecast = as.numeric(future_36$hybrid),
  Lower = as.numeric(future_36$hybrid) - future_ci_width,
  Upper = as.numeric(future_36$hybrid) + future_ci_width
)

# Print forecast with CI
cat("\n36-Month Forecast with 95% Confidence Intervals:\n")
print(forecast_df)





# 10. Enhanced Visualization
forecast_plot <- autoplot(temperature_ts) +
  autolayer(ts(test_predictions$hybrid, start = time(test)[1], frequency = 12),
            series = "Test Forecast", color = "blue") +
  autolayer(ts(future_36$hybrid, start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", color = "red") +
  labs(title = "SARIMA-XGBoost Hybrid Forecast",
       subtitle = paste("Test Metrics:",
                        "RMSE =", round(metrics$RMSE, 3),
                        "| MAE =", round(metrics$MAE, 3),
                        "| MAPE =", round(metrics$MAPE, 3), "%"),
       y = "Temperature") +
  theme_minimal()
forecast_plot

residual_plot <- ggplot(data.frame(Residuals = as.numeric(test) - as.numeric(test_predictions$hybrid)),
                        aes(x = Residuals)) +
  geom_histogram(bins = 30, fill = "steelblue", alpha = 0.7) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals Distribution") +
  theme_minimal()
##########################################
residuals <- test - test_predictions$hybrid

# A. QQ Plot
qq_plot <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
  stat_qq(color = "blue") + 
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals",
       subtitle = "Checking normality of residuals",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_minimal()
qq_plot
# B. ACF Plot
acf_plot <- ggAcf(residuals) + 
  ggtitle("ACF of Residuals") +
  labs(subtitle = "Checking for autocorrelation") +
  theme_minimal()
acf_plot
# C. PACF Plot
pacf_plot <- ggPacf(residuals) + 
  ggtitle("PACF of Residuals") +
  labs(subtitle = "Partial autocorrelation analysis") +
  theme_minimal()
pacf_plot
# 2. Actual vs Predicted Plot ------------------------------------------------

results <- data.frame(
  Time = time(test),
  Actual = as.numeric(test),
  Predicted = test_predictions$hybrid
)

actual_vs_pred <- ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", size = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "Actual vs Predicted Values",
       subtitle = "SARIMA-XGBoost Hybrid Model Performance on Test Set",
       x = "Time", y = "Temperature (°C)",
       color = "") +
  theme_minimal() +
  theme(legend.position = "top")
actual_vs_pred

forecast_plot <- autoplot(temperature_ts) +
  autolayer(ts(future_36$hybrid, 
               start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", 
            color = "red") +
  labs(title = "36-Month Temperature Forecast",
       subtitle = "SARIMA-XGBoost Hybrid Model",
       y = "Temperature (°C)") +
  theme_minimal() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("36-Month Forecast" = "red"))

print(forecast_plot)



#############
# Create test period comparison plot
test_plot <- autoplot(temperature_ts) +
  autolayer(test, series = "Actual Test Data", color = "black") +
  autolayer(ts(test_predictions$hybrid, 
               start = time(test)[1], 
               frequency = 12),
            series = "Hybrid Predictions", 
            color = "blue") +
  labs(title = "Actual vs Predicted Temperatures (Test Period)",
       subtitle = paste("Test Metrics: RMSE =", round(metrics$RMSE, 2),
                        "MAE =", round(metrics$MAE, 2)),
       y = "Temperature (°C)") +
  scale_color_manual(values = c("Actual Test Data" = "black", 
                                "Hybrid Predictions" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top")

print(test_plot)
# Create future forecast plot


###
# Create test period comparison plot (actual vs predicted only)
# Simplified test comparison plot
ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 0.8) +
  geom_line(aes(y = Predicted, color = "Predicted"), 
            linetype = "dashed", linewidth = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "Actual vs Predicted Temperature (Test Period)",
       x = "Time (Months)", 
       y = "Temperature (°C)", 
       color = "") +
  theme_minimal() +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold", size = 14),
        axis.title = element_text(size = 12))
#################
library(forecast)
library(ggplot2)
library(scales) # For date formatting

# 1. Convert time indices to proper dates
time_to_date <- function(times, frequency) {
  years <- floor(times)
  months <- floor((times - years) * frequency + 1)
  as.Date(paste(years, months, "1", sep = "-"))
}

# For test period
test_dates <- time_to_date(time(test), frequency = 12)
# For forecast period
last_date <- time_to_date(end(temperature_ts), frequency = 12)
forecast_dates <- seq(last_date + months(1), by = "month", length.out = 36)

# 2. Test Period Comparison (autoplot version)
test_plot <- autoplot(temperature_ts) +
  autolayer(test, series = "Actual Test Data", color = "black") +
  autolayer(ts(test_predictions$hybrid, 
               start = time(test)[1], 
               frequency = 12),
            series = "Hybrid Predictions", 
            color = "blue") +
  labs(title = "Actual vs Predicted Temperatures (Test Period)",
       subtitle = paste("Test Metrics: RMSE =", round(metrics$RMSE, 2),
                        "MAE =", round(metrics$MAE, 2)),
       y = "Temperature (°C)") +
  scale_color_manual(values = c("Actual Test Data" = "black", 
                                "Hybrid Predictions" = "blue")) +
  theme_minimal() +
  theme(legend.position = "top")

# 3. Future Forecast Plot (autoplot version)
forecast_plot <- autoplot(temperature_ts) +
  autolayer(ts(future_36$hybrid, 
               start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", 
            color = "red") +
  labs(title = "36-Month Temperature Forecast",
       subtitle = "SARIMA-XGBoost Hybrid Model",
       y = "Temperature (°C)") +
  theme_minimal() +
  theme(legend.position = "top") +
  scale_color_manual(values = c("36-Month Forecast" = "red"))

# 4. Enhanced Test Comparison (ggplot version)
test_comparison_plot <- ggplot() +
  geom_line(data = data.frame(
    Date = test_dates,
    Temperature = as.numeric(test)
  ), aes(x = Date, y = Temperature, color = "Actual Test Data"), linewidth = 0.5) +
  geom_line(data = data.frame(
    Date = test_dates,
    Temperature = as.numeric(test_predictions$hybrid)
  ), aes(x = Date, y = Temperature, color = "Hybrid Predictions"), 
  linewidth = 1, linetype = "dashed") +
  labs(title = "Actual vs Predicted Temperatures (Test Period)",
       subtitle = paste("RMSE:", round(metrics$RMSE, 2), 
                        "| MAE:", round(metrics$MAE, 2),
                        "| MAPE:", round(metrics$MAPE, 2), "%"),
       y = "Temperature (°C)",
       x = "Date") +
  scale_color_manual(name = "",
                     values = c("Actual Test Data" = "black", 
                                "Hybrid Predictions" = "blue")) +
  scale_x_date(labels = date_format("%Y-%m"), date_breaks = "6 months") +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top",
        plot.title = element_text(face = "bold"),
        panel.grid.minor = element_blank(),
        axis.text.x = element_text(angle = 45, hjust = 1))

# Print all plots
print(test_plot)
print(forecast_plot)
print(test_comparison_plot)

