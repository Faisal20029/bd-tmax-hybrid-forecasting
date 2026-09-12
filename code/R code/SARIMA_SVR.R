#SARIMA+SVR(nijer code)

# Load required packages
library(forecast)
library(e1071)  # For SVR
library(ggplot2)
library(Metrics)
library(gridExtra)

# Set random seed for reproducibility

## 1. Data Preparation --------------------------------------------------------
# Assuming you have a time series object called 'temperature_ts'
# If not, create one from your data:
# temperature_ts <- ts(your_data_vector, frequency=12)

# Split into train and test sets
train_size <- floor(0.8 * length(temperature_ts))
train <- window(temperature_ts, end = time(temperature_ts)[train_size])
test <- window(temperature_ts, start = time(temperature_ts)[train_size + 1])

## 2. SARIMA Model ------------------------------------------------------------
# Fit SARIMA model (adjust order as needed for your data)
sarima_model <- auto.arima(train,seasonal = TRUE,approximation = FALSE,stepwise = FALSE)
sarima_residuals <- residuals(sarima_model)

## 3. Prepare SVR Dataset -----------------------------------------------------
create_svr_dataset <- function(residuals, timesteps = 12) {
  X <- matrix(NA, nrow = length(residuals) - timesteps, ncol = timesteps)
  y <- numeric(length(residuals) - timesteps)
  
  for(i in (timesteps + 1):length(residuals)) {
    X[i - timesteps, ] <- residuals[(i - timesteps):(i - 1)]
    y[i - timesteps] <- residuals[i]
  }
  
  colnames(X) <- paste0("Lag", 1:timesteps)
  return(data.frame(X, y = y))
}

svr_data <- create_svr_dataset(sarima_residuals)

## 4. Normalization Functions -------------------------------------------------
normalize_data <- function(x) {
  (x - min(x)) / (max(x) - min(x) + 1e-10)
}

denormalize <- function(x, orig_min, orig_max) {
  x * (orig_max - orig_min) + orig_min
}

# Store normalization parameters
y_min <- min(svr_data$y)
y_max <- max(svr_data$y)
x_mins <- sapply(svr_data[1:12], min)
x_maxs <- sapply(svr_data[1:12], max)

# Create normalized dataset
svr_data_normalized <- as.data.frame(lapply(svr_data, normalize_data))

## 5. Train SVR Model ---------------------------------------------------------
svr_formula <- as.formula("y ~ .")
svr_model <- svm(svr_formula, data = svr_data_normalized,
                 type = "eps-regression",
                 kernel = "radial",
                 cost = 1,
                 gamma = 0.1,
                 epsilon = 0.1)

## 6. Hybrid Prediction Function ----------------------------------------------
hybrid_predict <- function(svr_model, sarima_model, svr_data, 
                           newdata = NULL, h = NULL, timesteps = 12) {
  
  # Get stored normalization parameters
  y_min <- min(svr_data$y)
  y_max <- max(svr_data$y)
  x_mins <- sapply(svr_data[1:timesteps], min)
  x_maxs <- sapply(svr_data[1:timesteps], max)
  
  if (is.null(newdata) && !is.null(h)) {
    # Future forecasting mode
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(residuals(sarima_model), timesteps)
    svr_fc <- numeric(h)
    
    for(i in 1:h) {
      # Create input with correct column names
      input <- data.frame(t(last_residuals))
      colnames(input) <- paste0("Lag", 1:timesteps)
      
      # Normalize input using training parameters
      input_norm <- as.data.frame(
        Map(function(x, m, M) (x - m)/(M - m + 1e-10),
            input, x_mins, x_maxs
        ))  # Added closing parenthesis here
      
      # Predict and denormalize
      pred <- predict(svr_model, input_norm)
      svr_fc[i] <- pred * (y_max - y_min) + y_min
      last_residuals <- c(last_residuals[-1], svr_fc[i])
    }
    
    return(list(
      sarima = sarima_fc$mean,
      svr_residuals = svr_fc,
      hybrid = sarima_fc$mean + svr_fc
    ))
  } 
  else if (!is.null(newdata)) {
    # Test data evaluation mode
    h <- length(newdata)
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(residuals(sarima_model), timesteps)
    svr_fc <- numeric(h)
    
    for(i in 1:h) {
      # Create input with correct column names
      input <- data.frame(t(last_residuals))
      colnames(input) <- paste0("Lag", 1:timesteps)
      
      # Normalize input using training parameters
      input_norm <- as.data.frame(
        Map(function(x, m, M) (x - m)/(M - m + 1e-10),
            input, x_mins, x_maxs
        ))  # Added closing parenthesis here
      
      # Predict and denormalize
      pred <- predict(svr_model, input_norm)
      svr_fc[i] <- pred * (y_max - y_min) + y_min
      last_residuals <- c(last_residuals[-1], svr_fc[i])
    }
    
    return(list(
      sarima = sarima_fc$mean,
      svr_residuals = svr_fc,
      hybrid = sarima_fc$mean + svr_fc
    ))
  } 
  else {
    stop("Either 'newdata' or 'h' must be provided")
  }
}

## 7. Model Evaluation --------------------------------------------------------
# Test the model
test_predictions <- hybrid_predict(
  svr_model = svr_model,
  sarima_model = sarima_model,
  svr_data = svr_data,
  newdata = test
)

# Calculate performance metrics
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
metrics_vector_SVR <- c(
  RMSE  = round(metrics$RMSE, 4),
  MAE   = round(metrics$MAE, 4),
  MAPE  = round(metrics$MAPE, 4),
  MASE  = round(metrics$MASE, 4),
  R2    = round(metrics$R2, 4),
  ACF1  = round(metrics$ACF1, 4)
)


print(metrics_vector_SVR)

## 8. Future Forecasting ------------------------------------------------------
# Refit SARIMA on full data
final_sarima <- Arima(temperature_ts, model = sarima_model)

# Generate 36-month forecast
future_forecast <- hybrid_predict(
  svr_model = svr_model,
  sarima_model = final_sarima,
  svr_data = svr_data,
  h = 36
)
print(future_forecast$hybrid)
########################################################95% CI
final_sarima <- Arima(temperature_ts, model = sarima_model)

# Generate SARIMA forecast with prediction intervals
sarima_fc <- forecast(final_sarima, h = 36, level = c(80, 95))

# Generate hybrid forecast
future_forecast <- hybrid_predict(
  svr_model = svr_model,
  sarima_model = final_sarima,
  svr_data = svr_data,
  h = 36
)

# Calculate hybrid confidence intervals
# We approximate by adding the SVR residuals to SARIMA's intervals
hybrid_upper_95 <- sarima_fc$upper[,"95%"] + future_forecast$svr_residuals
hybrid_lower_95 <- sarima_fc$lower[,"95%"] + future_forecast$svr_residuals
hybrid_upper_80 <- sarima_fc$upper[,"80%"] + future_forecast$svr_residuals
hybrid_lower_80 <- sarima_fc$lower[,"80%"] + future_forecast$svr_residuals

# Create time index for future forecasts
future_time <- seq(end(temperature_ts)[1] + 1/12, 
                   by = 1/12, 
                   length.out = 36)
cat("\nHybrid Forecast with Confidence Intervals:\n")
print(data.frame(
  Date = future_time,
  Forecast = future_forecast$hybrid,
  Lower_80 = hybrid_lower_80,
  Upper_80 = hybrid_upper_80,
  Lower_95 = hybrid_lower_95,
  Upper_95 = hybrid_upper_95
))

#########################################################
## 9. Visualization -----------------------------------------------------------
# Forecast plot
autoplot(temperature_ts) +
  autolayer(ts(future_forecast$hybrid, 
               start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", color = "red") +
  labs(title = "SARIMA-SVR Hybrid Forecast", y = "Temperature")

# Actual vs Predicted plot
actual_vs_pred <- data.frame(
  Date = as.numeric(time(test)),
  Actual = as.numeric(test),
  Predicted = as.numeric(test_predictions$hybrid)
)
p1 <- ggplot(actual_vs_pred, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual"),size=1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed",size=0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "Actual vs Predicted Values", y = "Temperature") +
  theme_minimal()
p1
# Residual plot
residual_data <- data.frame(
  Date = as.numeric(time(test)),
  Residual = as.numeric(test) - as.numeric(test_predictions$hybrid)
)

p2 <- ggplot(residual_data, aes(x = Date, y = Residual)) +
  geom_point(color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = FALSE, color = "green") +
  labs(title = "Residual Analysis") +
  theme_minimal()

# Combine plots
grid.arrange(p1, p2, ncol = 1)
##############
## 10. Residual Diagnostics ---------------------------------------------------
# Create residual diagnostics plots
residuals <- as.numeric(test) - as.numeric(test_predictions$hybrid)

# ACF plot
acf_plot <- ggAcf(residuals) + 
  ggtitle("ACF of Residuals") +
  theme_minimal()

# PACF plot
pacf_plot <- ggPacf(residuals) + 
  ggtitle("PACF of Residuals") +
  theme_minimal()

# QQ plot
qq_data <- data.frame(residuals = residuals)
qq_plot <- ggplot(qq_data, aes(sample = residuals)) + 
  stat_qq(color = "blue") + 
  stat_qq_line(color = "red") +
  ggtitle("QQ Plot of Residuals") +
  xlab("Theoretical Quantiles") + 
  ylab("Sample Quantiles") +
  theme_minimal()

# Histogram of residuals
hist_plot <- ggplot(qq_data, aes(x = residuals)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Distribution of Residuals") +
  theme_minimal()

# Combine all diagnostic plots
grid.arrange(acf_plot, pacf_plot, qq_plot, hist_plot, ncol = 2)

# Enhanced forecast plot
forecast_plot <- autoplot(temperature_ts, size = 0.5) +
  autolayer(ts(test_predictions$hybrid, 
               start = time(test)[1], 
               frequency = 12),
            series = "Test Forecast", 
            color = "blue",  # Nice blue color
            size = 0.5) +
  autolayer(ts(future_forecast$hybrid,  # Changed from future_36 to future_forecast to match your previous code
               start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", 
            color = "red",  # Nice orange color
            size = 0.5) +
  labs(title = "SARIMA-SVR Hybrid Forecast",
       subtitle = paste("Test Metrics:",
                        "RMSE =", round(metrics$RMSE, 3),
                        "| MAE =", round(metrics$MAE, 3),
                        "| MAPE =", round(metrics$MAPE, 3), "%"),
       y = "Temperature",
       x = "Time") +
  scale_color_manual(values = c("Test Forecast" = "blue", 
                                "36-Month Forecast" = "red"),
                     name = "Forecast") +
  theme_minimal(base_size = 12) +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(color = "black"),
        legend.position = "bottom",
        panel.grid.minor = element_blank())

# Display the plot
print(forecast_plot)
