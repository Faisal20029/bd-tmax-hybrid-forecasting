#########LSTM model(1) etai holo final
library(MLmetrics)
set.seed(886)
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
temperature <- as.numeric(temperature_ts)  
length(temperature)
library(keras)
library(tensorflow)
library(Metrics)
library(ggplot2)
library(forecast)  # For ACF1 and MASE

set.seed(22)  
tf$random$set_seed(22) 

# Custom MASE function
mase <- function(actual, predicted, seasonal_period=12) {
  naive_error <- mean(abs(diff(actual, lag=seasonal_period)))
  mean(abs(actual - predicted)) / naive_error
}


train_size <- floor(0.8 * length(temperature))
train_data <- temperature[1:train_size]
test_data <- temperature[(train_size + 1):length(temperature)]

min_temp <- min(train_data)  
max_temp <- max(train_data) 

scale01 <- function(x) { (x - min_temp) / (max_temp - min_temp) }
inverse_scale <- function(x) { x * (max_temp - min_temp) + min_temp }

train_scaled <- scale01(train_data)
test_scaled <- scale01(test_data)
temperature_scaled <- scale01(temperature)

create_sequences <- function(data, timesteps) {
  X <- NULL
  y <- NULL
  for (i in 1:(length(data) - timesteps)) {
    X <- rbind(X, data[i:(i + timesteps - 1)])
    y <- c(y, data[i + timesteps])
  }
  list(X = X, y = y)
}

timesteps <- 12  
train_seq <- create_sequences(train_scaled, timesteps)
test_seq <- create_sequences(test_scaled, timesteps)

# 4.
X_train <- array(train_seq$X, dim = c(nrow(train_seq$X), timesteps, 1))
y_train <- train_seq$y
X_test <- array(test_seq$X, dim = c(nrow(test_seq$X), timesteps, 1))
y_test <- test_seq$y

# 5. LSTM model
model <- keras_model_sequential()

# LSTM layer add
model %>%
  layer_lstm(units = 128, return_sequences = TRUE, input_shape = c(timesteps, 1)) %>%
  layer_batch_normalization() %>%  
  layer_dropout(rate = 0.2) %>%  
  layer_lstm(units = 64, return_sequences = FALSE) %>%  
  layer_dense(units = 1)  

# model compile
model %>% compile(
  loss = 'mean_squared_error',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = list('mae')  # MAE (Mean Absolute Error) মেট্রিক
)


summary(model)

# 6. Callbacks make
early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 15,
  restore_best_weights = TRUE 
)

reduce_lr <- callback_reduce_lr_on_plateau(
  monitor = "val_loss",
  factor = 0.1,
  patience = 5 
)

# 7. model training
history <- model %>% fit(
  X_train, y_train,
  epochs = 200,  
  batch_size = 32,  
  validation_split = 0.1,  
  callbacks = list(early_stop, reduce_lr),
  verbose = 1  
)
#8
predicted_test <- model %>% predict(X_test)
predicted_test_rescaled <- inverse_scale(predicted_test)
actual_test_rescaled <- inverse_scale(y_test)

# 9. Calculate all metrics
residuals <- actual_test_rescaled - predicted_test_rescaled
metrics <- data.frame(
  RMSE = round(rmse(actual_test_rescaled, predicted_test_rescaled), 4),
  MAE = round(mae(actual_test_rescaled, predicted_test_rescaled), 4),
  MAPE = round(MAPE(predicted_test_rescaled, actual_test_rescaled) * 100, 4),
  MASE = round(mase(actual_test_rescaled, predicted_test_rescaled), 4),
  R2 = round(cor(actual_test_rescaled, predicted_test_rescaled)^2, 4),
  ACF1 = round(Acf(residuals, plot = FALSE)$acf[2], 4)
)


cat("Comprehensive Test Set Metrics LSTM:\n")
print(metrics)


# 10. Create Time Series Objects
test_forecast_ts <- ts(predicted_test_rescaled, 
                       start = time(temperature_ts)[train_size + timesteps + 1],
                       frequency = frequency(temperature_ts))

# Future Forecast (36 Months)
last_sequence <- tail(temperature_scaled, timesteps)
future_preds <- numeric(36)

for (i in 1:36) {
  input_seq <- array(last_sequence, dim = c(1, timesteps, 1))
  pred <- model %>% predict(input_seq)
  future_preds[i] <- pred
  last_sequence <- c(last_sequence[-1], pred)
}

future_preds_rescaled <- inverse_scale(future_preds)

# Confidence Intervals
ci_width <- 1.96 * sd(residuals)
forecast_df <- data.frame(
  Month = 1:36,
  Temperature = future_preds_rescaled,
  Lower = future_preds_rescaled - ci_width,
  Upper = future_preds_rescaled + ci_width
)
print(forecast_df)
future_forecast_ts <- ts(future_preds_rescaled,
                         start = end(temperature_ts)[1] + 1/12,
                         frequency = frequency(temperature_ts))

# 11. Forecast Plot (No CI fill)
forecast_plot <- autoplot(temperature_ts) +
  autolayer(test_forecast_ts, series = "Test Forecast", color = "blue") +
  autolayer(future_forecast_ts, series = "36-Month Forecast", color = "red") +
  labs(title = "LSTM Temperature Forecast",
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

# 12. Actual vs Predicted Plot
results <- data.frame(
  Time = (train_size + timesteps + 1):length(temperature),
  Actual = actual_test_rescaled,
  Predicted = predicted_test_rescaled
)

actual_vs_predicted <- ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), size = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", size = 0.8) +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "LSTM: Actual vs Predicted Values",
       x = "Time Index",
       y = "Temperature (°C)") +
  theme_minimal()
actual_vs_predicted
# 13. Residual Diagnostics
library(gridExtra)

qq_plot <- ggplot(data.frame(residuals = residuals), aes(sample = residuals)) +
  stat_qq(color = "blue") + 
  stat_qq_line(color = "red") +
  labs(title = "Q-Q Plot of Residuals",
       x = "Theoretical Quantiles",
       y = "Sample Quantiles") +
  theme_minimal()
qq_plot
acf_plot <- ggAcf(residuals) + 
  ggtitle("ACF of Residuals") +
  theme_minimal()
acf_plot
pacf_plot <- ggPacf(residuals) + 
  ggtitle("PACF of Residuals") +
  theme_minimal()
pacf_plot
autoplot(temperature_ts) +
  autolayer(ts(forecast_df$Temperature, 
               start = end(temperature_ts)[1] + 1/12, 
               frequency = 12),
            series = "36-Month Forecast", color = "red") +
  labs(title = "LSTM Forecast", y = "Temperature")


resid_hist <- ggplot(data.frame(residuals = residuals), aes(x = residuals)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "skyblue", color = "black") +
  geom_density(color = "red", size = 1) +
  ggtitle("Residual Distribution") +
  theme_minimal()

# 14. Combine All Plots
diagnostic_plots <- grid.arrange(
  actual_vs_predicted,
  qq_plot,
  forecast_plot,
  grid.arrange(acf_plot, pacf_plot, resid_hist, ncol = 3),
  ncol = 1,
  heights = c(1, 1, 1.5, 1)
)

# Display
print(diagnostic_plots)

############################### eta holo set.seed diye
library(keras)
library(tensorflow)
library(Metrics)
library(ggplot2)

# 1. রিপ্রোডুসিবিলিটি সেটআপ ------------------------------------------------
set.seed(183L)  # L suffix যোগ করে ইন্টিজার বানানো
tf$random$set_seed(183L)
tf$keras$utils$set_random_seed(183L)

# CPU/GPU রিপ্রোডুসিবিলিটি কনফিগারেশন
session_conf <- tf$compat$v1$ConfigProto(
  intra_op_parallelism_threads = 1L,
  inter_op_parallelism_threads = 1L
)
sess <- tf$compat$v1$Session(graph = tf$compat$v1$get_default_graph(), config = session_conf)
tf$compat$v1$keras$backend$set_session(sess)

# 2. ডেটা প্রস্তুতি ----------------------------------------------------------
temperature <- as.numeric(temperature_ts)  # আপনার ডেটা লোড করুন

# ট্রেন-টেস্ট স্প্লিট (80-20)
train_size <- as.integer(floor(0.8 * length(temperature)))
train_data <- temperature[1L:train_size]
test_data <- temperature[(train_size + 1L):length(temperature)]

# স্কেলিং (শুধুমাত্র ট্রেন ডেটা ব্যবহার)
min_temp <- min(train_data)
max_temp <- max(train_data)
scale01 <- function(x) { (x - min_temp) / (max_temp - min_temp) }
inverse_scale <- function(x) { x * (max_temp - min_temp) + min_temp }

train_scaled <- scale01(train_data)
test_scaled <- scale01(test_data)

# 3. সিকোয়েন্স তৈরি ----------------------------------------------------------
create_sequences <- function(data, timesteps) {
  X <- matrix(NA_real_, nrow = length(data) - timesteps, ncol = timesteps)
  y <- rep(NA_real_, length(data) - timesteps)
  
  for (i in 1L:(length(data) - timesteps)) {
    X[i,] <- data[i:(i + timesteps - 1L)]
    y[i] <- data[i + timesteps]
  }
  list(X = X, y = y)
}

timesteps <- 12L  # 12 মাসের হিস্ট্রি
train_seq <- create_sequences(train_scaled, timesteps)
test_seq <- create_sequences(test_scaled, timesteps)

# 4. ডেটা রিশেপ -------------------------------------------------------------
X_train <- array(train_seq$X, dim = c(
  as.integer(nrow(train_seq$X)),
  as.integer(timesteps),
  1L
))

X_test <- array(test_seq$X, dim = c(
  as.integer(nrow(test_seq$X)),
  as.integer(timesteps),
  1L
))

y_train <- train_seq$y
y_test <- test_seq$y

# 5. LSTM মডেল তৈরি ---------------------------------------------------------
model <- keras_model_sequential() %>%
  layer_lstm(units = 128L, 
             return_sequences = TRUE,
             input_shape = c(timesteps, 1L),
             kernel_initializer = initializer_glorot_uniform(seed = 123L)) %>%
  layer_batch_normalization() %>%
  layer_dropout(rate = 0.2, seed = 123L) %>%
  layer_lstm(units = 64L,
             kernel_initializer = initializer_glorot_uniform(seed = 123L)) %>%
  layer_dense(units = 1L,
              kernel_initializer = initializer_glorot_uniform(seed = 123L))

# 6. মডেল কম্পাইল ----------------------------------------------------------
model %>% compile(
  loss = 'huber_loss',  # আউটলায়ার রোবাস্ট
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = list('mae')
)

# 7. কলব্যাক সেটআপ ---------------------------------------------------------
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    patience = 15L,
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.1,
    patience = 5L
  )
)

# 8. মডেল ট্রেনিং ----------------------------------------------------------
history <- model %>% fit(
  X_train, y_train,
  epochs = 200L,
  batch_size = 32L,
  validation_split = 0.1,
  callbacks = callbacks,
  verbose = 1L,
  shuffle = FALSE  # রিপ্রোডুসিবিলিটির জন্য
)

# 9. মডেল ইভ্যালুয়েশন ------------------------------------------------------
predicted_test <- model %>% predict(X_test)
predicted_test_rescaled <- inverse_scale(predicted_test)
actual_test_rescaled <- inverse_scale(y_test)

metrics <- list(
  RMSE = rmse(actual_test_rescaled, predicted_test_rescaled),
  MAE = mae(actual_test_rescaled, predicted_test_rescaled),
  R2 = cor(actual_test_rescaled, predicted_test_rescaled)^2
)

cat("Model Performance Metrics:\n")
cat(sprintf("RMSE: %.2f\nMAE: %.2f\nR-squared: %.2f\n", 
            metrics$RMSE, metrics$MAE, metrics$R2))

# 10. ভিজুয়ালাইজেশন --------------------------------------------------------
results <- data.frame(
  Time = (train_size + timesteps + 1L):length(temperature),
  Actual = actual_test_rescaled,
  Predicted = predicted_test_rescaled
)

# Actual vs Predicted প্লট
ggplot(results, aes(x = Time)) +
  geom_line(aes(y = Actual, color = "Actual"), linewidth = 1) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed", linewidth = 1) +
  geom_ribbon(aes(ymin = Predicted*0.95, ymax = Predicted*1.05), 
              alpha = 0.2, fill = "orange") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  labs(title = "Actual vs Predicted Temperature",
       subtitle = sprintf("RMSE: %.2f | MAE: %.2f", metrics$RMSE, metrics$MAE),
       x = "Time (Months)", y = "Temperature (°C)") +
  theme_minimal()

# 11. ফিউচার ফোরকাস্টিং (36 মাস) -------------------------------------------
last_seq <- tail(test_scaled, timesteps)
future_preds <- numeric(36L)  # 3 বছরের পূর্বাভাস

for (i in 1L:36L) {
  input_array <- array_reshape(last_seq, dim = c(1L, timesteps, 1L))
  next_pred <- predict(model, input_array)
  future_preds[i] <- next_pred
  last_seq <- c(last_seq[-1L], next_pred)
}

forecast_df <- data.frame(
  Month = (length(temperature) + 1L):(length(temperature) + 36L),
  Forecast = inverse_scale(future_preds)
)

# ফোরকাস্ট প্লট
ggplot() +
  geom_line(data = data.frame(Time = 1L:length(temperature), Temp = temperature), 
            aes(x = Time, y = Temp, color = "Historical")) +
  geom_line(data = forecast_df, aes(x = Month, y = Forecast, color = "Forecast"), 
            linetype = "dashed") +
  scale_color_manual(values = c("Historical" = "blue", "Forecast" = "red")) +
  labs(title = "36-Month Temperature Forecast", 
       x = "Time (Months)", y = "Temperature (°C)") +
  theme_minimal()

# ফোরকাস্টেড ভ্যালু প্রিন্ট
print(forecast_df)

##################################XGBOOST v(2) etai final
library(xgboost)
library(tidyverse)
library(Metrics)
library(ggplot2)
library(forecast)

set.seed(199)                          
temperature
# 1. Custom MASE function
mase <- function(actual, predicted, seasonal_period=12) {
  naive_error <- mean(abs(diff(actual, lag=seasonal_period)))
  mean(abs(actual - predicted)) / naive_error
}


# 2. Feature Engineering
create_features <- function(data, timesteps=15) {
  X <- matrix(NA, nrow=length(data)-timesteps, ncol=timesteps+2)
  y <- numeric(length(data)-timesteps)
  
  for(i in (timesteps+1):length(data)) {
    X[i-timesteps, 1:timesteps] <- data[(i-timesteps):(i-1)]
    month <- (i %% 12) + 1
    X[i-timesteps, timesteps+1] <- month
    season <- ceiling(month / 3)
    X[i-timesteps, timesteps+2] <- season
    y[i-timesteps] <- data[i]
  }
  
  list(X=X, y=y)
}

# 3. Data Preparation
set.seed(199)
timesteps <- 15
features <- create_features(temperature, timesteps)

# 4. Train-Test Split
train_size <- floor(0.8 * nrow(features$X))
train_X <- features$X[1:train_size,]
train_y <- features$y[1:train_size]
test_X <- features$X[(train_size+1):nrow(features$X),]
test_y <- features$y[(train_size+1):nrow(features$X)]

# 5. Model Training
dtrain <- xgb.DMatrix(train_X, label=train_y)
dtest <- xgb.DMatrix(test_X, label=test_y)

params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,
  max_depth = 8,
  gamma = 0.1,
  subsample = 0.7,
  colsample_bytree = 0.7
)

model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train=dtrain, test=dtest),
  early_stopping_rounds = 20,
  verbose = 1
)

# 6. Evaluation with All Metrics

preds <- predict(model, dtest)
residuals <- test_y - preds

metrics <- data.frame(
  RMSE = rmse(test_y, preds),
  MAE = mae(test_y, preds),
  MAPE = MAPE(preds, test_y)*100,  # Percentage format
  MASE = mase(test_y, preds),
  R2 = cor(test_y, preds)^2,
  ACF1 = Acf(residuals, plot=FALSE)$acf[2]
)

cat("Comprehensive Model Metrics XGBOOST:\n")
print(metrics)

# 7. Visualization

results <- data.frame(
  Time = (timesteps+train_size+1):length(temperature),
  Actual = test_y,
  Predicted = preds
)

ggplot(results, aes(x=Time)) +
  geom_line(aes(y=Actual, color="Actual")) +
  geom_line(aes(y=Predicted, color="Predicted"), linetype="dashed") +
  scale_color_manual(values=c("Actual"="blue", "Predicted"="red")) +
  labs(title="প্রকৃত vs পূর্বাভাসিত তাপমাত্রা", x="সময় (মাস)", y="তাপমাত্রা (°C)") +
  theme_minimal()

par(mfrow=c(1,3))
plot(residuals, main="Residuals over Time", type='l')
hist(residuals, main="Residual Distribution")
qqnorm(residuals); qqline(residuals)
par(mfrow=c(1,1))

# 8. Future Forecast with Confidence Intervals
last_data <- tail(temperature, timesteps)
future_preds <- numeric(36)
future_lower <- numeric(36)
future_upper <- numeric(36)

for(i in 1:36) {
  month <- (length(temperature) + i) %% 12 + 1
  season <- ceiling(month / 3)
  input <- matrix(c(last_data, month, season), nrow=1)
  pred <- predict(model, input)
  # Estimate prediction interval
  future_preds[i] <- pred
  future_lower[i] <- pred - 1.96 * sd(residuals)
  future_upper[i] <- pred + 1.96 * sd(residuals)
  
  last_data <- c(last_data[-1], pred)
}

forecast_df <- data.frame(
  Month = 1:36,
  Temperature = future_preds,
  Lower = future_lower,
  Upper = future_upper
)
print(forecast_df)

# Combined plot

ggplot(forecast_df, aes(x=Month, y=Temperature)) +
  geom_line(color="darkgreen") +
  labs(title="৩৬ মাসের তাপমাত্রা পূর্বাভাস", x="মাস", y="তাপমাত্রা (°C)") +
  theme_minimal()

# ধরো তোমার temperature নামে historical ডেটা আছে
# future_preds হলো ৩৬ মাসের পূর্বাভাস

total_data <- data.frame(
  Time = 1:(length(temperature) + 36),
  Temperature = c(temperature, future_preds),
  Lower = c(rep(NA, length(temperature)), future_lower),
  Upper = c(rep(NA, length(temperature)), future_upper),
  Type = c(rep("Historical", length(temperature)), rep("Forecast", 36))
)
ggplot(total_data, aes(x = Time, y = Temperature, color = Type)) +
  geom_line(data = subset(total_data, Type == "Historical"), size = 0.5) +
  geom_line(data = subset(total_data, Type == "Forecast"), linetype = "dashed", size = 0.5) +
  scale_color_manual(values = c("Historical" = "blue", "Forecast" = "red")) +
  labs(
    title = "36-Month Temperature Forecast",
    x = "Time (Months)",
    y = "Temperature"
  ) +
  theme_minimal()



