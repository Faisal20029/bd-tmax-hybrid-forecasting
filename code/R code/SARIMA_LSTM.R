# SARIMA + LSTM Hybrid Model on Temperature temperature_ts
################ লাইব্রেরি লোড করুন
set.seed(156)
library(forecast)
library(keras)
library(ggplot2)
library(gridExtra)
library(knitr)
# ১. ডেটা লোড ও প্রিপ্রসেসিং (উদাহরণ হিসেবে AirPassengers ডেটা ব্যবহার করা হলো)

temperature_ts

# ট্রেন-টেস্ট স্প্লিট (80-20)
train_size <- floor(0.8 * length(temperature_ts))
train <- window(temperature_ts, end = time(temperature_ts)[train_size])
test <- window(temperature_ts, start = time(temperature_ts)[train_size + 1])

# ২. SARIMA মডেল ফিট করুন
sarima_model99 <- auto.arima(train_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
sarima_residuals <- residuals(sarima_model99)

# ৩. LSTM-এর জন্য ডেটা প্রস্তুত করুন
create_lstm_data <- function(residuals, timesteps = 12) {
  X <- matrix(NA, nrow = length(residuals) - timesteps, ncol = timesteps)
  y <- numeric(length(residuals) - timesteps)
  
  for (i in (timesteps + 1):length(residuals)) {
    X[i - timesteps, ] <- residuals[(i - timesteps):(i - 1)]
    y[i - timesteps] <- residuals[i]
  }
  
  # 3D টেনসরে রূপান্তর (samples, timesteps, features)
  X_array <- array(X, dim = c(nrow(X), timesteps, 1))
  list(X = X_array, y = y)
}

lstm_data <- create_lstm_data(sarima_residuals)

# ৪. নরমালাইজেশন (LSTM-এর জন্য গুরুত্বপূর্ণ)
train_min <- min(lstm_data$X)
train_max <- max(lstm_data$X)

normalize <- function(x, min_val = train_min, max_val = train_max) {
  (x - min_val) / (max_val - min_val + 1e-10)
}

denormalize <- function(x, min_val = train_min, max_val = train_max) {
  x * (max_val - min_val + 1e-10) + min_val
}

# নরমালাইজড ডেটা
X_train <- normalize(lstm_data$X)
y_train <- normalize(lstm_data$y)

# ৫. LSTM মডেল বিল্ড করুন
model <- keras_model_sequential() %>%
  layer_lstm(units = 50, return_sequences = TRUE, input_shape = c(12, 1)) %>%
  layer_dropout(0.2) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = "mse"
)

# ৬. মডেল ট্রেন করুন
history <- model %>% fit(
  X_train, y_train,
  epochs = 100,
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(
    callback_early_stopping(patience = 10),
    callback_reduce_lr_on_plateau(factor = 0.1, patience = 5)
  )
)

plot(history)  # লস কার্ভ দেখুন

# ৭. হাইব্রিড প্রেডিকশন ফাংশন
hybrid_predict_lstm <- function(model, sarima_model, newdata = NULL, h = NULL, timesteps = 12) {
  if (is.null(newdata) && !is.null(h)) {
    # ফিউচার ফোরকাস্ট
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(residuals(sarima_model), timesteps)
    lstm_fc <- numeric(h)
    
    for (i in 1:h) {
      input <- array(last_residuals, dim = c(1, timesteps, 1))
      input_norm <- normalize(input, train_min, train_max)
      pred <- predict(model, input_norm)[1]
      pred_denorm <- denormalize(pred, train_min, train_max)
      lstm_fc[i] <- pred_denorm
      last_residuals <- c(last_residuals[-1], pred_denorm)
    }
    
    return(list(
      sarima = sarima_fc$mean,
      lstm_residuals = lstm_fc,
      hybrid = sarima_fc$mean + lstm_fc
    ))
  }
  else if (!is.null(newdata)) {
    # টেস্ট ডেটা ইভ্যালুয়েশন
    h <- length(newdata)
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(residuals(sarima_model), timesteps)
    lstm_fc <- numeric(h)
    
    for (i in 1:h) {
      input <- array(last_residuals, dim = c(1, timesteps, 1))
      input_norm <- normalize(input, train_min, train_max)
      pred <- predict(model, input_norm)[1]
      pred_denorm <- denormalize(pred, train_min, train_max)
      lstm_fc[i] <- pred_denorm
      last_residuals <- c(last_residuals[-1], pred_denorm)
    }
    
    return(list(
      sarima = sarima_fc$mean,
      lstm_residuals = lstm_fc,
      hybrid = sarima_fc$mean + lstm_fc
    ))
  }
}

# ৮. টেস্ট ডেটায় প্রেডিকশন
test_predictions <- hybrid_predict_lstm(model, sarima_model99, newdata = test)

# ৯. মেট্রিক্স ক্যালকুলেশন
calculate_metrics <- function(actual, predicted) {
  list(
    RMSE = sqrt(mean((actual - predicted)^2)),
    MAE = mean(abs(actual - predicted)),
    MAPE = mean(abs((actual - predicted)/actual)) * 100,
    MASE = mean(abs((actual - predicted) / mean(abs(actual - mean(actual))))),
    R2 = cor(actual, predicted)^2,
    ACF1 = Acf(actual - predicted, plot = FALSE)$acf[2]  # ACF at lag 1
  )
}
metrics <- calculate_metrics(test, test_predictions$hybrid)
print(paste("Hybrid Model Metrics:"))
print(metrics)
metrics_vector0 <- c(
  RMSE  = round(metrics$RMSE, 4),
  MAE   = round(metrics$MAE, 4),
  MAPE  = round(metrics$MAPE, 4),
  MASE  = round(metrics$MASE, 4),
  R2    = round(metrics$R2, 4),
  ACF1  = round(metrics$ACF1, 4)
)

print(metrics_vector0)

# ১০. ভিজুয়ালাইজেশন
# আসল বনাম প্রেডিক্টেড
actual_vs_pred <- data.frame(
  Date = as.numeric(time(test)),
  Actual = as.numeric(test),
  Predicted = as.numeric(test_predictions$hybrid)
)

p1 <- ggplot(actual_vs_pred, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual")) +
  geom_line(aes(y = Predicted, color = "Predicted"), linetype = "dashed") +
  labs(title = "Actual vs Predicted (Test Data)", y = "Value") +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red")) +
  theme_minimal()

# রেসিডুয়াল প্লট
residual_data <- data.frame(
  Date = as.numeric(time(test)),
  Residual = as.numeric(test) - as.numeric(test_predictions$hybrid)
)

p2 <- ggplot(residual_data, aes(x = Date, y = Residual)) +
  geom_point(color = "purple") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Residuals Plot") +
  theme_minimal()

grid.arrange(p1, p2, ncol = 1)

# ১১. ফিউচার ফোরকাস্ট (36 মাস)
final_sarima <- Arima(temperature_ts, model = sarima_model)
future_forecast <- hybrid_predict_lstm(model, final_sarima, h = 36) 
future_forecast

# ফোরকাস্ট প্লট
autoplot(temperature_ts) +
  autolayer(ts(future_forecast$hybrid, start = end(temperature_ts)[1] + 1/12, frequency = 12),
            series = "36-Month Forecast", color = "red") +
  labs(title = "SARIMA-LSTM Hybrid Forecast", y = "Value")

