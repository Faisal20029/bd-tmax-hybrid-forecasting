#SARIMA+ANN(DNN)
set.seed(123) 
library(forecast)
library(keras)
library(ggplot2)
library(Metrics)

# ১. ডেটা প্রস্তুতকরণ
temperature_ts # data
train_size <- floor(0.8 * length(temperature_ts))
train <- window(temperature_ts, end = time(temperature_ts)[train_size])
test <- window(temperature_ts, start = time(temperature_ts)[train_size + 1])

# ২. SARIMA মডেল
sarima_model99
sarima_residuals <- residuals(sarima_model99)

# ৩. DNN ডেটা প্রস্তুতকরণ (সমস্যা সমাধান)
create_dnn_temperature_ts <- function(residuals, timesteps = 12) {
  X <- matrix(NA, nrow = length(residuals) - timesteps, ncol = timesteps)
  y <- numeric(length(residuals) - timesteps)
  
  for(i in (timesteps + 1):length(residuals)) {
    X[i - timesteps, ] <- residuals[(i - timesteps):(i - 1)]
    y[i - timesteps] <- residuals[i]
  }
  
  # ডাইমেনশন ঠিক করতে array রূপান্তর
  list(X = array(X, dim = c(nrow(X), ncol(X), 1)), 
       y = y)
}

dnn_temperature_ts <- create_dnn_temperature_ts(sarima_residuals)

# ৪. নরমালাইজেশন
normalize <- function(x) { (x - min(x)) / (max(x) - min(x) + 1e-10) }

# ৫. DNN মডেল (ANN)
model <- keras_model_sequential() %>%
  layer_dense(units = 128, activation = 'relu', input_shape = c(12)) %>%  # ANN (DNN) প্রথম লেয়ার
  layer_dropout(0.2) %>%  # ড্রপআউট লেয়ার
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(1)  # আউটপুট লেয়ার

model %>% compile(
  optimizer = optimizer_adam(learning_rate = 0.001),
  loss = 'mse'
)

# ৬. ট্রেনিং ডেটা প্রস্তুতকরণ
X_train <- apply(dnn_temperature_ts$X, 1:2, normalize)  # সঠিক নরমালাইজেশন
X_train <- array(X_train, dim = c(dim(X_train), 1))  # 3D টেনসরে রূপান্তর

history <- model %>% fit(
  X_train, dnn_temperature_ts$y,
  epochs = 200,  # সাধারণত 100-500 রেঞ্জ ভালো কাজ করে
  batch_size = 32,
  validation_split = 0.2,
  verbose = 1,
  callbacks = list(
    callback_early_stopping(patience = 10),  # 10 epochs ধরে উন্নতি না হলে ট্রেনিং স্টপ করবে
    callback_reduce_lr_on_plateau(factor = 0.1, patience = 5)  # লার্নিং রেট অটো-অ্যাডজাস্ট করবে
  )
)
plot(history)

# ৭. হাইব্রিড প্রেডিকশন ফাংশন (সমস্যা সমাধান)
hybrid_predict <- function(model, sarima_model, newtemperature_ts = NULL, h = NULL, timesteps = 12) {
  
  if (is.null(newtemperature_ts) && !is.null(h)) {
    # ভবিষ্যত ফোরকাস্ট
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(residuals(sarima_model), timesteps)
    dnn_fc <- numeric(h)
    
    for(i in 1:h) {
      input <- matrix(last_residuals, nrow = 1)
      input_norm <- apply(input, 2, normalize)
      input_array <- array(input_norm, dim = c(1, timesteps, 1))  # 3D ইনপুট
      pred <- predict(model, input_array)[1]
      dnn_fc[i] <- pred
      last_residuals <- c(last_residuals[-1], pred)
    }
    
    return(list(
      sarima = sarima_fc$mean,
      dnn_residuals = dnn_fc,
      hybrid = sarima_fc$mean + dnn_fc
    ))
  }
  else if (!is.null(newtemperature_ts)) {
    # টেস্ট ডেটা ইভ্যালুয়েশন
    h <- length(newtemperature_ts)
    sarima_fc <- forecast(sarima_model, h = h)
    last_residuals <- tail(residuals(sarima_model), timesteps)
    dnn_fc <- numeric(h)
    
    for(i in 1:h) {
      input <- matrix(last_residuals, nrow = 1)
      input_norm <- apply(input, 2, normalize)
      input_array <- array(input_norm, dim = c(1, timesteps, 1))  # 3D ইনপুট
      pred <- predict(model, input_array)[1]
      dnn_fc[i] <- pred
      last_residuals <- c(last_residuals[-1], pred)
    }
    
    return(list(
      sarima = sarima_fc$mean,
      dnn_residuals = dnn_fc,
      hybrid = sarima_fc$mean + dnn_fc
    ))
  }
  else {
    stop("Either 'newtemperature_ts' or 'h' must be provided")
  }
}

# ৮. টেস্টিং ও ভিজুয়ালাইজেশন
test_predictions <- hybrid_predict(model, sarima_model99, newtemperature_ts = test)
metrics <- calculate_metrics(test, test_predictions$hybrid)
print(metrics)
metrics_vector1 <- c(
  RMSE  = round(metrics$RMSE, 4),
  MAE   = round(metrics$MAE, 4),
  MAPE  = round(metrics$MAPE, 4),
  MASE  = round(metrics$MASE, 4),
  R2    = round(metrics$R2, 4),
  ACF1  = round(metrics$ACF1, 4)
)


print(metrics_vector1)
# ৯. ভবিষ্যত ফোরকাস্ট
final_sarima <- Arima(temperature_ts, model = sarima_model)
future_forecast <- hybrid_predict(model, final_sarima, h = 36)
print(future_forecast)

# ১০. প্লটিং
autoplot(temperature_ts) +
  autolayer(ts(future_forecast$hybrid, start = end(temperature_ts)[1] + 1/12, frequency = 12), 
            series = "36-Month Forecast", color = "red") +
  labs(title = "SARIMA-ANN Hybrid Forecast", y = "Temperature")
library(gridExtra)

# আসল বনাম প্রেডিক্টেড ডেটা ফ্রেম তৈরি
actual_vs_pred <- data.frame(
  Date = as.numeric(time(test)),  # সময়কে সংখ্যায় রূপান্তর
  Actual = as.numeric(test),
  Predicted = as.numeric(test_predictions$hybrid)
)

# মূল প্লট
p1 <- ggplot(actual_vs_pred, aes(x = Date)) +
  geom_line(aes(y = Actual, color = "Actual value")) +
  geom_line(aes(y = Predicted, color = "Prediction"), linetype = "dashed") +
  labs(title = "Actual vs Predicted Values", 
       y = "Temperature", 
       x = "Time") +
  scale_color_manual(name = "Legend",
                     values = c("Actual value" = "blue", "Prediction" = "red")) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "bottom")

# রেসিডুয়াল প্লট
residual_data <- data.frame(
  Date = as.numeric(time(test)),
  Residual = as.numeric(test) - as.numeric(test_predictions$hybrid)
)

p2 <- ggplot(residual_data, aes(x = Date, y = Residual)) +
  geom_point(color = "purple", alpha = 0.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  geom_smooth(method = "loess", se = FALSE, color = "darkgreen") +  # ট্রেন্ড লাইন
  labs(title = "Residual Analysis", 
       y = "Residuals", 
       x = "Time") +
  theme_minimal(base_size = 12)

# পাশাপাশি প্লট
grid.arrange(p1, p2, ncol = 1, heights = c(2, 1.5))

#################

