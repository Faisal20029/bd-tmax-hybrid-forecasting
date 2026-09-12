#only XGBOOST
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
  RMSE = round(rmse(test_y, preds),4),
  MAE = round(mae(test_y, preds),4),
  MAPE = round(MAPE(preds, test_y)*100,4),  # Percentage format
  MASE = round(mase(test_y, preds),4),
  R2 = round(cor(test_y, preds)^2,4),
  ACF1 = round(Acf(residuals, plot=FALSE)$acf[2],4)
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





