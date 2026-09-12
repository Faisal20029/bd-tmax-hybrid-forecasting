rm(list = ls())
library(tseries)
library(forecast)
library(FinTS)
library(forecast)
library(tseries)
library(urca)
library(ggplot2)
library(readxl)

data<-read_excel("C:/Users/hp/Desktop/Research paper/data/newdata.xlsx", 
                 sheet = "Mymensingh")
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
str(data)
#===============================
# 1. Time Series Decomposition
#===============================
decomp <- decompose(temperature_ts)
plot(decomp)
#===============================
# 2. ACF & PACF Plot Together
#===============================
par(mfrow = c(2, 1), mar = c(4, 4, 3, 1))  # এক লাইনে দুইটা plot
acf(temperature_ts, main = "ACF plot of the Observed Temperature Data")
pacf(temperature_ts, main = "PACF plot of the Observed Temperature Data")
par(mfrow = c(1,1))   # reset layout

#===============================
# 3. ADF Test
#===============================
adf_result <- adf.test(temperature_ts, alternative = "stationary")
print(adf_result)

#===============================
# 4. PP Test
#===============================
pp_result <- pp.test(temperature_ts)
print(pp_result)

View(data)
str(data)
summary(data)
sum(is.na(data$Temperature))
temperature_ts <- ts(data$Temperature, start = c(1972, 1), frequency = 12)
head(temperature_ts)
tail(temperature_ts)

acf(temperature_ts, main = "ACF - Autocorrelation Function")
pacf(temperature_ts, main = "PACF - Partial Autocorrelation Function")

adf_result <- adf.test(temperature_ts, alternative = "stationary")
adf_result

kpss_result <- ur.kpss(temperature_ts)
summary(kpss_result)

# Apply PP test on full series
pp_result <- pp.test(temperature_ts)
print(pp_result)


# First differencing
diff_temp <- diff(temperature_ts)

# Plot original vs differenced
par(mfrow = c(2,1))
plot(temperature_ts, main = "Original Temperature Time Series", col = "blue")
plot(diff_temp, main = "First Differenced Series", col = "darkgreen")

# ADF Test on differenced series
adf_diff <- adf.test(diff_temp, alternative = "stationary")
print(adf_diff)

# PP Test on differenced series
pp_diff <- pp.test(diff_temp)
print(pp_diff)

# KPSS Test on differenced series
kpss_diff <- ur.kpss(diff_temp)
summary(kpss_diff)
autoplot(temperature_ts)


# Detrending
time <- 1:length(temperature_ts)
trend_model <- lm(temperature_ts ~ time)
detrended_series <- ts(residuals(trend_model), frequency = 12, start = c(1972, 1))

# Plot the detrended series
library(ggplot2)
autoplot(detrended_series) + ggtitle("Detrended Temperature Series")

# ADF test on detrended series
library(tseries)
adf.test(detrended_series)
library(urca)

# KPSS test on detrended series
kpss_detrended <- ur.kpss(detrended_series)
summary(kpss_detrended)

stl_decomp <- stl(temperature_ts, s.window = "periodic")
plot(stl_decomp)

decomp <- decompose(temperature_ts)
plot(decomp)

diff_seasonal <- diff(temperature_ts, lag = 12)
adf.test(diff_seasonal)




plot(temperature_ts, main = "Monthly temperature Time Series", xlab = "Time", ylab = "Total temperature (mm)")
ggseasonplot(temperature_ts) #its help to eki mashe proti year kemon pattern chilo
ggsubseriesplot(temperature_ts)# monthly detail,প্রতি মাসে প্রতিটি বছরের temperature কেমন ছিলো সেটা বিশ্লেষণ করা
ggtsdisplay(temperature_ts)
nsdiffs(temperature_ts)
ndiffs(temperature_ts)
#auto model
fit_sarima <- auto.arima(temperature_ts,
                         seasonal = TRUE,
                         stepwise = FALSE,     
                         approximation = FALSE 
)
summary(fit_sarima)

checkresiduals(fit_sarima)

# Correct SARIMA model using forecast::Arima (NOT stats::arima)
manual_model4 <- Arima(temperature_ts,order = c(1, 1, 1),seasonal = list(order = c(1, 1, 1), period = 12))
summary(manual_model4)#AIC=1685.06

manual_model5 <- Arima(temperature_ts,order = c(1,1,1),
                       seasonal = list(order = c(1,1,1), period = 12))
summary(manual_model5)

manual_model <- Arima(temperature_ts,order = c(1, 0, 1),seasonal = list(order = c(1, 1, 1), period = 12))
summary(manual_model) #AIC=1686.76

#SARIMA(2,0,1)(2,1,1)[12]
manual_model2 <- Arima(temperature_ts,order = c(2, 0, 1),seasonal = list(order = c(2, 1, 1), period = 12))
summary(manual_model2) #AIC=1680.46
checkresiduals(manual_model2) 

manual_model3 <- Arima(temperature_ts,order = c(2, 0, 1),seasonal = list(order = c(2, 1, 2), period = 12))
summary(manual_model3)#AIC=1682.48


residuals_sarima<- residuals(fit_sarima)
qqnorm(residuals_sarima, main = "QQ Plot of Residuals (White Noise Errors)")
qqline(residuals_sarima, col = "red")  


fitted_vals <- fitted(fit_sarima)
ts.plot(temperature_ts, fitted_vals, col = c("black", "blue"), lty = c(1, 2),lwd=1.5,
        main = "Actual vs Fitted temperature",
        ylab = "temperature (mm)",
        xlab = "Year")
legend("topright", legend = c("Actual", "Fitted"),
       col = c("black", "blue"), lty = c(1, 2),cex = 0.5)


# Boxplot for outlier detection
boxplot(temperature_ts, main = "Boxplot for Outlier Detection", ylab = "temperature")
# Decompose the time series using STL decomposition
decomposed_data <- stl(temperature_ts, s.window = "periodic")
seasonal_part<-decomposed_data$time.series[,"seasonal"]
seasonal_part
residual_part<-decomposed_data$time.series[,"remainder"]
residual_part

# Plot the decomposition
plot(decomposed_data)

# Forecasting future values
forecast_values <- forecast(fit_sarima, h = 36) # h = 24 মানে 24 মাসের (2 বছর) ভবিষ্যত পূর্বাভাস
print(forecast_values$mean)

# Plot the forecast
plot(forecast_values, main = "Forecasted Temperature (Future 2 Years)", xlab = "Time", ylab = "Temperature (°C)", col = "blue")

# Show the forecasted values
forecast_values$mean # এটি আগামীর ভবিষ্যত মানগুলির তালিকা দেখাবে


#########################

