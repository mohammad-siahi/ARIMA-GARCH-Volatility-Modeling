###############################################################
# ARIMA & GARCH Modeling on Monthly Stock Returns (ENEL.MI)
# Description:
# This script downloads ENEL stock data from Yahoo Finance,
# computes monthly log returns, estimates an ARIMA model
# for the conditional mean, tests for ARCH effects,
# and fits a GARCH(1,1) model for conditional volatility.
###############################################################

############################
# 1 - LOAD REQUIRED PACKAGES
############################

library(quantmod)    # Download financial data
library(forecast)    # ARIMA modeling
library(tseries)     # ADF test
library(FinTS)       # ARCH LM test
library(rugarch)     # GARCH modeling
library(xts)         # Time series handling


############################
# 2 - DOWNLOAD DATA
############################

ticker <- "ENEL.MI"

# Get data from Yahoo Finance
data <- getSymbols(ticker, src = "yahoo", auto.assign = FALSE)

# Use adjusted closing prices
price <- Ad(data)


############################
# 3 - COMPUTE MONTHLY LOG RETURNS
############################

# Convert daily prices to monthly prices (last observation of month)
monthly_price <- to.monthly(price, indexAt = "lastof", drop.time = TRUE)
monthly_close <- monthly_price[,4]

# Compute log returns (in percentage terms)
returns <- diff(log(monthly_close)) * 100
returns <- na.omit(returns)


############################
# 4 - PLOT RETURNS
############################

plot(returns,
     main = "Monthly Log Returns - ENEL",
     ylab = "Return (%)")
abline(h = 0, lty = 2)


############################
# 5 - DESCRIPTIVE STATISTICS
############################

cat("Mean:", mean(returns), "\n")
cat("Std Dev:", sd(returns), "\n")
cat("Variance:", var(returns), "\n")


############################
# 6 - T-TEST (Mean = 0)
############################
# H0: Mean return = 0
# H1: Mean return ≠ 0

t_test_result <- t.test(as.numeric(returns))
print(t_test_result)


############################
# 7 - STATIONARITY TEST (ADF)
############################
# H0: Non-stationary
# H1: Stationary

adf_result <- adf.test(as.numeric(returns))
print(adf_result)


############################
# 8 - ARIMA MODEL (CONDITIONAL MEAN)
############################

# Convert to ts object (monthly frequency)
y_ts <- ts(as.numeric(returns), frequency = 12)

# Automatic ARIMA selection
fit_arima <- auto.arima(y_ts, seasonal = FALSE)

summary(fit_arima)


############################
# 9 - ARIMA RESIDUAL DIAGNOSTICS
############################
# H0: No autocorrelation

checkresiduals(fit_arima)


############################
# 10 - ARCH LM TEST (Before GARCH)
############################
# H0: No ARCH effect

res_arima <- residuals(fit_arima)
arch_test_before <- ArchTest(res_arima, lags = 12)
print(arch_test_before)


###############################################################
# GARCH MODELING
###############################################################

############################
# 11 - SPECIFY GARCH(1,1)
############################

spec <- ugarchspec(
  variance.model = list(model = "sGARCH",
                        garchOrder = c(1,1)),
  mean.model     = list(armaOrder = c(0,0),
                        include.mean = TRUE),
  distribution.model = "norm"
)


############################
# 12 - FIT GARCH MODEL
############################

fit_garch <- ugarchfit(spec = spec, data = returns)
show(fit_garch)


############################
# 13 - VOLATILITY PERSISTENCE
############################
# Persistence = alpha + beta
# Measures how long volatility shocks last

params <- coef(fit_garch)
persistence <- unname(params["alpha1"] + params["beta1"])

cat("Volatility Persistence:", persistence, "\n")


############################
# 14 - DIAGNOSTICS AFTER GARCH
############################

# Standardized residuals
z <- residuals(fit_garch, standardize = TRUE)

# Ljung-Box test on residuals
lb_res <- Box.test(z, lag = 24, type = "Ljung-Box")
print(lb_res)

# Ljung-Box test on squared residuals
lb_sq_res <- Box.test(z^2, lag = 24, type = "Ljung-Box")
print(lb_sq_res)

# ARCH LM test after GARCH
arch_test_after <- ArchTest(z, lags = 12)
print(arch_test_after)


############################
# 15 - PLOT CONDITIONAL VOLATILITY
############################

sigma_t <- sigma(fit_garch)

plot(sigma_t,
     type = "l",
     main = "Estimated Conditional Volatility - GARCH(1,1)",
     ylab = "Sigma")


############################
# 16 - FORECAST 12 MONTHS AHEAD
############################

garch_forecast <- ugarchforecast(fit_garch, n.ahead = 12)

sigma_fc <- as.numeric(sigma(garch_forecast))
mu_fc    <- as.numeric(fitted(garch_forecast))

# Generate correct future dates
last_date <- as.Date(last(index(returns)))
future_dates <- seq(last_date, by = "month", length.out = 13)[-1]

sigma_fc_xts <- xts(sigma_fc, order.by = future_dates)
mu_fc_xts    <- xts(mu_fc, order.by = future_dates)


############################
# 17 - PLOT FORECASTED VOLATILITY
############################

plot(sigma_fc_xts,
     type = "l",
     main = "Forecasted Volatility (Next 12 Months)",
     ylab = "Sigma")


###############################################################
# FINAL INTERPRETATION
#
# - ARIMA models the conditional mean of returns.
# - ARCH LM test checks for volatility clustering.
# - GARCH captures time-varying volatility.
# - Persistence parameter (alpha + beta) indicates
#   how long volatility shocks remain in the system.
###############################################################