# ARIMA Rainfall Forecasting Documentation

## What is ARIMA?
ARIMA (AutoRegressive Integrated Moving Average) is a popular statistical model for time series forecasting. It models a time series based on its own past values, differences, and lagged forecast errors.

## How ARIMA Was Applied
- **Data Selection:**
  - The user selects a region and a rainfall metric (e.g., monthly mean rainfall).
  - The time series for the selected region and metric is extracted from the processed rainfall dataset.
- **Preprocessing:**
  - The time series is sorted by date and missing values are dropped.
- **Model Fitting:**
  - An ARIMA model (order (1,1,1) by default) is fitted to the historical rainfall data using the `statsmodels` library.
- **Forecasting:**
  - The model forecasts rainfall for the next 12 periods (e.g., months).
- **Visualization:**
  - Both historical and forecasted values are plotted for comparison.
  - Model summary statistics are displayed for interpretation.

## Assumptions & Limitations
- ARIMA assumes the time series is stationary or can be made stationary by differencing.
- The default order (1,1,1) may not be optimal for all regions/metrics; further tuning may improve results.
- ARIMA does not handle seasonality unless extended (e.g., SARIMA).
- Forecast accuracy depends on the quality and length of the historical data. 