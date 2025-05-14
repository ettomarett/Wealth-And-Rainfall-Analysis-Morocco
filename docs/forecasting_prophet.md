# Prophet Rainfall Forecasting Documentation

## What is Prophet?
Prophet is an open-source time series forecasting tool developed by Facebook. It is designed to handle time series data with strong seasonal effects and missing data, and is robust to outliers and trend changes.

## How Prophet Was Applied
- **Data Selection:**
  - The user selects a region and a rainfall metric (e.g., monthly mean rainfall).
  - The time series for the selected region and metric is extracted from the processed rainfall dataset.
- **Preprocessing:**
  - The time series is formatted for Prophet (columns: 'ds' for date, 'y' for value) and missing values are dropped.
- **Model Fitting:**
  - A Prophet model is fitted to the historical rainfall data.
- **Forecasting:**
  - The model forecasts rainfall for the next 12 periods (e.g., months).
- **Visualization:**
  - Both historical and forecasted values are plotted for comparison.
  - The forecast includes uncertainty intervals (yhat_lower, yhat_upper).
  - The forecast table for the next 12 periods is displayed.

## Assumptions & Limitations
- Prophet assumes the time series has clear trends and/or seasonality.
- The default settings may not capture all patterns; further tuning (e.g., adding holidays, changing seasonality) may improve results.
- Forecast accuracy depends on the quality and length of the historical data. 