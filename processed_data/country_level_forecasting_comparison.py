import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Import models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False

# File paths
train_path = Path(__file__).parent / 'country_train.csv'
test_path = Path(__file__).parent / 'country_test.csv'

# Load data
train = pd.read_csv(train_path, parse_dates=['date'])
test = pd.read_csv(test_path, parse_dates=['date'])

# Prepare series
y_train = train['rfh_avg'].values
future_dates = test['date']

results = {}

# ARIMA
try:
    arima_model = ARIMA(y_train, order=(1,1,1)).fit()
    arima_forecast = arima_model.forecast(steps=len(test))
    arima_mae = mean_absolute_error(test['rfh_avg'], arima_forecast)
    results['ARIMA'] = arima_mae
except Exception as e:
    print(f"ARIMA error: {e}")
    results['ARIMA'] = None

# SARIMA
try:
    sarima_model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
    sarima_forecast = sarima_model.forecast(steps=len(test))
    sarima_mae = mean_absolute_error(test['rfh_avg'], sarima_forecast)
    results['SARIMA'] = sarima_mae
except Exception as e:
    print(f"SARIMA error: {e}")
    results['SARIMA'] = None

# Holt-Winters
try:
    hw_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
    hw_forecast = hw_model.forecast(steps=len(test))
    hw_mae = mean_absolute_error(test['rfh_avg'], hw_forecast)
    results['Holt-Winters'] = hw_mae
except Exception as e:
    print(f"Holt-Winters error: {e}")
    results['Holt-Winters'] = None

# Prophet
if prophet_available:
    try:
        prophet_train = train.rename(columns={'date': 'ds', 'rfh_avg': 'y'})[['ds', 'y']]
        m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
        m.fit(prophet_train)
        future = pd.DataFrame({'ds': test['date']})
        forecast = m.predict(future)
        prophet_forecast = forecast['yhat'].values
        prophet_mae = mean_absolute_error(test['rfh_avg'], prophet_forecast)
        results['Prophet'] = prophet_mae
    except Exception as e:
        print(f"Prophet error: {e}")
        results['Prophet'] = None
else:
    print("Prophet not installed.")
    results['Prophet'] = None

# Print and save results
print("\nModel MAE (lower is better):")
for model, mae in results.items():
    print(f"{model}: {mae}")

# Save results to file
results_path = Path(__file__).parent / 'country_model_comparison_results.csv'
pd.DataFrame(list(results.items()), columns=['Model', 'MAE']).to_csv(results_path, index=False)
print(f"Results saved to {results_path}")

# Plot forecasts
plt.figure(figsize=(10,6))
plt.plot(train['date'], y_train, label='Train')
plt.plot(test['date'], test['rfh_avg'], label='Test', color='black')
if results['ARIMA'] is not None:
    plt.plot(test['date'], arima_forecast, label='ARIMA Forecast')
if results['SARIMA'] is not None:
    plt.plot(test['date'], sarima_forecast, label='SARIMA Forecast')
if results['Holt-Winters'] is not None:
    plt.plot(test['date'], hw_forecast, label='Holt-Winters Forecast')
if results['Prophet'] is not None:
    plt.plot(test['date'], prophet_forecast, label='Prophet Forecast')
plt.legend()
plt.title('Country-level Rainfall Forecasts')
plt.xlabel('Date')
plt.ylabel('Mean Rainfall (rfh_avg)')
plt.tight_layout()
plt.savefig(Path(__file__).parent / 'country_forecast_comparison.png')
plt.show() 