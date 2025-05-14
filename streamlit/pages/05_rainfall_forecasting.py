import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
try:
    from prophet import Prophet
    prophet_available = True
except ImportError:
    prophet_available = False
    st.warning("Prophet is not installed. Some forecasting features will be disabled.")

st.set_page_config(page_title="Rainfall Forecasting", layout="wide")
st.title("Rainfall Forecasting for Morocco")
st.markdown("""
This page provides time series forecasting of national rainfall using four algorithms: **ARIMA**, **Prophet**, **SARIMA**, and **Holt-Winters**. 
Compare the forecasts and accuracy metrics to understand future rainfall patterns.
""")

# Load processed rainfall data
processed_dir = Path(__file__).parents[2] / "processed_data"
data_path = processed_dir / "cleaned_rainfall_data.csv"
if not data_path.exists():
    st.error("Processed rainfall data not found. Please run the data processing script first.")
    st.stop()

# Load and prepare national-level data
rainfall_df = pd.read_csv(data_path)
date_col = 'date'

# Calculate national average
national_data = rainfall_df.groupby(date_col)['rfh_avg'].mean().reset_index()
national_data = national_data.sort_values(date_col)

# Global test period selection
st.sidebar.markdown("---")
st.sidebar.subheader("Forecast Settings")
max_test_periods = min(60, len(national_data) // 4)  # Maximum 60 or 1/4 of data
n_test_global = st.sidebar.number_input(
    "Number of periods for test set (e.g., months)", 
    min_value=6, 
    max_value=max_test_periods, 
    value=12,
    help="This will be used across all tabs for consistency"
)

def split_train_test(data, n_test):
    """Helper function to split data into train and test sets"""
    if len(data) <= n_test:
        st.error(f"Not enough data to split: only {len(data)} periods.")
        return data, pd.DataFrame()
    return data.iloc[:-n_test], data.iloc[-n_test:]

# Update the create_forecast_plot function to handle different test periods
def create_forecast_plot(historical_data, test_data, forecast_data, title, include_components=False):
    """
    Helper function to create consistent forecast plots across tabs
    """
    fig, ax = plt.subplots(figsize=(12,6))
    
    # Calculate date range for x-axis (36 months)
    last_three_years_start = pd.to_datetime(historical_data['date'].iloc[-1]) - pd.DateOffset(months=36)
    forecast_end = pd.to_datetime(test_data['date'].iloc[-1]) if len(test_data) > 0 else pd.to_datetime(forecast_data.index[-1])
    
    # Get last three years of data
    last_years_mask = pd.to_datetime(historical_data['date']) >= last_three_years_start
    last_years_data = historical_data[last_years_mask]
    
    # Plot data
    ax.plot(pd.to_datetime(last_years_data['date']), last_years_data['rfh_avg'], 
            label='Historical (Last 3 Years)', color='blue')
    
    if len(test_data) > 0:
        ax.plot(pd.to_datetime(test_data['date']), test_data['rfh_avg'], 
                label=f'Test ({len(test_data)} periods)', color='black')
    
    # Plot forecast
    if isinstance(forecast_data, dict):
        for name, forecast in forecast_data.items():
            ax.plot(pd.to_datetime(test_data['date']), forecast, label=f'{name} Forecast')
    else:
        ax.plot(pd.to_datetime(test_data['date']), forecast_data, label='Forecast', color='red')
    
    # Set x-axis limits and format
    ax.set_xlim([last_three_years_start, forecast_end])
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Styling
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_title(f"{title} (Test Set: {len(test_data)} periods)")
    ax.set_xlabel('Date')
    ax.set_ylabel('Mean Rainfall (rfh_avg)')
    plt.xticks(rotation=45, ha='right')
    
    # Legend
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Layout
    plt.tight_layout()
    
    return fig

# Tabs for each algorithm
country_tab, arima_tab, prophet_tab, sarima_tab, holtwinters_tab, comparison_tab = st.tabs([
    "Country Comparison", "ARIMA", "Prophet", "SARIMA", "Holt-Winters", "Comparison"
])

with country_tab:
    st.header("Country-level Rainfall Forecasting Comparison")
    st.markdown("""
    This tab compares all four forecasting algorithms on the national-level rainfall time series.
    You can adjust the test set size in the sidebar to see how each model performs.
    """)
    
    # Load and prepare data
    data_path = processed_dir / "rainfall_minimal_clean.csv"
    if not data_path.exists():
        st.error(f"Country-level minimal rainfall data not found at {data_path}. Please run the data processing script first.")
        st.stop()
    
    df = pd.read_csv(data_path, parse_dates=['date'])
    country_df = df.groupby('date', as_index=False)['rfh_avg'].mean().sort_values('date')
    st.write(f"Total periods: {len(country_df)}")

    if len(country_df) <= n_test_global:
        st.error(f"Not enough data to split: only {len(country_df)} periods.")
    else:
        train = country_df.iloc[:-n_test_global]
        test = country_df.iloc[-n_test_global:]
        y_train = train['rfh_avg'].values
        results = {}
        forecasts = {}
        
        # ARIMA
        try:
            st.write("Attempting ARIMA fit...")
            arima_model = ARIMA(y_train, order=(1,1,1)).fit()
            arima_forecast = arima_model.forecast(steps=n_test_global)
            arima_mae = mean_absolute_error(test['rfh_avg'], arima_forecast)
            results['ARIMA'] = arima_mae
            forecasts['ARIMA'] = arima_forecast
        except Exception as e:
            st.warning(f"ARIMA error: {e}")
            results['ARIMA'] = None
        
        # SARIMA
        try:
            sarima_model = SARIMAX(y_train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)
            sarima_forecast = sarima_model.forecast(steps=n_test_global)
            sarima_mae = mean_absolute_error(test['rfh_avg'], sarima_forecast)
            results['SARIMA'] = sarima_mae
            forecasts['SARIMA'] = sarima_forecast
        except Exception as e:
            st.warning(f"SARIMA error: {e}")
            results['SARIMA'] = None
        # Holt-Winters
        try:
            hw_model = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=12).fit()
            hw_forecast = hw_model.forecast(steps=n_test_global)
            hw_mae = mean_absolute_error(test['rfh_avg'], hw_forecast)
            results['Holt-Winters'] = hw_mae
            forecasts['Holt-Winters'] = hw_forecast
        except Exception as e:
            st.warning(f"Holt-Winters error: {e}")
            results['Holt-Winters'] = None
        # Prophet
        if prophet_available:
            try:
                st.write("Attempting Prophet fit...")
                prophet_train = train.rename(columns={'date': 'ds', 'rfh_avg': 'y'})[['ds', 'y']]
                m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
                m.fit(prophet_train)
                future = m.make_future_dataframe(periods=n_test_global, freq='M')
                forecast = m.predict(future)
                prophet_forecast = forecast['yhat'].tail(n_test_global).values
                prophet_mae = mean_absolute_error(test['rfh_avg'], prophet_forecast)
                results['Prophet'] = prophet_mae
                forecasts['Prophet'] = prophet_forecast
            except Exception as e:
                st.warning(f"Prophet error: {e}")
                results['Prophet'] = None
        else:
            st.info("Prophet is not installed. Skipping Prophet forecast.")
            results['Prophet'] = None

        # Show results
        st.subheader("Model Accuracy (MAE)")
        st.write(pd.DataFrame(list(results.items()), columns=["Model", "MAE"]))
        
        # Plot
        fig = create_forecast_plot(
            historical_data=train.reset_index(), 
            test_data=test, 
            forecast_data=forecasts,
            title='Country-level Rainfall Forecasts (Last 3 Years + Forecast)'
        )
        st.pyplot(fig)
        
        # Summary
        best_model = min([(k, v) for k, v in results.items() if v is not None], key=lambda x: x[1], default=None)
        st.markdown("""
        **Summary:**
        - The model with the lowest MAE is considered the most accurate for this test set.
        - Lower MAE means predictions are closer to actual values.
        - Try different test set sizes in the sidebar to see how model performance changes.
        """)
        if best_model:
            st.success(f"Best model: {best_model[0]} (MAE: {best_model[1]:.2f})")
        else:
            st.warning("No model produced a valid forecast.")

with arima_tab:
    st.header("ARIMA Forecasting")
    st.markdown("ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model for time series forecasting.")
    try:
        # Split data
        train_data, test_data = split_train_test(national_data, n_test_global)
        
        # Fit model on training data
        ts_train = train_data.set_index(date_col)['rfh_avg'].dropna()
        model = ARIMA(ts_train, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=n_test_global)
        
        # Calculate MAE if we have test data
        if len(test_data) > 0:
            mae = mean_absolute_error(test_data['rfh_avg'], forecast)
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        
        # Create plot
        fig = create_forecast_plot(
            historical_data=national_data,
            test_data=test_data,
            forecast_data=forecast,
            title='ARIMA Forecast'
        )
        st.pyplot(fig)
        st.text(model_fit.summary())
    except Exception as e:
        st.error(f"ARIMA model error: {e}")

with prophet_tab:
    st.header("Prophet Forecasting")
    st.markdown("Prophet is a time series forecasting model developed by Facebook, designed for business time series with strong seasonal effects.")
    
    if not prophet_available:
        st.warning("Prophet is not installed. Please install Prophet to use this feature.")
    else:
        try:
            # Split data
            train_data, test_data = split_train_test(national_data, n_test_global)
            
            # Prepare data for Prophet
            prophet_df = train_data[[date_col, 'rfh_avg']].rename(columns={date_col: 'ds', 'rfh_avg': 'y'}).dropna()
            m = Prophet(yearly_seasonality=True, daily_seasonality=False, weekly_seasonality=False)
            m.fit(prophet_df)
            
            # Make forecast
            future = m.make_future_dataframe(periods=n_test_global, freq='M')
            forecast = m.predict(future)
            
            # Calculate MAE if we have test data
            if len(test_data) > 0:
                mae = mean_absolute_error(test_data['rfh_avg'], forecast['yhat'].tail(n_test_global))
                st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
            
            # Create plot
            fig = create_forecast_plot(
                historical_data=national_data,
                test_data=test_data,
                forecast_data=forecast['yhat'].tail(n_test_global),
                title='Prophet Forecast'
            )
            st.pyplot(fig)
            
            # Show components
            st.subheader("Forecast Components")
            fig_comp = m.plot_components(forecast)
            st.pyplot(fig_comp)
                
        except Exception as e:
            st.error(f"Prophet model error: {e}")

with sarima_tab:
    st.header("SARIMA Forecasting")
    st.markdown("SARIMA (Seasonal ARIMA) extends ARIMA to handle seasonality in time series data.")
    try:
        # Split data
        train_data, test_data = split_train_test(national_data, n_test_global)
        
        # Fit model
        ts_train = train_data.set_index(date_col)['rfh_avg'].dropna()
        model = SARIMAX(ts_train, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=n_test_global)
        
        # Calculate MAE if we have test data
        if len(test_data) > 0:
            mae = mean_absolute_error(test_data['rfh_avg'], forecast)
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        
        # Create plot
        fig = create_forecast_plot(
            historical_data=national_data,
            test_data=test_data,
            forecast_data=forecast,
            title='SARIMA Forecast'
        )
        st.pyplot(fig)
        st.text(model_fit.summary())
    except Exception as e:
        st.error(f"SARIMA model error: {e}")

with holtwinters_tab:
    st.header("Holt-Winters Forecasting")
    st.markdown("Holt-Winters Exponential Smoothing captures level, trend, and seasonality in time series data.")
    try:
        # Split data
        train_data, test_data = split_train_test(national_data, n_test_global)
        
        # Fit model
        ts_train = train_data.set_index(date_col)['rfh_avg'].dropna()
        model = ExponentialSmoothing(ts_train, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        forecast = model_fit.forecast(n_test_global)
        
        # Calculate MAE if we have test data
        if len(test_data) > 0:
            mae = mean_absolute_error(test_data['rfh_avg'], forecast)
            st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        
        # Create plot
        fig = create_forecast_plot(
            historical_data=national_data,
            test_data=test_data,
            forecast_data=forecast,
            title='Holt-Winters Forecast'
        )
        st.pyplot(fig)
        st.text(model_fit.summary())
    except Exception as e:
        st.error(f"Holt-Winters model error: {e}")

with comparison_tab:
    st.header("Model Comparison")
    st.markdown("""
    This tab compares the forecasts of all models for the national rainfall data. We show:
    - The historical data and all model forecasts on one plot
    - The forecasted values for the next 12 periods
    - A simple error metric (MAE) for each model, based on the test set
    - A plain-language summary to help you interpret the results
    """)
    
    # Add diagnostic information
    st.subheader("Data Diagnostics")
    ts = national_data.set_index(date_col)['rfh_avg'].dropna()
    
    # Check total data points
    st.write(f"Total data points: {len(ts)}")
    
    # Check for NaNs in last 12 points
    last_12 = ts.tail(12)
    nan_count = last_12.isna().sum()
    st.write(f"NaN values in last 12 points: {nan_count}")
    
    # Display last 12 points
    st.write("Last 12 data points:")
    st.dataframe(last_12)
    
    # Add a separator
    st.markdown("---")
    
    import warnings
    warnings.filterwarnings("ignore")
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    try:
        from prophet import Prophet
        prophet_available = True
    except ImportError:
        prophet_available = False
    ts = national_data.set_index(date_col)['rfh_avg'].dropna()
    results = {}
    forecast_index = pd.date_range(ts.index[-1] + pd.offsets.DateOffset(months=1), periods=12, freq='M')
    # Prepare test set (last 12 points)
    if len(ts) < 48:
        st.warning("Not enough data for a robust comparison (need at least 48 data points). Results may be less reliable.")
    train = ts.iloc[:-12] if len(ts) > 24 else ts
    test = ts.iloc[-12:] if len(ts) > 24 else None
    # Helper for MAE calculation
    def safe_mae(forecast, test):
        try:
            if forecast is None or test is None:
                return None
            # Convert both to numpy arrays for consistent comparison
            if isinstance(forecast, (pd.Series, pd.DataFrame)):
                forecast = forecast.values
            if isinstance(test, (pd.Series, pd.DataFrame)):
                test = test.values
            forecast = np.array(forecast).flatten()
            test = np.array(test).flatten()
            
            # Ensure same length
            min_len = min(len(forecast), len(test))
            forecast = forecast[:min_len]
            test = test[:min_len]
            
            # Check for valid data
            if min_len == 0 or np.isnan(forecast).any() or np.isnan(test).any():
                return None
                
            return float(np.mean(np.abs(forecast - test)))
        except Exception as e:
            st.write(f"MAE calculation error: {str(e)}")
            return None
    # ARIMA
    try:
        st.write("Attempting ARIMA fit...")
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        arima_forecast = model_fit.forecast(steps=12)
        # Fix index to use forecast_index
        arima_forecast.index = forecast_index
        st.write(f"ARIMA forecast generated: {arima_forecast[:5]}")
        results['ARIMA'] = {'forecast': arima_forecast, 'model': model_fit}
        if test is not None:
            mae = safe_mae(arima_forecast[:len(test)], test)
            st.write(f"ARIMA MAE calculation: {mae}")
            if mae is not None:
                results['ARIMA']['mae'] = mae
    except Exception as e:
        st.write(f"ARIMA error: {str(e)}")
        results['ARIMA'] = {'error': str(e)}
    # Prophet
    if prophet_available:
        try:
            st.write("Attempting Prophet fit...")
            prophet_df = train.reset_index().rename(columns={date_col: 'ds', 'rfh_avg': 'y'}).dropna()
            m = Prophet()
            m.fit(prophet_df)
            future = m.make_future_dataframe(periods=12, freq='M')
            forecast = m.predict(future)
            prophet_forecast = forecast['yhat'].iloc[-12:].values
            st.write(f"Prophet forecast generated: {prophet_forecast[:5]}")
            prophet_forecast = pd.Series(prophet_forecast, index=test.index if test is not None and len(test)==12 else forecast_index)
            results['Prophet'] = {'forecast': prophet_forecast, 'model': m}
            if test is not None:
                mae = safe_mae(prophet_forecast[:len(test)], test)
                st.write(f"Prophet MAE calculation: {mae}")
                if mae is not None:
                    results['Prophet']['mae'] = mae
        except Exception as e:
            st.write(f"Prophet error: {str(e)}")
            results['Prophet'] = {'error': str(e)}
    else:
        results['Prophet'] = {'error': 'Prophet not installed'}
    # SARIMA
    try:
        model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
        model_fit = model.fit(disp=False)
        sarima_forecast = model_fit.forecast(steps=12)
        # Fix index to use forecast_index
        sarima_forecast.index = forecast_index
        results['SARIMA'] = {'forecast': sarima_forecast, 'model': model_fit}
        if test is not None:
            mae = safe_mae(sarima_forecast[:len(test)], test)
            if mae is not None:
                results['SARIMA']['mae'] = mae
    except Exception as e:
        results['SARIMA'] = {'error': str(e)}
    # Holt-Winters
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
        model_fit = model.fit()
        hw_forecast = model_fit.forecast(12)
        # Fix index to use forecast_index
        hw_forecast.index = forecast_index
        results['Holt-Winters'] = {'forecast': hw_forecast, 'model': model_fit}
        if test is not None:
            mae = safe_mae(hw_forecast[:len(test)], test)
            if mae is not None:
                results['Holt-Winters']['mae'] = mae
    except Exception as e:
        results['Holt-Winters'] = {'error': str(e)}
    # Plot all forecasts
    fig, ax = plt.subplots(figsize=(12,6))
    ts.plot(ax=ax, label='Historical', color='black')
    for name, res in results.items():
        if 'forecast' in res:
            pd.Series(res['forecast'], index=forecast_index[:len(res['forecast'])]).plot(ax=ax, label=f'{name} Forecast')
    ax.set_title('Forecast Comparison')
    ax.legend()
    st.pyplot(fig)
    # Show forecast tables
    st.subheader("Forecasted Values (Next 12 Periods)")
    
    # Create forecast table with consistent date formatting
    forecast_table = pd.DataFrame(
        {name: res['forecast'].values if isinstance(res.get('forecast'), (pd.Series, np.ndarray)) 
         else res.get('forecast') if 'forecast' in res else None 
         for name, res in results.items()}, 
        index=forecast_index
    )
    
    # Format dates consistently
    forecast_table.index = forecast_table.index.strftime('%Y-%m-%d')
    
    # Debug information
    st.write("Debug: Forecast details:")
    for name, res in results.items():
        if 'forecast' in res:
            forecast = res['forecast']
            st.write(f"{name}:")
            st.write(f"  Type: {type(forecast)}")
            st.write(f"  Values (first 3 dates):")
            for i, (idx, val) in enumerate(forecast.head(3).items()):
                st.write(f"    {idx.strftime('%Y-%m-%d')}: {val:.4f}")
            if 'mae' in res:
                st.write(f"  MAE: {res['mae']:.4f}")
    
    st.dataframe(forecast_table)

    # Show MAE with more detail and sorting
    st.subheader("Model Accuracy (MAE on Last 12 Known Points)")
    mae_table = {}
    for name, res in results.items():
        if 'mae' in res and res['mae'] is not None and not (np.isnan(res['mae']) or np.isinf(res['mae'])):
            mae_table[name] = res['mae']
    
    # Sort models by MAE
    sorted_models = sorted(mae_table.items(), key=lambda x: x[1])
    
    # Display sorted MAE table with formatting
    mae_display = {}
    for name, mae in sorted_models:
        mae_display[name] = f"{mae:.2f}"
    st.write("Models sorted by accuracy (lower is better):")
    st.write(mae_display)
    # Plain-language summary
    st.subheader("Summary (Simple Terms)")
    valid_maes = [(name, res['mae']) for name, res in results.items() if 'mae' in res and res['mae'] is not None and not (np.isnan(res['mae']) or np.isinf(res['mae']))]
    summary = ""
    if valid_maes:
        best_model = min(valid_maes, key=lambda x: x[1])
        summary += f"- The model with the lowest error (MAE) on recent data is **{best_model[0]}** (MAE: {best_model[1]:.2f}).\n"
    else:
        summary += "- No model produced a valid MAE. This may be due to insufficient data or model convergence issues.\n"
    for name, res in results.items():
        if 'error' in res:
            summary += f"- {name} could not run: {res['error']}\n"
    summary += "- Lower MAE means the model's predictions are closer to the actual values.\n"
    summary += "- All models forecast the next 12 periods, but their predictions may differ.\n"
    summary += "- For stable, seasonal data, Holt-Winters and SARIMA often perform well.\n"
    summary += "- For data with trend and less seasonality, ARIMA or Prophet may be better.\n"
    st.markdown(summary) 