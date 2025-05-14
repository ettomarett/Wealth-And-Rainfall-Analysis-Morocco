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
country_tab, arima_tab, prophet_tab, sarima_tab, holtwinters_tab, report_tab = st.tabs([
    "Country Comparison", "ARIMA", "Prophet", "SARIMA", "Holt-Winters", "Report & Conclusion"
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
        st.subheader(f"Model Accuracy (MAE) for {n_test_global} test periods")
        results_df = pd.DataFrame(list(results.items()), columns=["Model", "MAE"])
        results_df = results_df.sort_values('MAE', ascending=True).reset_index(drop=True)
        st.dataframe(results_df)
        
        # Plot
        fig = create_forecast_plot(
            historical_data=train.reset_index(), 
            test_data=test, 
            forecast_data=forecasts,
            title=f'Country-level Rainfall Forecasts (Test Set: {n_test_global} periods)'
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

with report_tab:
    st.header("Rainfall Forecasting Models: Analysis & Conclusion")
    
    st.subheader("Overview of Models")
    
    # ARIMA
    st.markdown("""
    #### 1. ARIMA (AutoRegressive Integrated Moving Average)
    - **Strengths:**
        - Excellent for handling trends in data
        - Works well with stationary or differenced stationary data
        - Good for short-term forecasting
        - Simple and interpretable model structure
    - **Limitations:**
        - Cannot naturally handle seasonal data
        - Assumes linear relationships in data
        - Requires stationary data or differencing
        - May struggle with complex patterns
    """)
    
    # Prophet
    st.markdown("""
    #### 2. Prophet
    - **Strengths:**
        - Robust handling of missing data and outliers
        - Automatically detects seasonality at multiple levels
        - Handles holidays and special events
        - Works well with non-linear trends
        - User-friendly with minimal parameter tuning
    - **Limitations:**
        - May overfit with limited data
        - Less transparent than traditional statistical models
        - Can be computationally intensive
        - Sometimes produces overly smooth forecasts
    """)
    
    # SARIMA
    st.markdown("""
    #### 3. SARIMA (Seasonal ARIMA)
    - **Strengths:**
        - Explicitly models seasonal patterns
        - Combines trend and seasonal components
        - Good for data with clear seasonal cycles
        - Statistically rigorous approach
    - **Limitations:**
        - More complex parameter tuning required
        - Needs longer time series for seasonal patterns
        - Can be sensitive to parameter choices
        - Assumes consistent seasonal patterns
    """)
    
    # Holt-Winters
    st.markdown("""
    #### 4. Holt-Winters
    - **Strengths:**
        - Handles both trend and seasonality
        - Adaptive to changing patterns
        - Simple to understand and implement
        - Works well with clear seasonal patterns
    - **Limitations:**
        - Can be sensitive to outliers
        - Requires at least 2 full seasonal cycles
        - May struggle with irregular patterns
        - Less flexible than some modern approaches
    """)
    
    # Comparative Analysis
    st.subheader("Comparative Analysis")
    st.markdown("""
    Based on our rainfall forecasting results with 60 test periods:
    
    1. **Best Performers:**
       - Prophet (MAE: 8.4131) and Holt-Winters (MAE: 8.9054) show superior performance
       - Both models handle Morocco's rainfall patterns effectively
    
    2. **Seasonal Patterns:**
       - Prophet's ability to detect multiple seasonality levels proves valuable
       - Holt-Winters' adaptive approach works well with Morocco's seasonal variations
    
    3. **Traditional Models:**
       - SARIMA (MAE: 10.2341) performs moderately well
       - ARIMA (MAE: 11.2997) shows higher error rates, likely due to seasonal patterns
    
    4. **Overall Insights:**
       - Modern approaches (Prophet) outperform traditional methods
       - Adaptive methods show better accuracy for Morocco's rainfall patterns
    """)
    
    # Conclusion
    st.subheader("Conclusion & Recommendations")
    st.markdown("""
    For Morocco's rainfall forecasting:
    
    1. **Best Overall Model:**
       - Prophet demonstrates the highest accuracy (MAE: 8.4131)
       - Particularly effective at capturing complex patterns in Moroccan rainfall
    
    2. **Complementary Approaches:**
       - Use Prophet as the primary forecasting tool
       - Holt-Winters as a robust backup or verification model
       - SARIMA for specific cases where traditional methods are preferred
       - ARIMA mainly for short-term, trend-focused analysis
    
    3. **Practical Application:**
       - Prioritize Prophet for long-term forecasting
       - Consider ensemble methods weighted towards Prophet and Holt-Winters
       - Regular retraining improves forecast accuracy
    
    4. **Future Improvements:**
       - Fine-tune Prophet's hyperparameters for even better performance
       - Explore combining Prophet and Holt-Winters in an ensemble
       - Incorporate additional climate variables
       - Consider regional variations in model selection
    """)
    
    # References
    st.markdown("""
    ---
    ### References
    1. Box, G. E. P., & Jenkins, G. (1990). Time Series Analysis, Forecasting and Control
    2. Taylor, S. J., & Letham, B. (2017). Prophet: Forecasting at scale
    3. Hyndman, R.J., & Athanasopoulos, G. (2021). Forecasting: Principles and Practice
    """) 