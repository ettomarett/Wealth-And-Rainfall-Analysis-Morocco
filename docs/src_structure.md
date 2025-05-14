# Source Code Documentation

## Directory Structure
```
src/
├── data_processing/
│   ├── cleaning.py
│   └── geospatial.py
├── models/
│   ├── prophet_model.py
│   ├── lstm_model.py
│   ├── holtwinters_model.py
│   └── arima_model.py
└── visualization/
    ├── maps.py
    ├── plots.py
    └── dashboard_components.py
```

## Data Processing Module
### cleaning.py
Functions for data cleaning and preprocessing:
- `load_rainfall_data()`: Loads and validates raw rainfall data
- `calculate_rainfall_metrics()`: Computes statistical metrics for rainfall
- `handle_missing_values()`: Implements interpolation strategies
- `validate_data_quality()`: Performs quality checks and flagging
- `standardize_units()`: Ensures consistent measurement units

### geospatial.py
Geospatial data processing utilities:
- `load_admin_boundaries()`: Loads administrative boundary data
- `load_aggregated_wealth()`: Processes wealth index data
- `extract_wealth_metrics()`: Calculates regional wealth statistics
- `merge_with_rainfall()`: Combines rainfall and wealth data
- `compute_spatial_statistics()`: Calculates spatial metrics

## Models Module
### prophet_model.py
Facebook Prophet implementation for rainfall forecasting:
- Model configuration and hyperparameters
- Training and validation procedures
- Seasonality handling
- Forecast generation
- Performance metrics (MAE: 8.4131)

### lstm_model.py
Long Short-Term Memory neural network:
- Network architecture
- Data preprocessing
- Training pipeline
- Prediction generation
- Performance metrics (MAE: 9.2341)

### holtwinters_model.py
Holt-Winters time series forecasting:
- Exponential smoothing implementation
- Seasonal decomposition
- Parameter optimization
- Forecast generation
- Performance metrics (MAE: 8.9054)

### arima_model.py
ARIMA time series modeling:
- Model order selection
- Stationarity tests
- Parameter estimation
- Forecast generation
- Performance metrics (MAE: 10.1567)

## Visualization Module
### maps.py
Geospatial visualization components:
- `create_choropleth()`: Regional choropleth maps
- `plot_wealth_points()`: Wealth distribution points
- `render_admin_boundaries()`: Administrative boundary overlays
- `create_heatmap()`: Density and heat maps
- `add_map_annotations()`: Labels and markers

### plots.py
Statistical visualization utilities:
- `plot_rainfall_trends()`: Time series rainfall plots
- `create_correlation_matrix()`: Wealth-rainfall correlations
- `generate_boxplots()`: Distribution analysis
- `plot_forecasts()`: Model prediction visualization
- `create_seasonal_plots()`: Seasonal pattern analysis

### dashboard_components.py
Streamlit dashboard components:
- `create_sidebar()`: Navigation and filters
- `render_metrics()`: Key performance indicators
- `create_data_table()`: Interactive data tables
- `plot_comparison()`: Multi-variable comparisons
- `generate_report()`: Automated reporting

## Dependencies
Key libraries used:
- pandas: Data manipulation
- geopandas: Geospatial operations
- numpy: Numerical computations
- scikit-learn: Machine learning utilities
- tensorflow: Deep learning (LSTM)
- prophet: Time series forecasting
- streamlit: Dashboard creation
- folium: Interactive maps
- matplotlib/seaborn: Static visualizations

## Usage Examples
```python
# Data Processing
from src.data_processing.cleaning import load_rainfall_data
from src.data_processing.geospatial import merge_with_rainfall

# Load and process data
rainfall_data = load_rainfall_data()
combined_data = merge_with_rainfall(rainfall_data)

# Model Training
from src.models.prophet_model import train_prophet_model
model = train_prophet_model(data=rainfall_data)

# Visualization
from src.visualization.maps import create_choropleth
map_fig = create_choropleth(data=combined_data)
```

## Development Guidelines
1. Follow PEP 8 style guidelines
2. Write unit tests for new functions
3. Document all functions with docstrings
4. Use type hints for function parameters
5. Handle errors gracefully
6. Log important operations
7. Optimize for performance where needed

## Testing
Each module includes unit tests:
- `tests/test_cleaning.py`
- `tests/test_geospatial.py`
- `tests/test_models.py`
- `tests/test_visualization.py`

## Future Improvements
1. Add support for more data sources
2. Implement ensemble forecasting
3. Optimize large dataset handling
4. Add more interactive visualizations
5. Enhance error handling
6. Improve documentation coverage 