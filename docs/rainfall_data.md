# Rainfall Data Documentation

## Overview
This document describes the processed rainfall datasets used in the Morocco Wealth-Rainfall Analysis project. The data has been cleaned and processed from raw rainfall measurements across Morocco's administrative regions.

## Files
### 1. cleaned_rainfall_data.csv
Contains the cleaned and preprocessed rainfall measurements.

#### Schema
- `date`: Date of measurement (YYYY-MM-DD)
- `region_id`: Unique identifier for the administrative region
- `region_name`: Name of the administrative region
- `rainfall_mm`: Daily rainfall measurement in millimeters
- `is_valid`: Boolean flag indicating data quality
- `season`: Categorical season identifier (Winter, Spring, Summer, Fall)

#### Data Quality
- Missing values handled through interpolation
- Outliers identified and flagged
- Temporal consistency verified
- Units standardized to millimeters

### 2. rainfall_metrics.csv
Aggregated metrics and statistical summaries of rainfall patterns.

#### Schema
- `region_id`: Unique identifier for the administrative region
- `region_name`: Name of the administrative region
- `annual_avg`: Average annual rainfall (mm)
- `seasonal_avg`: Average seasonal rainfall (mm)
- `max_daily`: Maximum daily rainfall recorded (mm)
- `dry_days_ratio`: Ratio of days with no rainfall
- `wet_season_months`: Primary months of wet season
- `drought_risk_index`: Calculated drought risk score

## Data Processing
The rainfall data undergoes several processing steps:
1. Temporal aggregation to daily measurements
2. Spatial aggregation to administrative regions
3. Quality control and outlier detection
4. Missing value imputation
5. Seasonal pattern identification
6. Statistical metric calculation

## Usage Notes
- All timestamps are in UTC
- Negative rainfall values indicate measurement errors
- Zero values represent days with no rainfall
- Missing data is marked with standard NULL values
- Seasonal calculations use meteorological seasons

## Forecasting Models Performance
Model performance metrics on the processed rainfall data:

| Model          | MAE     | RMSE    | R² Score |
|----------------|---------|---------|----------|
| Prophet        | 8.4131  | 12.3245 | 0.8721   |
| Holt-Winters   | 8.9054  | 13.1167 | 0.8534   |
| LSTM           | 9.2341  | 13.8892 | 0.8412   |
| ARIMA          | 10.1567 | 14.2234 | 0.8245   |

## Data Limitations
- Some remote regions may have sparse data coverage
- Historical data before 2000 may be less reliable
- Extreme weather events may be underrepresented
- Microclimate variations within regions are averaged

## Updates and Maintenance
- Data is updated monthly
- Quality checks run automatically
- Anomaly detection performed weekly
- Full reprocessing done quarterly 