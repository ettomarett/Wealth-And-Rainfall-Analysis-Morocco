# Morocco Wealth-Rainfall Analysis Dashboard

## Project Overview
This project explores the relationship between wealth and rainfall patterns across Morocco's regions, incorporating advanced forecasting models for future rainfall predictions. It integrates geospatial, economic, and meteorological data to provide interactive visualizations and data-driven insights for researchers, policymakers, and the public.

## Key Findings
1. **Rainfall Forecasting**:
   - Prophet model shows highest accuracy (MAE: 8.4131)
   - Holt-Winters demonstrates strong performance (MAE: 8.9054)
   - Modern approaches outperform traditional methods
   - Seasonal patterns significantly impact prediction accuracy

2. **Wealth-Rainfall Relationship**:
   - Analysis of regional wealth distribution
   - Correlation studies between rainfall patterns and economic indicators
   - Impact assessment of rainfall variability on different regions
   - Identification of vulnerable areas

## Features
- Unified dataset combining wealth and rainfall data at the regional level
- Interactive Streamlit dashboard with:
  - Rainfall and wealth data exploration
  - Geographic visualizations (choropleth maps, wealth points)
  - Advanced forecasting models
  - Correlation and seasonal analysis
  - Data dictionary and methodology explanations
- Efficient data processing pipeline
- Machine learning models for rainfall prediction

## Data Sources
- **Rainfall Data:** Historical rainfall measurements for Moroccan regions
- **Wealth Data:** Relative Wealth Index (RWI) with geospatial coordinates
- **Administrative Boundaries:** GeoJSON and CSV files for Morocco's regions



## Environment Setup
### 1. Create and Activate Conda Environment
```bash
conda create -n dddm_env python=3.10
conda activate dddm_env
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Prepare Data
- Place raw datasets in the `Datasets/` directory
- Run the data processing script:
```bash
python process_data.py
```

### 4. Launch the Dashboard
```bash
cd streamlit
streamlit run app.py
```

## Key Dependencies
- streamlit==1.45.1
- pandas (latest)
- numpy==2.1.3
- tensorflow==2.19.0
- prophet==1.1.6
- geopandas==1.0.1
- matplotlib==3.10.0
- seaborn==0.13.2

## Usage
- **Home**: Overview of project goals and data status
- **Rainfall Data**: Explore historical patterns and forecasts
- **Wealth Data**: Visualize wealth distribution and metrics
- **Region Boundaries**: Interactive maps and regional statistics
- **Unified Dataset**: Combined analysis and correlations

## Troubleshooting
- If you see missing data warnings, ensure all required files are present in `Datasets/` and `processed_data/`
- For map performance issues, the dashboard uses a sampled subset of wealth points
- Run `python process_data.py` if processed data files are missing
- Check console logs for detailed error messages

## Project Status
Current development phase focuses on:
1. Data Preparation and Validation
2. Model Development and Testing
3. Dashboard Enhancement
4. Documentation and Testing

## Data sources: [The Humanitarian Data Exchange](https://data.humdata.org/).

[Rainfall Data](https://data.humdata.org/dataset/mar-rainfall-subnational).
[Relative Wealth Index Data](https://data.humdata.org/dataset/relative-wealth-index).
[Subnational Administrative Boundaries](https://data.humdata.org/dataset/geoboundaries-admin-boundaries-for-morocco).



## Credits
- Developed by ETTALBI OMAR
- © 2025 Morocco Wealth-Rainfall Analysis Project
---
For questions, contributions, or issues, please contact the project maintainers. 