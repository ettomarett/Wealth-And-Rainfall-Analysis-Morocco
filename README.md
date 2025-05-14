# Morocco Wealth-Rainfall Analysis Dashboard

## Project Overview
This project explores the relationship between wealth and rainfall patterns across Morocco's regions. It integrates geospatial, economic, and meteorological data to provide interactive visualizations and data-driven insights for researchers, policymakers, and the public.

## Features
- Unified dataset combining wealth and rainfall data at the regional level
- Interactive Streamlit dashboard with:
  - Rainfall and wealth data exploration
  - Geographic visualizations (choropleth maps, wealth points)
  - Correlation and seasonal analysis
  - Data dictionary and methodology explanations
- Efficient data processing pipeline

## Data Sources
- **Rainfall Data:** Historical rainfall measurements for Moroccan regions
- **Wealth Data:** Relative Wealth Index (RWI) with geospatial coordinates
- **Administrative Boundaries:** GeoJSON and CSV files for Morocco's regions

## Project Structure
```
├── process_data.py                # Main data processing script
├── processed_data/                # Processed datasets and map data
├── Datasets/                      # Raw datasets (CSV, GeoJSON)
├── streamlit/
│   ├── app.py                     # Main Streamlit app
│   └── pages/
│       ├── 01_rainfall_data.py    # Rainfall data exploration
│       ├── 02_wealth_data.py      # Wealth data exploration
│       ├── 03_region_boundaries.py# Region boundaries visualization
│       └── 04_unified_dataset.py  # Unified dataset analysis
├── docs/                          # Documentation (optional)
```

## How to Run
### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
- Place raw datasets in the `Datasets/` directory (see Data Sources above).
- Run the data processing script to generate processed files:
```bash
python process_data.py
```

### 3. Launch the Dashboard
```bash
cd streamlit
streamlit run app.py
```

The dashboard will open in your browser. Use the sidebar to navigate between data views.

## Usage
- **Rainfall Data:** Explore rainfall patterns by region and time.
- **Wealth Data:** Visualize the distribution of wealth points and regional averages.
- **Region Boundaries:** View administrative boundaries and related statistics.
- **Unified Dataset:** Analyze the combined wealth-rainfall data, correlations, and seasonal effects.

## Troubleshooting
- If you see missing data warnings, ensure all required files are present in `Datasets/` and `processed_data/`.
- If the map is slow, only a sample of wealth points is shown for performance.
- For errors about missing files, rerun `process_data.py` and check file paths.

## Credits
- Developed by Omar Ettalbi
- Data sources: [The Humanitarian Data Exchange](https://data.humdata.org/).

---
For questions or contributions, please open an issue or pull request. 