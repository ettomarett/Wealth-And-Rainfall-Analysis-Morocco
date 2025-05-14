# Morocco Wealth-Rainfall Analysis Project Plan (Revised)

## Project Overview
This project aims to analyze the relationship between relative wealth index and rainfall patterns across different regions in Morocco. We'll use Python for data processing and analysis, and Streamlit for creating an interactive dashboard to visualize our findings.

## Data Sources
1. **mar_adm2.csv** - Morocco's administrative regions data (metadata)
2. **mar-rainfall-adm2-full.csv** - Rainfall data across Moroccan regions
3. **morocco_relative_wealth_index.csv** - Geospatial wealth index data
4. **merged_adm2_data.geojson** - Geographic boundaries for Moroccan regions
5. **adm2_with_rwi.geojson** - Pre-aggregated wealth data by administrative region

## Project Phases

### Phase 1: Data Preparation and Integration
- [ ] Clean and preprocess rainfall and administrative datasets
- [ ] Load GeoJSON data with region boundaries and pre-aggregated wealth metrics
- [ ] Calculate rainfall metrics by region (annual and seasonal)
- [ ] Create a unified dataset linking regions, wealth metrics, and rainfall patterns
- [ ] Handle any missing values

### Phase 2: Exploratory Data Analysis
- [ ] Calculate basic statistics to understand the distributions
- [ ] Perform correlation analysis between wealth and various rainfall metrics
- [ ] Identify patterns, trends, and potential relationships
- [ ] Generate initial visualizations (scatterplots, heatmaps, choropleth maps)
- [ ] Analyze regional variations and clustering

### Phase 3: Advanced Analysis
- [ ] Conduct regression analysis to quantify wealth-rainfall relationships
- [ ] Perform time series analysis to identify temporal patterns
- [ ] Investigate potential causal relationships
- [ ] Segment regions based on wealth-rainfall characteristics
- [ ] Identify outlier regions for case study analysis

### Phase 4: Streamlit Dashboard Development
- [x] Design dashboard layout and user flow
- [x] Create interactive maps using GeoJSON boundaries showing:
  - Regional wealth distribution
  - Rainfall patterns
  - Combined wealth-rainfall visualizations
- [x] Implement filtering capabilities by:
  - Time period
  - Region/province
  - Wealth/rainfall metrics
- [x] Build interactive visualization components:
  - Correlation plots
  - Time series graphs
  - Statistical summaries
- [x] Add explanatory text and insights for users
- [x] Organize interface with tabs for better navigation
- [x] Enhance RWI explanation to clarify "relative" measurement context
- [x] Improve map sizing and interactivity for better user experience

### Phase 5: Documentation and Deployment
- [ ] Create technical documentation for code and methodologies
- [ ] Write summary of findings and insights
- [ ] Prepare user guide for the Streamlit application
- [ ] Set up deployment for the Streamlit app

## Technical Requirements

### Environment Setup
- Python 3.8+
- Required libraries:
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib/seaborn (static visualizations)
  - geopandas (geospatial data handling)
  - folium (interactive maps)
  - scikit-learn (statistical analysis)
  - statsmodels (time series analysis)
  - streamlit (dashboard interface)
  - streamlit_folium (for embedding folium maps in Streamlit)
  - plotly (interactive visualizations)

### Project Structure
```
DDDM/
│
├── Datasets/
│   ├── mar_adm2.csv
│   ├── mar-rainfall-adm2-full.csv
│   ├── morocco_relative_wealth_index.csv
│   ├── merged_adm2_data.geojson
│   └── adm2_with_rwi.geojson
│
├── docs/
│   ├── plan.md
│   └── findings.md (to be created)
│
├── processed_data/
│   ├── rainfall_metrics.csv
│   └── unified_dataset.csv
│
├── src/
│   ├── data_processing/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── integration.py
│   │   └── pipeline.py
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── exploratory.py
│   │   ├── statistical.py
│   │   └── geospatial.py
│   │
│   └── visualization/
│       ├── __init__.py
│       ├── maps.py
│       └── plots.py
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_wealth_rainfall_analysis.ipynb
│   └── 03_visualization_prototypes.ipynb
│
├── streamlit/
│   ├── app.py
│   ├── pages/
│   │   ├── 01_overview.py
│   │   ├── 02_regional_analysis.py
│   │   └── 03_time_series.py
│   │
│   └── components/
│       ├── maps.py
│       └── charts.py
│
├── run_data_processing.py
├── requirements.txt
└── README.md
```

## Timeline
1. **Week 1**: Data preparation and integration
2. **Week 2**: Exploratory data analysis
3. **Week 3**: Advanced analysis
4. **Week 4**: Dashboard development
5. **Week 5**: Documentation and finalization

## Expected Outcomes
1. A comprehensive analysis of the relationship between wealth and rainfall in Morocco
2. Interactive Streamlit dashboard with accurate geographic visualizations
3. Insights that could inform policy decisions related to:
   - Climate adaptation strategies
   - Economic development initiatives
   - Resource allocation
   - Infrastructure planning 