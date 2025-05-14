"""
Morocco Wealth-Rainfall Analysis Dashboard

Main application file for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Get the correct path to the datasets
project_root = Path(os.path.dirname(os.path.abspath(__file__))).parent
dataset_dir = project_root / "Datasets"
processed_dir = project_root / "processed_data"

# Import data processing functions
from src.data_processing.cleaning import load_rainfall_data, calculate_rainfall_metrics
from src.data_processing.geospatial import (
    load_admin_boundaries, 
    load_aggregated_wealth,
    extract_wealth_metrics,
    merge_with_rainfall
)

# Set page configuration
st.set_page_config(
    page_title="Home",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Morocco Wealth-Rainfall Analysis")

st.markdown("""
This comprehensive dashboard analyzes the relationship between relative wealth indices and rainfall patterns 
across Morocco's regions, incorporating advanced forecasting models for future rainfall predictions.

### Project Overview
Our data-driven decision-making project combines multiple data sources and analytical approaches:
1. **Regional Data**: Administrative boundaries and socioeconomic indicators
2. **Historical Rainfall**: Detailed rainfall measurements across regions
3. **Wealth Metrics**: Relative wealth index with geospatial distribution
4. **Forecasting Models**: Advanced time series prediction using multiple algorithms

### Key Findings
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

### Navigate the Dashboard
Explore our comprehensive analysis through different sections:
- **Rainfall Data**: Historical patterns and regional distribution
- **Wealth Data**: Economic indicators and spatial analysis
- **Region Boundaries**: Administrative divisions and geographical context
- **Unified Dataset**: Integrated analysis of wealth and rainfall
- **Rainfall Forecasting**: Advanced predictive models and comparisons

### Project Impact
Our analysis provides valuable insights for:
- Regional development planning
- Resource allocation strategies
- Climate adaptation policies
- Economic resilience assessment
""")

# Sidebar for data loading status
st.sidebar.title("Data Status")

# Check both raw and processed data
raw_data_status = {
    "Raw Rainfall Data": (dataset_dir / "mar-rainfall-adm2-full.csv").exists(),
    "Raw Wealth Data": (dataset_dir / "morocco_relative_wealth_index.csv").exists(),
    "Raw Admin Data": (dataset_dir / "mar_adm2.csv").exists(),
    "Raw Admin Boundaries": (dataset_dir / "merged_adm2_data.geojson").exists(),
    "Raw Aggregated Wealth": (dataset_dir / "adm2_with_rwi.geojson").exists()
}

processed_data_status = {
    "Processed Rainfall Data": (processed_dir / "cleaned_rainfall_data.csv").exists(),
    "Processed Rainfall Metrics": (processed_dir / "rainfall_metrics.csv").exists(),
    "Processed Admin Data": (processed_dir / "admin_data.csv").exists(),
    "Processed Wealth Points": (processed_dir / "wealth_points_sample.csv").exists(),
    "Processed Unified Dataset": (processed_dir / "unified_dataset.csv").exists()
}

# Display raw data status
st.sidebar.subheader("Raw Data Status")
for data_type, available in raw_data_status.items():
    if available:
        st.sidebar.success(f"✅ {data_type}")
    else:
        st.sidebar.error(f"❌ {data_type}")

# Display processed data status
st.sidebar.subheader("Processed Data Status")
for data_type, available in processed_data_status.items():
    if available:
        st.sidebar.success(f"✅ {data_type}")
    else:
        st.sidebar.error(f"❌ {data_type}")

# Add information about the project phases
st.sidebar.markdown("---")
st.sidebar.subheader("Project Phases")
st.sidebar.markdown("""
1. **Data Preparation** *(current phase)*
2. Exploratory Analysis
3. Advanced Analysis
4. Dashboard Enhancement
5. Documentation
""")

# Main content area
st.header("Data Processing Information")

# Display information about processed files
summary_path = processed_dir / "processing_summary.txt"
if summary_path.exists():
    with open(summary_path, "r") as f:
        summary_content = f.read()
    
    st.code(summary_content, language="")
else:
    st.warning("Processing summary not found. Please run the data processing script first.")
    
    # Add button to run processing script
    if st.button("Run Data Processing"):
        st.info("Running data processing script...")
        try:
            import subprocess
            result = subprocess.run(["python", str(project_root / "process_data.py")], capture_output=True, text=True)
            st.success("Data processing completed!")
            st.code(result.stdout)
        except Exception as e:
            st.error(f"Error running processing script: {e}")

# Footer
st.markdown("---")
st.markdown("© 2025 Morocco Wealth-Rainfall Analysis Project | Created by ETTALBI OMAR") 