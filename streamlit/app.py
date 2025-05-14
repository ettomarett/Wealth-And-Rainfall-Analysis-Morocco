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
    page_title="Morocco Wealth-Rainfall Analysis",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page content
st.title("Morocco Wealth-Rainfall Analysis")

st.markdown("""
This dashboard provides interactive visualizations and analysis of the relationship between 
relative wealth index and rainfall patterns across different regions in Morocco.

### Project Overview
This data-driven decision-making project integrates several key datasets:
1. Administrative regions data for Morocco
2. Historical rainfall measurements
3. Relative wealth index data with geospatial coordinates
4. Geographic boundaries (GeoJSON) for precise mapping

### Key Research Questions
- Is there a correlation between regional wealth and rainfall patterns?
- Do wealthier regions have more consistent rainfall patterns?
- Are economically disadvantaged areas more susceptible to rainfall extremes?
- How might changing rainfall patterns affect economic prosperity in different regions?

### Navigate the Dashboard
Use the sidebar to explore different aspects of the data analysis:
- **Rainfall Data**: Explore rainfall patterns across Morocco
- **Wealth Data**: Examine relative wealth distribution
- **Region Boundaries**: View Morocco's administrative regions
- **Unified Dataset**: Analyze the combined wealth-rainfall data
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
st.markdown("© 2025 Morocco Wealth-Rainfall Analysis Project") 