"""
Morocco Wealth-Rainfall Analysis Dashboard - Rainfall Data Page

This page allows users to load and visualize rainfall data across Morocco.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import plotly.express as px
from pathlib import Path
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set page title
st.set_page_config(
    page_title="Rainfall Data - Morocco Analysis",
    page_icon="🌧️",
    layout="wide"
)

# Get the correct path to the datasets
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
dataset_dir = project_root / "Datasets"
processed_dir = project_root / "processed_data"

st.title("Rainfall Data Analysis")
st.markdown("Explore rainfall patterns across different regions in Morocco.")

# Load rainfall data from processed files
@st.cache_data
def load_processed_data():
    """Load and cache the processed rainfall data and metrics"""
    try:
        # Check for processed data first
        rainfall_path = processed_dir / "cleaned_rainfall_data.csv"
        metrics_path = processed_dir / "rainfall_metrics.csv"
        
        # If processed data exists, load it
        if rainfall_path.exists() and metrics_path.exists():
            rainfall_df = pd.read_csv(rainfall_path)
            rainfall_df['date'] = pd.to_datetime(rainfall_df['date'])
            
            rainfall_metrics = pd.read_csv(metrics_path)
            
            return rainfall_df, rainfall_metrics
        else:
            st.warning("Processed data files not found. Please run the data processing script first.")
            return None, None
    except Exception as e:
        st.error(f"Error loading processed rainfall data: {e}")
        return None, None

# Load admin data for region names
@st.cache_data
def load_admin_data():
    """Load and cache the admin data for region names"""
    try:
        # Try to load from processed data first
        filepath = processed_dir / "admin_data.csv"
        if filepath.exists():
            admin_df = pd.read_csv(filepath)
            return admin_df
        else:
            # Fall back to raw data
            filepath = dataset_dir / "mar_adm2.csv"
            if filepath.exists():
                admin_df = pd.read_csv(filepath)
                return admin_df
            else:
                st.error(f"Could not find admin data file at {filepath}")
                return None
    except Exception as e:
        st.error(f"Error loading admin data: {e}")
        return None

# Load the data
with st.spinner("Loading rainfall data..."):
    rainfall_df, rainfall_metrics = load_processed_data()
    admin_df = load_admin_data()

# Check if data loaded successfully
if rainfall_df is None:
    st.error("Failed to load rainfall data. Please run the data processing script first.")
    
    # Add button to run processing script
    if st.button("Run Data Processing"):
        st.info("Running data processing script...")
        try:
            import subprocess
            result = subprocess.run(["python", str(project_root / "process_data.py")], capture_output=True, text=True)
            st.success("Data processing completed! Please refresh the page.")
            st.code(result.stdout)
        except Exception as e:
            st.error(f"Error running processing script: {e}")
    
    st.stop()

# Merge admin data to get region names
if admin_df is not None:
    region_names = admin_df[['ADM2_PCODE', 'ADM2_EN']].drop_duplicates()
    merged_rainfall = rainfall_df.merge(region_names, on='ADM2_PCODE', how='left')
else:
    merged_rainfall = rainfall_df
    st.warning("Admin data not loaded. Region names will not be displayed.")

# Main tabs for organizing content
main_tabs = st.tabs(["Overview", "Regional Data", "Time Series Analysis", "Map Visualization", "Understanding the Data"])

# Tab 1: Overview
with main_tabs[0]:
    st.header("Rainfall Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Records", len(rainfall_df))
        st.metric("Unique Regions", rainfall_df['ADM2_PCODE'].nunique())

    with col2:
        date_range = f"{rainfall_df['date'].min().strftime('%Y-%m-%d')} to {rainfall_df['date'].max().strftime('%Y-%m-%d')}"
        st.metric("Date Range", date_range)
        st.metric("Years Covered", len(rainfall_df['date'].dt.year.unique()))

    # Display the first few rows of the dataset
    with st.expander("View raw rainfall data"):
        st.write("First 100 rows of the rainfall dataset:")
        st.dataframe(merged_rainfall.head(100))
        
    # Download the processed data
    st.subheader("Download Processed Data")
    if rainfall_metrics is not None:
        csv = rainfall_metrics.to_csv(index=False)
        st.download_button(
            label="Download Rainfall Metrics as CSV",
            data=csv,
            file_name="rainfall_metrics.csv",
            mime="text/csv",
        )
    
    # Show data dictionary
    st.subheader("Data Dictionary")
    with st.expander("Rainfall Metrics Explanation"):
        st.markdown("""
        | Metric | Description |
        | ------ | ----------- |
        | rfh_avg | Average rainfall (mm) |
        | r1h_avg | Average 1-day maximum rainfall (mm) |
        | r3h_avg | Average 3-day maximum rainfall (mm) |
        | rfh_avg_cv | Coefficient of variation for rainfall |
        | rfh_avg_winter | Average winter rainfall (mm) |
        | rfh_avg_spring | Average spring rainfall (mm) |
        | rfh_avg_summer | Average summer rainfall (mm) |
        | rfh_avg_autumn | Average autumn rainfall (mm) |
        | seasonal_variability | Measure of seasonal rainfall variability |
        """)

# Tab 2: Regional Data
with main_tabs[1]:
    st.header("Explore Regional Rainfall Data")
    
    # Data selection options
    col1, col2, col3 = st.columns(3)

    with col1:
        # Select region
        regions = sorted(rainfall_df['ADM2_PCODE'].unique())
        if admin_df is not None:
            region_dict = dict(zip(admin_df['ADM2_PCODE'], admin_df['ADM2_EN']))
            regions_with_names = [f"{code} - {region_dict.get(code, 'Unknown')}" for code in regions]
            selected_region_with_name = st.selectbox("Select Region", regions_with_names, key="region_selector_regional")
            selected_region = selected_region_with_name.split(" - ")[0]
        else:
            selected_region = st.selectbox("Select Region", regions, key="region_selector_regional_nonames")

    with col2:
        # Select year if available
        years = sorted(rainfall_df['date'].dt.year.unique())
        selected_year = st.selectbox("Select Year", years, key="year_selector_regional")

    with col3:
        # Select rainfall metric
        rainfall_metrics_list = [col for col in rainfall_df.columns if col.startswith('r') and col not in ['rfq', 'r1q', 'r3q']]
        selected_metric = st.selectbox("Select Rainfall Metric", 
                                      rainfall_metrics_list,
                                      format_func=lambda x: f"{x} (mm)",
                                      key="metric_selector_regional")
    
    # Filter data based on selection
    region_data = rainfall_df[rainfall_df['ADM2_PCODE'] == selected_region].copy()
    if 'ADM2_EN' in region_data.columns:
        region_name = region_data['ADM2_EN'].iloc[0]
    else:
        region_name = selected_region
    
    # Add year column if not already present
    if 'year' not in region_data.columns:
        region_data['year'] = region_data['date'].dt.year
        
    year_data = region_data[region_data['year'] == selected_year]
    
    # Monthly patterns
    st.subheader(f"Monthly Rainfall Patterns for {region_name} in {selected_year}")
    
    if len(year_data) >= 12:  # Only show if we have enough data
        # Add month column
        year_data['month'] = year_data['date'].dt.month
        
        # Group by month and calculate mean
        monthly_data = year_data.groupby('month')[selected_metric].mean().reset_index()
        
        fig = px.bar(
            monthly_data,
            x='month',
            y=selected_metric,
            title=f"Monthly {selected_metric} in {selected_year} for {region_name}",
            labels={selected_metric: f"Avg Rainfall ({selected_metric}) in mm", 'month': 'Month'},
            color=selected_metric,
            color_continuous_scale='Blues'
        )
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ),
            yaxis_title=f"Avg Rainfall ({selected_metric}) in mm",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"Insufficient data for monthly patterns in {selected_year}")
    
    # Regional comparison
    st.subheader("Regional Rainfall Comparison")
    
    if rainfall_metrics is not None:
        # Merge with admin data for region names
        if admin_df is not None:
            metrics_with_names = rainfall_metrics.merge(
                admin_df[['ADM2_PCODE', 'ADM2_EN']].drop_duplicates(),
                on='ADM2_PCODE',
                how='left'
            )
            metrics_with_names['region'] = metrics_with_names['ADM2_EN']
        else:
            metrics_with_names = rainfall_metrics.copy()
            metrics_with_names['region'] = metrics_with_names['ADM2_PCODE']
        
        # Filter for relevant metric
        metric_col = f"{selected_metric}_mean"
        if metric_col in metrics_with_names.columns:
            fig = px.bar(
                metrics_with_names.sort_values(metric_col, ascending=False).head(10),
                x='region',
                y=metric_col,
                title=f"Top 10 Regions by Average {selected_metric}",
                labels={metric_col: f"Avg {selected_metric} (mm)", 'region': 'Region'},
                color=metric_col,
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(
                xaxis_title="Region",
                yaxis_title=f"Average {selected_metric} (mm)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Metric {metric_col} not available in the calculated metrics")
    else:
        st.warning("Rainfall metrics not calculated. Cannot show regional comparison.")

# Tab 3: Time Series Analysis
with main_tabs[2]:
    st.header("Rainfall Time Series Analysis")
    
    # Select region
    regions = sorted(rainfall_df['ADM2_PCODE'].unique())
    if admin_df is not None:
        region_dict = dict(zip(admin_df['ADM2_PCODE'], admin_df['ADM2_EN']))
        regions_with_names = [f"{code} - {region_dict.get(code, 'Unknown')}" for code in regions]
        selected_region_with_name = st.selectbox("Select Region", regions_with_names, key="region_selector_timeseries")
        selected_region = selected_region_with_name.split(" - ")[0]
    else:
        selected_region = st.selectbox("Select Region", regions, key="region_selector_timeseries_nonames")
    
    # Select rainfall metric
    rainfall_metrics_list = [col for col in rainfall_df.columns if col.startswith('r') and col not in ['rfq', 'r1q', 'r3q']]
    selected_metric = st.selectbox("Select Rainfall Metric", 
                                  rainfall_metrics_list,
                                  format_func=lambda x: f"{x} (mm)",
                                  key="metric_selector_timeseries")
    
    # Filter data based on selection
    region_data = rainfall_df[rainfall_df['ADM2_PCODE'] == selected_region].copy()
    if 'ADM2_EN' in region_data.columns:
        region_name = region_data['ADM2_EN'].iloc[0]
    else:
        region_name = selected_region
    
    # Time series plot for selected region
    st.subheader(f"Rainfall Time Series for {region_name}")
    
    # Create time series plot
    fig = px.line(
        region_data, 
        x='date', 
        y=selected_metric,
        title=f"{selected_metric} over time in {region_name}",
        labels={selected_metric: f"Rainfall ({selected_metric}) in mm", 'date': 'Date'},
        line_shape='linear'
    )
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"Rainfall ({selected_metric}) in mm",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add year-over-year comparison
    if 'year' not in region_data.columns:
        region_data['year'] = region_data['date'].dt.year
    if 'month' not in region_data.columns:
        region_data['month'] = region_data['date'].dt.month
    
    # Group by year and month
    monthly_by_year = region_data.groupby(['year', 'month'])[selected_metric].mean().reset_index()
    
    # Create a pivot table for year-over-year comparison
    pivot_data = monthly_by_year.pivot(index='month', columns='year', values=selected_metric)
    
    st.subheader("Year-over-Year Monthly Comparison")
    st.write("Average monthly rainfall across different years")
    
    # Create year-over-year comparison plot
    fig = px.line(
        pivot_data, 
        x=pivot_data.index, 
        y=pivot_data.columns,
        labels={'value': f"Average {selected_metric} (mm)", 'x': 'Month'},
        title=f"Year-over-Year Monthly {selected_metric} Comparison for {region_name}"
    )
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ),
        xaxis_title="Month",
        yaxis_title=f"Average {selected_metric} (mm)",
        height=500,
        legend_title="Year"
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Tab 4: Map Visualization
with main_tabs[3]:
    st.header("Rainfall Map Visualization")
    st.subheader("Regional Rainfall Distribution")

    try:
        # Load admin boundaries from GeoJSON
        admin_boundaries_path = dataset_dir / "merged_adm2_data.geojson"
        
        if admin_boundaries_path.exists():
            admin_gdf = gpd.read_file(str(admin_boundaries_path))
            
            if rainfall_metrics is not None:
                # Allow user to select which rainfall metric to display
                available_rainfall_cols = [col for col in rainfall_metrics.columns 
                                        if (col.startswith('rfh_') or col.startswith('r1h_') or col.startswith('r3h_')) 
                                        and not col.endswith('_cv') and rainfall_metrics[col].dtype in ['float64', 'float32']]
                
                if available_rainfall_cols:
                    selected_map_metric = st.selectbox(
                        "Select rainfall metric to display on map",
                        available_rainfall_cols,
                        format_func=lambda x: f"{x} (mm)",
                        key="rainfall_map_metric"
                    )
                    
                    # Merge rainfall metrics with admin boundaries
                    merged_gdf = admin_gdf.merge(
                        rainfall_metrics[['ADM2_PCODE', selected_map_metric]],
                        on='ADM2_PCODE',
                        how='left'
                    )
                    
                    # Create a folium map
                    def create_rainfall_map(gdf, rainfall_column):
                        """Create a Folium map with region boundaries colored by rainfall"""
                        # Center map on Morocco
                        center = [31.7917, -7.0926]
                        m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')
                        
                        # Get min and max values for colormap
                        vmin = gdf[rainfall_column].min()
                        vmax = gdf[rainfall_column].max()
                        
                        # Create colormap
                        colormap = LinearColormap(
                            colors=['#ffffd9', '#edf8b1', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'],
                            vmin=vmin,
                            vmax=vmax,
                            caption=f'Rainfall ({rainfall_column}) (mm)'
                        )
                        
                        # Add choropleth layer
                        folium.GeoJson(
                            gdf.__geo_interface__,
                            style_function=lambda feature: {
                                'fillColor': colormap(feature['properties'][rainfall_column]) 
                                            if feature['properties'][rainfall_column] is not None else 'gray',
                                'color': 'black',
                                'weight': 1,
                                'fillOpacity': 0.7
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=['ADM2_PCODE', 'ADM2_EN', rainfall_column],
                                aliases=['Region Code:', 'Region Name:', f'Rainfall ({rainfall_column}) (mm):'],
                                localize=True
                            )
                        ).add_to(m)
                        
                        # Add the colormap to the map
                        colormap.add_to(m)
                        
                        return m
                    
                    # Create and display the map
                    rainfall_map = create_rainfall_map(merged_gdf, selected_map_metric)
                    folium_static(rainfall_map, width=1000, height=600)
                    
                    # Add explanation for the map
                    st.info("""
                    **About this map:** This choropleth map shows the distribution of rainfall across different regions in Morocco.
                    
                    Each region is colored according to its rainfall value, with darker blue colors indicating higher rainfall amounts.
                    
                    You can hover over a region to see its name and rainfall value, or change the rainfall metric using the dropdown above.
                    """)
                else:
                    st.warning("No suitable rainfall metrics available for mapping")
            else:
                st.warning("Rainfall metrics not available for mapping")
        else:
            st.warning("Admin boundaries file not found. Cannot display rainfall map.")
    except Exception as e:
        st.error(f"Error creating rainfall map: {e}")

# Tab 5: Understanding the Data
with main_tabs[4]:
    st.header("Understanding Rainfall Data")
    st.markdown("""
    ### What is this rainfall data about?

    This dataset shows how much rain falls in different parts of Morocco. Think of it as a record of how wet each region gets during the year.

    ### Why does it matter?

    - **Farmers need rain to grow crops**
    - **Rain fills reservoirs for drinking water**
    - **Too little rain can cause droughts and problems for communities**

    ### Where did this data come from?

    The data comes from weather stations and satellites. Scientists collect daily measurements and combine them to see the big picture.

    ### What do the numbers mean?

    - Numbers like "rfh_avg" show the average amount of rain (in millimeters)
    - 10mm means a small puddle, 100mm means a lot of rain

    ### How to read the maps and charts

    - **Blue colors** mean more rain
    - **Yellow/brown** mean less rain
    - Time series charts show how rain changes over time
    - Bar charts compare rain between regions

    This data helps us understand Morocco's climate and how it affects people and the land.
    """)
    
    # Add a simple rainfall visualization
    st.subheader("Visualizing Rainfall Amounts")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Light Rain: 1-5mm**")
        st.image("https://www.metoffice.gov.uk/binaries/content/gallery/metofficegovuk/hero-images/weather/rain/light-rain.jpg", 
                caption="Enough to make the ground damp")
    
    with col2:
        st.markdown("**Moderate Rain: 5-20mm**")
        st.image("https://www.metoffice.gov.uk/binaries/content/gallery/metofficegovuk/hero-images/weather/rain/heavy-rain.jpg", 
                caption="Puddles form, good for crops")
    
    with col3:
        st.markdown("**Heavy Rain: >20mm**")
        st.image("https://cdn.theatlantic.com/thumbor/Qs1addQltI-uBlXLJj5fFjMv5uA=/0x25:1199x699/1600x900/media/img/mt/2022/07/Flooding_Rain/original.jpg", 
                caption="Can cause flooding in some areas") 