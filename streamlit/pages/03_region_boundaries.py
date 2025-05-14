"""
Morocco Wealth-Rainfall Analysis Dashboard - Region Boundaries Page

This page allows users to explore Morocco's administrative regions and boundaries.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import os
import sys
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap
import numpy as np
from pathlib import Path

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import data processing functions
from src.data_processing.geospatial import load_admin_boundaries

# Get the correct path to the datasets
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
dataset_dir = project_root / "Datasets"
processed_dir = project_root / "processed_data"

# Set page title
st.set_page_config(
    page_title="Region Boundaries - Morocco Analysis",
    page_icon="🗺️",
    layout="wide"
)

st.title("Morocco Administrative Regions")
st.markdown("Explore the geographic boundaries of Morocco's administrative regions.")

# Load admin boundaries
@st.cache_data
def load_boundaries():
    """Load and cache the administrative boundaries"""
    try:
        filepath = dataset_dir / "merged_adm2_data.geojson"
        gdf = load_admin_boundaries(str(filepath))
        return gdf
    except Exception as e:
        st.error(f"Error loading admin boundaries: {e}")
        return None

# Load admin data for region attributes
@st.cache_data
def load_admin_data():
    """Load and cache the admin data for region attributes"""
    try:
        filepath = dataset_dir / "mar_adm2.csv"
        if filepath.exists():
            admin_df = pd.read_csv(filepath)
            return admin_df
        else:
            filepath = processed_dir / "admin_data.csv"
            if filepath.exists():
                admin_df = pd.read_csv(filepath)
                return admin_df
            else:
                st.error(f"Admin data file not found")
                return None
    except Exception as e:
        st.error(f"Error loading admin data: {e}")
        return None

# Load the data
with st.spinner("Loading administrative boundaries..."):
    boundaries_gdf = load_boundaries()
    admin_df = load_admin_data()

# Check if data loaded successfully
if boundaries_gdf is None:
    st.error("Failed to load administrative boundaries. Please check the file path and format.")
    st.stop()

# Main content layout
st.header("1. Administrative Regions Overview")

col1, col2 = st.columns(2)

with col1:
    st.metric("Number of Regions", len(boundaries_gdf))
    
    # If admin data is available, show province and prefecture counts
    if admin_df is not None and 'type_en' in admin_df.columns:
        province_count = admin_df[admin_df['type_en'] == 'Province'].shape[0]
        prefecture_count = admin_df[admin_df['type_en'] == 'Prefecture'].shape[0]
        st.metric("Provinces", province_count)
        st.metric("Prefectures", prefecture_count)

with col2:
    # Calculate total area if available
    if 'AREA_SQKM' in boundaries_gdf.columns:
        total_area = boundaries_gdf['AREA_SQKM'].sum()
        st.metric("Total Area", f"{total_area:,.2f} km²")
        
        # Calculate average area
        avg_area = boundaries_gdf['AREA_SQKM'].mean()
        st.metric("Average Region Area", f"{avg_area:,.2f} km²")
    elif 'area_sqkm' in boundaries_gdf.columns:
        total_area = boundaries_gdf['area_sqkm'].sum()
        st.metric("Total Area", f"{total_area:,.2f} km²")
        
        # Calculate average area
        avg_area = boundaries_gdf['area_sqkm'].mean()
        st.metric("Average Region Area", f"{avg_area:,.2f} km²")

# Map visualization
st.header("2. Interactive Region Map")

# Helper function for displaying map
def create_folium_map(gdf, color_column=None, fill_color='YlOrRd', legend_name='Value'):
    """Create a Folium map with region boundaries"""
    
    # Center map on Morocco
    center = [31.7917, -7.0926]
    m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')
    
    # Create colormap if color column is provided
    if color_column and color_column in gdf.columns:
        # Get min and max values for colormap
        vmin = gdf[color_column].min()
        vmax = gdf[color_column].max()
        
        # Create colormap
        colormap = LinearColormap(
            colors=['#FFFFCC', '#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026'],
            vmin=vmin,
            vmax=vmax,
            caption=legend_name
        )
        
        # Add choropleth layer
        folium.GeoJson(
            gdf.__geo_interface__,
            style_function=lambda feature: {
                'fillColor': colormap(feature['properties'][color_column]) 
                             if feature['properties'][color_column] is not None else 'gray',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['ADM2_PCODE', 'ADM2_EN', color_column],
                aliases=['Region Code:', 'Region Name:', f'{legend_name}:'],
                localize=True
            )
        ).add_to(m)
        
        # Add the colormap to the map
        colormap.add_to(m)
    else:
        # Add boundaries without coloring
        folium.GeoJson(
            gdf.__geo_interface__,
            style_function=lambda feature: {
                'fillColor': '#3186cc',
                'color': 'black',
                'weight': 1,
                'fillOpacity': 0.7
            },
            tooltip=folium.GeoJsonTooltip(
                fields=['ADM2_PCODE', 'ADM2_EN'],
                aliases=['Region Code:', 'Region Name:'],
                localize=True
            )
        ).add_to(m)
    
    return m

# Map options
st.subheader("Region Boundary Map")

# Add area to boundaries for visualization if available
area_col = None
for candidate in ['AREA_SQKM', 'area', 'area_sqkm']:
    if candidate in boundaries_gdf.columns:
        area_col = candidate
        break
        
# Create visualization options
map_options = ["Plain Boundaries"]
if area_col:
    map_options.append(f"Color by {area_col}")

# Allow admin data metrics to be visualized if available
admin_metrics = []
if admin_df is not None:
    # If we have admin metrics, find numeric columns
    numeric_cols = admin_df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude ID columns
    admin_metrics = [col for col in numeric_cols if not col.startswith('ADM') and '_id' not in col.lower()]
    if admin_metrics:
        # Add admin metrics to map options
        for metric in admin_metrics:
            map_options.append(f"Color by {metric}")

# Select map visualization
selected_map = st.selectbox("Select Map Visualization", map_options)

# Get GeoDataFrame for map visualization
if selected_map == "Plain Boundaries":
    # Display plain boundaries
    m = create_folium_map(boundaries_gdf)
elif area_col and selected_map == f"Color by {area_col}":
    # Display map colored by area
    m = create_folium_map(
        boundaries_gdf,
        color_column=area_col,
        legend_name="Area (km²)"
    )
elif admin_df is not None and selected_map.startswith("Color by "):
    # Get metric name
    metric = selected_map.replace("Color by ", "")
    
    # Check if metric exists in admin data
    if metric in admin_df.columns:
        # Merge admin data with boundaries
        merged_gdf = boundaries_gdf.merge(
            admin_df[['ADM2_PCODE', metric]],
            on='ADM2_PCODE',
            how='left'
        )
        
        # Create map with merged data
        m = create_folium_map(
            merged_gdf,
            color_column=metric,
            legend_name=metric
        )
    else:
        st.warning(f"Metric {metric} not found in admin data.")
        m = create_folium_map(boundaries_gdf)
else:
    # Fallback to plain boundaries
    m = create_folium_map(boundaries_gdf)

# Display the map
folium_static(m)

# Region information table
st.header("3. Region Information")

# Show region info table
if admin_df is not None:
    # Merge with boundaries to get area if available
    if area_col:
        display_df = admin_df.merge(
            boundaries_gdf[['ADM2_PCODE', area_col]],
            on='ADM2_PCODE',
            how='left'
        )
    else:
        display_df = admin_df.copy()
    
    # Add area column name
    if area_col and area_col in display_df.columns:
        display_df = display_df.rename(columns={area_col: 'Area (km²)'})
    
    # Display options
    col_options = list(display_df.columns)
    default_cols = ['ADM2_PCODE', 'ADM2_EN', 'ADM1_EN']
    if 'type_en' in display_df.columns:
        default_cols.append('type_en')
    if 'Area (km²)' in display_df.columns:
        default_cols.append('Area (km²)')
    
    # Let user choose columns to display
    display_cols = st.multiselect(
        "Select columns to display",
        options=col_options,
        default=[col for col in default_cols if col in col_options]
    )
    
    if not display_cols:
        display_cols = default_cols
    
    # Display table
    st.dataframe(display_df[display_cols], use_container_width=True)
    
    # Download option
    csv = display_df[display_cols].to_csv(index=False)
    st.download_button(
        "Download Region Data as CSV",
        csv,
        "morocco_regions.csv",
        "text/csv",
        key='download-csv'
    )
    
else:
    st.dataframe(
        boundaries_gdf[['ADM2_PCODE', 'ADM2_EN']],
        use_container_width=True
    )

# Show data dictionary
with st.expander("Data Dictionary - Region Attributes"):
    st.markdown("""
    | Column | Description |
    | ------ | ----------- |
    | ADM2_PCODE | Unique region identifier code |
    | ADM2_EN | Region name in English |
    | ADM2_AR | Region name in Arabic (if available) |
    | ADM1_EN | Province name in English |
    | ADM1_PCODE | Province identifier code |
    | AREA_SQKM | Area in square kilometers |
    | type_en | Type of administrative unit (Province or Prefecture) |
    """) 