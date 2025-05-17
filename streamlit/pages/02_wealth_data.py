"""
Morocco Wealth-Rainfall Analysis Dashboard - Wealth Data Page

This page allows users to explore the wealth index data for Morocco.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import geopandas as gpd
import plotly.express as px
import folium
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
from pathlib import Path
from branca.colormap import LinearColormap

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Set page title
st.set_page_config(
    page_title="Wealth Data - Morocco Analysis",
    page_icon="💰",
    layout="wide"
)

# Get the correct path to the datasets
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
dataset_dir = project_root / "Datasets"
processed_dir = project_root / "processed_data"

st.title("Relative Wealth Index Analysis")
st.markdown("Explore the distribution of relative wealth across Morocco.")

# Brief explanation about RWI
st.info("""
**About the Relative Wealth Index (RWI):** The RWI is a measure of wealth that is *relative to Morocco's average wealth level*.
- A value of 0 represents average wealth for Morocco
- Positive values (e.g., +1, +2) indicate areas wealthier than the Moroccan average
- Negative values (e.g., -1, -2) indicate areas poorer than the Moroccan average
- The scale typically ranges from -10 (extremely poor) to +10 (extremely wealthy)

The index is derived from proxy indicators visible in satellite imagery and other geospatial data.
""")

# Load wealth data from processed files
@st.cache_data
def load_wealth_points():
    """Load and cache the individual wealth points data"""
    try:
        # Try to load from processed data first
        filepath = processed_dir / "wealth_points_sample.csv"
        if filepath.exists():
            wealth_df = pd.read_csv(filepath)
            return wealth_df
        else:
            # Fall back to raw data
            filepath = dataset_dir / "morocco_relative_wealth_index.csv"
            if filepath.exists():
                wealth_df = pd.read_csv(filepath)
                return wealth_df
            else:
                st.error(f"Could not find wealth data file at {filepath}")
                return None
    except Exception as e:
        st.error(f"Error loading wealth points data: {e}")
        return None

@st.cache_data
def load_wealth_metrics():
    """Load and cache the pre-aggregated wealth data by region"""
    try:
        filepath = processed_dir / "wealth_metrics.csv"
        if filepath.exists():
            wealth_metrics = pd.read_csv(filepath)
            return wealth_metrics
        else:
            st.warning("Processed wealth metrics not found. Some visualizations may not be available.")
            return None
    except Exception as e:
        st.error(f"Error loading wealth metrics: {e}")
        return None

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
with st.spinner("Loading wealth data..."):
    wealth_points = load_wealth_points()
    wealth_metrics = load_wealth_metrics()
    admin_df = load_admin_data()

# Check if data loaded successfully
if wealth_points is None:
    st.error("Failed to load wealth data. Please run the data processing script first.")
    
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

# Main content tabs
main_tabs = st.tabs(["Overview", "Individual Points", "Regional Analysis", "Map Visualization", "Understanding the Data"])

# Tab 1: Overview
with main_tabs[0]:
    st.header("Wealth Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if wealth_points is not None:
            st.metric("Number of Individual Data Points", len(wealth_points))
            st.metric("Average RWI", round(wealth_points['rwi'].mean(), 3))
        
        if wealth_metrics is not None:
            st.metric("Number of Regions with Data", len(wealth_metrics))
    
    with col2:
        if wealth_points is not None:
            st.metric("Min RWI", round(wealth_points['rwi'].min(), 3))
            st.metric("Max RWI", round(wealth_points['rwi'].max(), 3))
            
            # Calculate positive and negative percentages
            positive_pct = (wealth_points['rwi'] > 0).mean() * 100
            negative_pct = (wealth_points['rwi'] < 0).mean() * 100
            st.metric("Locations Above Average Wealth", f"{positive_pct:.1f}%")
    
    # Add RWI explanation
    with st.expander("What is the Relative Wealth Index (RWI)?"):
        st.markdown("""
        The **Relative Wealth Index (RWI)** is a measure developed to estimate the relative standard of living
        using machine learning models on geospatial data. It typically ranges from -10 (most poor) to +10 (most wealthy).
        
        The index is based on features such as:
        - Building characteristics visible in satellite imagery
        - Nighttime light intensity
        - Infrastructure characteristics
        - Population density
        
        RWI values are standardized so that 0 represents the average wealth, with negative values
        indicating below-average wealth and positive values indicating above-average wealth.
        """)
    
    # View the first few rows
    with st.expander("View raw data samples"):
        st.subheader("Individual Wealth Points")
        if wealth_points is not None:
            st.write("First 100 rows of individual wealth points:")
            st.dataframe(wealth_points.head(100))
        
        st.subheader("Regional Wealth Metrics")
        if wealth_metrics is not None:
            if admin_df is not None:
                wealth_metrics_with_names = wealth_metrics.merge(
                    admin_df[['ADM2_PCODE', 'ADM2_EN']].drop_duplicates(),
                    on='ADM2_PCODE',
                    how='left'
                )
                st.dataframe(wealth_metrics_with_names)
            else:
                st.dataframe(wealth_metrics)

# Tab 2: Individual Points
with main_tabs[1]:
    st.header("Individual Wealth Points")
    
    if wealth_points is not None:
        # Histogram of RWI distribution
        st.subheader("Distribution of Relative Wealth Index")
        fig = px.histogram(
            wealth_points, 
            x="rwi", 
            nbins=50,
            title="Distribution of Relative Wealth Index (RWI)",
            labels={"rwi": "Relative Wealth Index"},
            color_discrete_sequence=["#1f77b4"]
        )
        
        fig.update_layout(
            xaxis_title="Relative Wealth Index (RWI)",
            yaxis_title="Count",
            height=500
        )
        
        # Add a vertical line at x=0 to mark the average wealth level
        fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Average Wealth", annotation_position="top right")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics by RWI category
        st.subheader("Statistics by Wealth Category")
        
        # Create wealth categories
        wealth_points['wealth_category'] = pd.cut(
            wealth_points['rwi'],
            bins=[-float('inf'), -2, -1, 0, 1, 2, float('inf')],
            labels=['Very Poor', 'Poor', 'Below Average', 'Above Average', 'Wealthy', 'Very Wealthy']
        )
        
        # Count by category
        category_counts = wealth_points['wealth_category'].value_counts().sort_index()
        
        # Create a bar chart
        fig = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Distribution by Wealth Category",
            labels={'x': 'Wealth Category', 'y': 'Count'},
            color=category_counts.index,
            color_discrete_sequence=px.colors.sequential.Viridis
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Individual wealth points data not loaded. Please check the dataset.")

# Tab 3: Regional Analysis
with main_tabs[2]:
    st.header("Aggregated Wealth by Region")
    
    if wealth_metrics is not None:
        # Merge with admin data for region names if available
        if admin_df is not None:
            wealth_metrics_with_names = wealth_metrics.merge(
                admin_df[['ADM2_PCODE', 'ADM2_EN']].drop_duplicates(),
                on='ADM2_PCODE',
                how='left'
            )
            if 'ADM2_EN' in wealth_metrics_with_names.columns:
                wealth_metrics_with_names['region_name'] = wealth_metrics_with_names['ADM2_EN']
            else:
                wealth_metrics_with_names['region_name'] = wealth_metrics_with_names['ADM2_PCODE']
        else:
            wealth_metrics_with_names = wealth_metrics.copy()
            wealth_metrics_with_names['region_name'] = wealth_metrics_with_names['ADM2_PCODE']
        
        # Display metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Number of Regions", len(wealth_metrics))
            
            # Check if rwi_mean column exists before calculating average
            if 'rwi_mean' in wealth_metrics.columns:
                st.metric("Avg Regional RWI", round(wealth_metrics['rwi_mean'].mean(), 3))
            else:
                # Try to find any column that might contain RWI
                rwi_cols = [col for col in wealth_metrics.columns if 'rwi' in col.lower() and wealth_metrics[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                if rwi_cols:
                    st.metric("Avg Regional RWI", round(wealth_metrics[rwi_cols[0]].mean(), 3))
                else:
                    st.metric("Avg Regional RWI", "N/A")
            
        with col2:
            # Safely access the min and max regions
            if 'rwi_mean' in wealth_metrics_with_names.columns and 'region_name' in wealth_metrics_with_names.columns:
                poorest_region = wealth_metrics_with_names.loc[wealth_metrics_with_names['rwi_mean'].idxmin()]
                richest_region = wealth_metrics_with_names.loc[wealth_metrics_with_names['rwi_mean'].idxmax()]
                
                st.metric("Poorest Region", 
                         f"{poorest_region['region_name']} ({round(poorest_region['rwi_mean'], 3)})")
                st.metric("Richest Region", 
                         f"{richest_region['region_name']} ({round(richest_region['rwi_mean'], 3)})")
            else:
                # Try to find alternative column
                rwi_cols = [col for col in wealth_metrics_with_names.columns if 'rwi' in col.lower() and wealth_metrics_with_names[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                if rwi_cols and 'region_name' in wealth_metrics_with_names.columns:
                    rwi_col = rwi_cols[0]
                    poorest_region = wealth_metrics_with_names.loc[wealth_metrics_with_names[rwi_col].idxmin()]
                    richest_region = wealth_metrics_with_names.loc[wealth_metrics_with_names[rwi_col].idxmax()]
                    
                    st.metric("Poorest Region", 
                             f"{poorest_region['region_name']} ({round(poorest_region[rwi_col], 3)})")
                    st.metric("Richest Region", 
                             f"{richest_region['region_name']} ({round(richest_region[rwi_col], 3)})")
                else:
                    st.metric("Poorest Region", "N/A")
                    st.metric("Richest Region", "N/A")
        
        # Bar chart of wealth by region
        st.subheader("Wealth Comparison by Region")
        
        # Determine which wealth column to use for visualization
        wealth_col = 'rwi_mean'
        if wealth_col not in wealth_metrics_with_names.columns:
            # Try to find a suitable alternative column
            rwi_cols = [col for col in wealth_metrics_with_names.columns if 'rwi' in col.lower() and wealth_metrics_with_names[col].dtype in ['float64', 'float32', 'int64', 'int32']]
            if rwi_cols:
                wealth_col = rwi_cols[0]
            else:
                st.warning("No suitable wealth metric column found for visualization")
        
        # Only show charts if we have a valid wealth column
        if wealth_col in wealth_metrics_with_names.columns:
            region_tabs = st.tabs(["Top Regions", "Bottom Regions", "Wealth Inequality"])
            
            with region_tabs[0]:
                # Top 10 wealthiest regions
                fig = px.bar(
                    wealth_metrics_with_names.sort_values(wealth_col, ascending=False).head(10),
                    x='region_name',
                    y=wealth_col,
                    title="Top 10 Wealthiest Regions",
                    labels={wealth_col: 'Mean RWI', 'region_name': 'Region'},
                    color=wealth_col,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Mean Relative Wealth Index",
                    xaxis={'categoryorder':'total descending'},
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with region_tabs[1]:
                # Bottom 10 poorest regions
                fig = px.bar(
                    wealth_metrics_with_names.sort_values(wealth_col).head(10),
                    x='region_name',
                    y=wealth_col,
                    title="Bottom 10 Poorest Regions",
                    labels={wealth_col: 'Mean RWI', 'region_name': 'Region'},
                    color=wealth_col,
                    color_continuous_scale='Viridis'
                )
                
                fig.update_layout(
                    xaxis_title="Region",
                    yaxis_title="Mean Relative Wealth Index",
                    xaxis={'categoryorder':'total ascending'},
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            with region_tabs[2]:
                # Wealth inequality analysis
                st.subheader("Wealth Inequality Analysis")

                # Check if required columns exist for wealth inequality analysis
                required_cols = ['rwi_std', 'rwi_mean']
                if all(col in wealth_metrics_with_names.columns for col in required_cols):
                    # Add coefficient of variation for wealth inequality measure
                    wealth_metrics_with_names['rwi_cv'] = wealth_metrics_with_names['rwi_std'] / abs(wealth_metrics_with_names['rwi_mean'])
                    
                    # Top regions by wealth inequality
                    fig = px.bar(
                        wealth_metrics_with_names.sort_values('rwi_cv', ascending=False).head(10),
                        x='region_name',
                        y='rwi_cv',
                        title="Top 10 Regions by Wealth Inequality",
                        labels={'rwi_cv': 'Coefficient of Variation', 'region_name': 'Region'},
                        color='rwi_cv',
                        color_continuous_scale='Reds'
                    )
                    
                    fig.update_layout(
                        xaxis_title="Region",
                        yaxis_title="Wealth Inequality (Coefficient of Variation)",
                        xaxis={'categoryorder':'total descending'},
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    # Try to find alternative metrics for inequality
                    std_cols = [col for col in wealth_metrics_with_names.columns if 'std' in col.lower() and wealth_metrics_with_names[col].dtype in ['float64', 'float32']]
                    mean_cols = [col for col in wealth_metrics_with_names.columns if 'mean' in col.lower() and col not in std_cols and wealth_metrics_with_names[col].dtype in ['float64', 'float32']]
                    
                    if std_cols and mean_cols:
                        # Create a makeshift inequality measure
                        std_col = std_cols[0]
                        mean_col = mean_cols[0]
                        wealth_metrics_with_names['inequality_measure'] = wealth_metrics_with_names[std_col] / abs(wealth_metrics_with_names[mean_col])
                        
                        # Top regions by wealth inequality
                        fig = px.bar(
                            wealth_metrics_with_names.sort_values('inequality_measure', ascending=False).head(10),
                            x='region_name',
                            y='inequality_measure',
                            title="Top 10 Regions by Wealth Inequality (Approximated)",
                            labels={'inequality_measure': 'Inequality Measure', 'region_name': 'Region'},
                            color='inequality_measure',
                            color_continuous_scale='Reds'
                        )
                        
                        fig.update_layout(
                            xaxis_title="Region",
                            yaxis_title="Wealth Inequality (Approximated)",
                            xaxis={'categoryorder':'total descending'},
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("Unable to calculate wealth inequality metrics due to missing required data (standard deviation and mean values).")
        else:
            st.warning("Unable to display region comparison charts due to missing wealth data")
    else:
        st.error("Aggregated wealth data not loaded. Please run the data processing script.") 

# Tab 4: Map Visualization
with main_tabs[3]:
    st.header("Wealth Data Maps")
    
    map_tabs = st.tabs(["Individual Points Map", "Regional Choropleth Map"])
    
    with map_tabs[0]:
        # Map visualization with sample of points
        st.subheader("Spatial Distribution of Wealth Points")
        
        if wealth_points is not None:
            # Create a map centered on Morocco
            m = folium.Map(location=[31.7917, -7.0926], zoom_start=6)
            
            # Create a marker cluster for the points
            marker_cluster = MarkerCluster().add_to(m)
            
            # Add points to the map with color based on RWI value
            for idx, row in wealth_points.iterrows():
                # Set color based on RWI value
                if row['rwi'] < -1.0:
                    color = 'red'
                elif row['rwi'] < 0:
                    color = 'orange'
                elif row['rwi'] < 1.0:
                    color = 'blue'
                else:
                    color = 'green'
                    
                # Add marker
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"RWI: {row['rwi']:.2f}"
                ).add_to(marker_cluster)
            
            # Display the map
            st.write("Sample of wealth data points (color indicates wealth level)")
            folium_static(m, width=1000, height=600)
            
            # Add explanation about the map numbers
            st.info("""
            **Note about the map numbers:** The large numbers (like 144, 239, 338) shown on the map represent the count of data points clustered in that area, not the RWI values themselves. 
            
            The Relative Wealth Index (RWI) values typically range from -10 to +10. The colors of the clusters indicate the general wealth level (green for wealthier areas, yellow for middle-range, orange/red for poorer areas).
            
            You can zoom in to see smaller clusters and individual data points.
            """)
        else:
            st.error("Individual wealth points data not loaded. Please check the dataset.")
    
    with map_tabs[1]:
        # Wealth Distribution Choropleth Map
        st.subheader("Wealth Distribution Map")
        
        # Load admin boundaries for mapping
        try:
            # Load admin boundaries from GeoJSON
            admin_boundaries_path = dataset_dir / "merged_adm2_data.geojson"
            if admin_boundaries_path.exists() and wealth_metrics is not None:
                admin_gdf = gpd.read_file(str(admin_boundaries_path))
                
                # Determine which wealth column to use
                wealth_col = 'rwi_mean'
                if wealth_col not in wealth_metrics.columns:
                    rwi_cols = [col for col in wealth_metrics.columns if 'rwi' in col.lower() and wealth_metrics[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                    if rwi_cols:
                        wealth_col = rwi_cols[0]
                    else:
                        st.warning("No suitable wealth metric column found for mapping")
                
                # Allow user to select which wealth metric to display
                available_wealth_cols = [col for col in wealth_metrics.columns if 'rwi' in col.lower() and wealth_metrics[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                if available_wealth_cols:
                    selected_wealth_col = st.selectbox(
                        "Select wealth metric to display on map",
                        available_wealth_cols,
                        index=available_wealth_cols.index(wealth_col) if wealth_col in available_wealth_cols else 0,
                        key="wealth_map_metric"
                    )
                    
                    # Merge wealth metrics with admin boundaries
                    merged_gdf = admin_gdf.merge(
                        wealth_metrics[['ADM2_PCODE', selected_wealth_col]],
                        on='ADM2_PCODE',
                        how='left'
                    )
                    
                    # Create a folium map
                    def create_wealth_map(gdf, wealth_column):
                        """Create a Folium map with region boundaries colored by wealth"""
                        # Center map on Morocco
                        center = [31.7917, -7.0926]
                        m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')
                        
                        # Get min and max values for colormap
                        vmin = gdf[wealth_column].min()
                        vmax = gdf[wealth_column].max()
                        
                        # Create colormap
                        colormap = LinearColormap(
                            colors=['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641'],
                            vmin=vmin,
                            vmax=vmax,
                            caption=f'Relative Wealth Index ({wealth_column})'
                        )
                        
                        # Convert GeoDataFrame to a format that can be serialized to JSON
                        geo_data = gdf.copy()
                        # Convert any datetime columns to string
                        for col in geo_data.columns:
                            if pd.api.types.is_datetime64_any_dtype(geo_data[col]):
                                geo_data[col] = geo_data[col].astype(str)
                        
                        # Add choropleth layer
                        folium.GeoJson(
                            geo_data.__geo_interface__,
                            style_function=lambda feature: {
                                'fillColor': colormap(feature['properties'][wealth_column]) 
                                            if feature['properties'][wealth_column] is not None else 'gray',
                                'color': 'black',
                                'weight': 1,
                                'fillOpacity': 0.7
                            },
                            tooltip=folium.GeoJsonTooltip(
                                fields=['ADM2_PCODE', 'ADM2_EN', wealth_column],
                                aliases=['Region Code:', 'Region Name:', f'Wealth Index ({wealth_column}):'],
                                localize=True
                            )
                        ).add_to(m)
                        
                        # Add the colormap to the map
                        colormap.add_to(m)
                        
                        return m
                    
                    # Create and display the map
                    wealth_map = create_wealth_map(merged_gdf, selected_wealth_col)
                    folium_static(wealth_map, width=1000, height=600)
                    
                    # Add explanation for interpreting the map
                    st.info("""
                    **About this map:** This choropleth map shows the distribution of relative wealth across different regions in Morocco.
                    
                    The colors indicate wealth levels relative to Morocco's average:
                    - Red/Orange: Below average wealth (negative RWI)
                    - Yellow: Near average wealth (RWI close to 0)
                    - Green: Above average wealth (positive RWI)
                    
                    You can hover over a region to see its name and exact wealth index value.
                    """)
                else:
                    st.warning("No wealth metrics available for mapping")
            else:
                st.warning("Admin boundaries file or wealth metrics not found. Cannot display wealth map.")
        except Exception as e:
            st.error(f"Error creating wealth map: {e}") 

# Tab 5: Understanding the Wealth Data
with main_tabs[4]:
    st.header("Understanding the Wealth Data")
    st.markdown("""
    ### What is this wealth data about?

    This dataset shows how wealthy or poor different places in Morocco are, based on surveys and satellite images. 
    Each point on the map is like a "score" for a small area, showing if people there are generally richer or poorer.

    ### Why does it matter?

    - **Helps find where people need more support**
    - **Shows which areas are doing well or struggling**
    - **Useful for planning schools, hospitals, and roads**

    ### Where did this data come from?

    The data comes from a project that uses household surveys and satellite pictures to estimate wealth. 
    Experts use computers to look for clues in the images (like roof types, roads, fields) and combine that with survey answers.

    ### What do the numbers mean?

    - The main number is called the **Relative Wealth Index (RWI)**
    - Higher numbers mean more wealth, lower numbers mean less
    - It's "relative"—so it compares places to each other, not to a fixed standard

    ### How to read the maps and charts

    - **Blue or green** areas are usually wealthier
    - **Yellow or red** areas are usually poorer
    - Dots show individual survey points; shaded regions show averages

    This data helps us see patterns of wealth and poverty across Morocco.
    """) 