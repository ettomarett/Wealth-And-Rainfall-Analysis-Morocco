"""
Morocco Wealth-Rainfall Analysis Dashboard - Unified Dataset Page

This page integrates rainfall and wealth data to enable analysis of their relationship.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium, folium_static
from branca.colormap import linear
from branca.colormap import LinearColormap
from folium.plugins import MarkerCluster
from pathlib import Path
import json

# Add the parent directory to sys.path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Get the correct path to the datasets
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))).parent
dataset_dir = project_root / "Datasets"
processed_dir = project_root / "processed_data"

# Import data processing functions
from src.data_processing.cleaning import load_rainfall_data, calculate_rainfall_metrics, handle_missing_values
from src.data_processing.geospatial import (
    load_admin_boundaries, 
    load_aggregated_wealth,
    extract_wealth_metrics,
    merge_with_rainfall
)

# Set page title
st.set_page_config(
    page_title="Unified Dataset - Morocco Analysis",
    page_icon="🔄",
    layout="wide"
)

st.title("Unified Wealth-Rainfall Dataset")
st.markdown("Explore the relationship between wealth and rainfall across Morocco's regions.")

# Add cache functions at the top of the file
@st.cache_data
def load_map_data():
    """Load pre-calculated map data from JSON files"""
    try:
        map_data_dir = processed_dir / "map_data"
        
        # Load metadata
        with open(map_data_dir / "metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Load wealth features
        with open(map_data_dir / "wealth_features.json", 'r') as f:
            wealth_features = json.load(f)
            
        # Load region data
        with open(map_data_dir / "region_data.json", 'r') as f:
            region_data = json.load(f)
            
        # Load region statistics
        with open(map_data_dir / "region_stats.json", 'r') as f:
            region_stats = json.load(f)
            
        if not all([metadata, wealth_features, region_data, region_stats]):
            raise ValueError("One or more required map data files are empty")
            
        return metadata, wealth_features, region_data, region_stats
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        return None, None, None, None

# Helper function to load unified dataset
@st.cache_data
def load_unified_dataset():
    """Load unified dataset from processed files"""
    try:
        # Check for processed unified dataset
        unified_path = processed_dir / "unified_dataset.csv"
        
        if unified_path.exists():
            unified_df = pd.read_csv(unified_path)
            
            # Only load admin data if needed for region count
            admin_df = None
            admin_path = processed_dir / "admin_data.csv"
            if admin_path.exists():
                admin_df = pd.read_csv(admin_path, usecols=['ADM2_PCODE'])
            
            return unified_df, admin_df
        else:
            st.warning("Unified dataset not found. Please run the data processing script first.")
            return None, None
        
    except Exception as e:
        st.error(f"Error loading unified dataset: {e}")
        return None, None

# Add a cached function for deterministic sampling of wealth points
@st.cache_data
def get_sampled_wealth_points(wealth_features, max_points):
    import random
    if len(wealth_features) > max_points:
        random.seed(max_points)
        return random.sample(wealth_features, max_points)
    else:
        return wealth_features

# Load the unified dataset
with st.spinner("Loading unified dataset..."):
    unified_df, admin_df = load_unified_dataset()

# Load wealth_points_df from processed_data/wealth_points_with_region.csv
wealth_points_path = processed_dir / "wealth_points_with_region.csv"
if wealth_points_path.exists():
    wealth_points_df = pd.read_csv(wealth_points_path)
else:
    wealth_points_df = None
    st.warning("Wealth points file not found: processed_data/wealth_points_with_region.csv")

# Load admin_gdf from Datasets/geoBoundaries-MAR-ADM2.geojson
admin_gdf_path = dataset_dir / "geoBoundaries-MAR-ADM2.geojson"
if admin_gdf_path.exists():
    admin_gdf = gpd.read_file(admin_gdf_path)
else:
    admin_gdf = None
    st.warning("Admin boundaries file not found: Datasets/geoBoundaries-MAR-ADM2.geojson")

# Check if data loaded successfully
if unified_df is None:
    st.error("Failed to load unified dataset. Please run the data processing script first.")
    
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

# Create tabs for main content sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Overview", "Correlation Analysis", "Geographic Visualization", "Seasonal Analysis", "Data Dictionary", "How Did We Join the Data?"])

# Select rainfall metric for analysis (used across tabs)
rainfall_metrics_list = [col for col in unified_df.columns 
                       if (col.startswith('rfh_') or col.startswith('r1h_') or col.startswith('r3h_'))
                       and not col.endswith('_cv')]

# Find wealth metrics (used across tabs)
wealth_metrics_list = [col for col in unified_df.columns if 'rwi' in col.lower()]

# Check if we have both rainfall and wealth data
if not rainfall_metrics_list:
    st.warning("No rainfall metrics found in the unified dataset.")
    rainfall_metrics_list = ["no_rainfall_data"]

if not wealth_metrics_list:
    st.warning("No wealth metrics found in the unified dataset.")
    wealth_metrics_list = ["no_wealth_data"]

# Tab 1: Overview
with tab1:
    st.header("Unified Dataset Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Regions", len(unified_df))
        
        # Calculate coverage
        if admin_df is not None:
            total_regions = admin_df['ADM2_PCODE'].nunique()
        else:
            total_regions = "Unknown"
        st.metric("Region Coverage", f"{len(unified_df)} / {total_regions}")
        
    with col2:
        # Check which key columns are available
        rainfall_present = any(col.startswith('rfh_') for col in unified_df.columns)
        wealth_present = any(col.startswith('rwi_') for col in unified_df.columns)
        
        st.metric("Rainfall Data Available", "✅ Yes" if rainfall_present else "❌ No")
        st.metric("Wealth Data Available", "✅ Yes" if wealth_present else "❌ No")

    # Display the unified dataset
    with st.expander("View unified dataset"):
        st.dataframe(unified_df)
        
    # Download the unified dataset
    st.subheader("Export Unified Dataset")
    csv = unified_df.to_csv(index=False)
    st.download_button(
        label="Download Unified Dataset (CSV)",
        data=csv,
        file_name="morocco_wealth_rainfall_unified.csv",
        mime="text/csv"
    )

# Tab 2: Correlation Analysis
with tab2:
    st.header("Correlation Analysis")
    
    st.subheader("Select Metrics for Analysis")
    col1, col2 = st.columns(2)

    with col1:
        selected_rainfall = st.selectbox(
            "Select Rainfall Metric",
            rainfall_metrics_list,
            format_func=lambda x: f"{x} (mm)" if x != "no_rainfall_data" else "No rainfall data available",
            key="correlation_rainfall_selector"
        )
        
    with col2:
        selected_wealth = st.selectbox(
            "Select Wealth Metric",
            wealth_metrics_list,
            index=wealth_metrics_list.index('rwi_mean') if 'rwi_mean' in wealth_metrics_list else 0,
            format_func=lambda x: x if x != "no_wealth_data" else "No wealth data available",
            key="correlation_wealth_selector"
        )

    # Scatter plot of rainfall vs wealth
    st.subheader("Rainfall vs Wealth Relationship")

    # Only show correlation analysis if we have valid data
    if selected_rainfall != "no_rainfall_data" and selected_wealth != "no_wealth_data":
        # Create scatter plot
        fig = px.scatter(
            unified_df,
            x=selected_rainfall,
            y=selected_wealth,
            trendline="ols",
            hover_name="ADM2_EN" if "ADM2_EN" in unified_df.columns else "ADM2_PCODE",
            labels={
                selected_rainfall: f"{selected_rainfall} (mm)",
                selected_wealth: selected_wealth
            },
            title=f"Relationship between {selected_rainfall} and {selected_wealth}"
        )

        # Customize plot
        fig.update_layout(
            xaxis_title=f"{selected_rainfall} (mm)",
            yaxis_title=selected_wealth,
            height=600
        )

        st.plotly_chart(fig, use_container_width=True)

        # Calculate correlation coefficient
        correlation = unified_df[[selected_rainfall, selected_wealth]].corr().iloc[0, 1]
        st.metric("Correlation Coefficient", round(correlation, 3))

        # Interpret correlation
        if abs(correlation) < 0.2:
            st.info("This indicates a very weak relationship between rainfall and wealth.")
        elif abs(correlation) < 0.4:
            st.info("This indicates a weak relationship between rainfall and wealth.")
        elif abs(correlation) < 0.6:
            st.info("This indicates a moderate relationship between rainfall and wealth.")
        elif abs(correlation) < 0.8:
            st.info("This indicates a strong relationship between rainfall and wealth.")
        else:
            st.info("This indicates a very strong relationship between rainfall and wealth.")
    else:
        st.warning("Cannot display correlation analysis - missing either rainfall or wealth data.")

    # Correlation matrix
    st.subheader("Correlation Matrix")

    # Only show this section if we have valid data to correlate
    if rainfall_metrics_list and rainfall_metrics_list[0] != "no_rainfall_data" and wealth_metrics_list and wealth_metrics_list[0] != "no_wealth_data":
        # Select columns for correlation matrix
        rainfall_cols = st.multiselect(
            "Select Rainfall Metrics",
            rainfall_metrics_list,
            default=rainfall_metrics_list[:3] if len(rainfall_metrics_list) >= 3 else rainfall_metrics_list,
            key="correlation_matrix_rainfall"
        )

        wealth_cols = st.multiselect(
            "Select Wealth Metrics",
            wealth_metrics_list,
            default=['rwi_mean'] if 'rwi_mean' in wealth_metrics_list else wealth_metrics_list[:1],
            key="correlation_matrix_wealth"
        )

        # Create correlation matrix if selections are made
        if rainfall_cols and wealth_cols:
            corr_cols = rainfall_cols + wealth_cols
            corr_matrix = unified_df[corr_cols].corr()
            
            # Create heatmap
            fig = px.imshow(
                corr_matrix,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                zmin=-1, zmax=1,
                title="Correlation Matrix"
            )
            
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one rainfall metric and one wealth metric.")
    else:
        st.warning("Cannot create correlation matrix - missing either rainfall or wealth metrics.")

# Tab 3: Geographic Visualization
with tab3:
    st.header("Geographic Visualization")
    
    # Combined Rainfall and Wealth Map
    st.subheader("Combined Rainfall and Wealth Map")

    # Load necessary data
    metadata, wealth_features, region_data, region_stats = load_map_data()
    
    if metadata and region_data:
        # Allow user to select rainfall metric to color the regions
        available_rainfall_cols = metadata['rainfall_metrics']
        
        if available_rainfall_cols:
            selected_rainfall_metric = st.selectbox(
                "Select Rainfall Metric to Color Regions",
                available_rainfall_cols,
                format_func=lambda x: f"{x} (mm)",
                key="rainfall_region_selector"
            )
            
            if selected_rainfall_metric in region_data:
                # Create base map
                center = [31.7917, -7.0926]
                m = folium.Map(location=center, zoom_start=6, tiles='CartoDB positron')
                
                # Get the data for the selected metric
                metric_data = region_data[selected_rainfall_metric]
                
                # Create colormap for rainfall using pre-calculated values
                rainfall_colormap = LinearColormap(
                    colors=['#ffffd9', '#edf8b1', '#7fcdbb', '#41b6c4', '#1d91c0', '#225ea8', '#0c2c84'],
                    vmin=metric_data['vmin'],
                    vmax=metric_data['vmax'],
                    caption=f'Rainfall ({selected_rainfall_metric}) (mm)'
                )
                
                # Add choropleth layer using pre-calculated region data
                folium.GeoJson(
                    metric_data['geo_data'],
                    style_function=lambda feature: {
                        'fillColor': rainfall_colormap(feature['properties'][selected_rainfall_metric]) 
                                    if feature['properties'][selected_rainfall_metric] is not None else 'gray',
                        'color': 'black',
                        'weight': 1,
                        'fillOpacity': 0.6
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['ADM2_PCODE', 'ADM2_EN', selected_rainfall_metric, 'avg_rwi'],
                        aliases=['Region Code:', 'Region Name:', f'Rainfall ({selected_rainfall_metric}) (mm):', 'Avg RWI:'],
                        localize=True,
                        labels=True,
                        sticky=True
                    )
                ).add_to(m)
                
                # Add the rainfall colormap to the map
                rainfall_colormap.add_to(m)
                
                # Remove the slider and use a fixed sample size for wealth points
                FIXED_SAMPLE_SIZE = 1000
                if wealth_features:
                    wealth_features_sample = get_sampled_wealth_points(wealth_features, FIXED_SAMPLE_SIZE)
                    
                    marker_cluster = MarkerCluster().add_to(m)
                    batch_size = 500
                    for i in range(0, len(wealth_features_sample), batch_size):
                        batch = wealth_features_sample[i:i + batch_size]
                        for feature in batch:
                            folium.CircleMarker(
                                location=[feature['lat'], feature['lon']],
                                radius=3,
                                color=feature['color'],
                                fill=True,
                                fill_opacity=0.7,
                                popup=f"RWI: {feature['rwi']:.2f}"
                            ).add_to(marker_cluster)
                    
                    st.write(f"Combined map: Regions colored by rainfall, points representing wealth data (showing {len(wealth_features_sample)} points)")
                    st.caption("Note: Zooming in will not increase the number of points, but you can increase the slider value for more detail.")
                    st_folium(m, width=1000, height=700)
                    
                    # Add explanation about the map
                    st.info("""
                    **About this map:**
                    
                    **Region colors**: The regions are colored according to rainfall values, with darker blue indicating higher rainfall.
                    
                    **Colored points**: The individual points represent wealth data:
                    - Green: RWI > 1.0 (higher wealth)
                    - Blue: RWI between 0 and 1.0 (moderate wealth)
                    - Orange: RWI between -1.0 and 0 (moderate poverty)
                    - Red: RWI < -1.0 (higher poverty)
                    
                    **Numbers on map**: The large numbers shown on the map represent the count of wealth data points clustered in that area, not the RWI values themselves.
                    
                    You can zoom in to see smaller clusters and individual wealth data points.
                    """)
                    
                    # Add statistics about the selected metric
                    if region_stats and selected_rainfall_metric in region_stats:
                        stats = region_stats[selected_rainfall_metric]
                        st.subheader(f"Statistics for {selected_rainfall_metric}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Mean", f"{stats['mean']:.2f} mm")
                        with col2:
                            st.metric("Median", f"{stats['median']:.2f} mm")
                        with col3:
                            st.metric("Std Dev", f"{stats['std']:.2f} mm")
                        
                        st.caption(f"Last updated: {metadata['last_updated']}")
                else:
                    st.error("Required wealth features not found. Please run the preprocessing script first.")
            else:
                st.error(f"Data for metric {selected_rainfall_metric} not found in the preprocessed data.")
        else:
            st.error("No rainfall metrics found in the preprocessed data.")
    else:
        st.error("Required map data not found. Please run the preprocessing script first.")

    # Regional Distribution
    st.subheader("Regional Distribution")
    
    try:
        # Create a simple plot for individual metrics
        # Merge with region names if available
        if 'ADM2_EN' in unified_df.columns:
            plot_df = unified_df.copy()
            plot_df['region'] = plot_df['ADM2_EN']
        else:
            plot_df = unified_df.copy()
            plot_df['region'] = plot_df['ADM2_PCODE']
        
        # Select metric to visualize
        all_metrics = rainfall_metrics_list + wealth_metrics_list
        selected_viz_metric = st.selectbox(
            "Select Metric to Visualize",
            all_metrics,
            index=all_metrics.index('rwi_mean') if 'rwi_mean' in all_metrics else 0,
            key="regional_bar_metric"
        )
        
        # Create bar chart visualization 
        fig = px.bar(
            plot_df.sort_values(selected_viz_metric, ascending=False),
            x='region',
            y=selected_viz_metric,
            title=f"Distribution of {selected_viz_metric} by Region",
            color=selected_viz_metric,
            color_continuous_scale='Viridis' if selected_viz_metric.startswith('rwi') else 'Blues'
        )
        
        fig.update_layout(
            xaxis_title="Region",
            yaxis_title=selected_viz_metric,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # Add RWI distribution chart
        if wealth_points_df is not None and 'ADM2_PCODE' in wealth_points_df.columns:
            st.subheader("Relative Wealth Index Distribution by Region")
            
            # Calculate average RWI per region
            avg_rwi = wealth_points_df.groupby('ADM2_PCODE')['rwi'].mean().reset_index()
            avg_rwi.columns = ['ADM2_PCODE', 'avg_rwi']
            
            # Get region names
            if 'ADM2_EN' in unified_df.columns:
                region_names = unified_df[['ADM2_PCODE', 'ADM2_EN']].drop_duplicates()
                avg_rwi = avg_rwi.merge(region_names, on='ADM2_PCODE', how='left')
                region_col = 'ADM2_EN'
            else:
                region_col = 'ADM2_PCODE'
            
            # Create bar chart for RWI
            fig_rwi = px.bar(
                avg_rwi.sort_values('avg_rwi', ascending=False),
                x=region_col,
                y='avg_rwi',
                title="Average Relative Wealth Index by Region",
                color='avg_rwi',
                color_continuous_scale='RdYlBu',  # Red-Yellow-Blue scale: Red for negative, Blue for positive
                labels={'avg_rwi': 'Average RWI', region_col: 'Region'}
            )
            
            fig_rwi.update_layout(
                xaxis_title="Region",
                yaxis_title="Average Relative Wealth Index",
                height=600,
                yaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='black')  # Add zero line for reference
            )
            
            st.plotly_chart(fig_rwi, use_container_width=True)
            
            # Add explanation of RWI values
            st.info("""
            **Understanding RWI Values:**
            - Positive values (blue) indicate regions with above-average wealth
            - Negative values (red) indicate regions with below-average wealth
            - The zero line represents the average wealth level
            """)
        
    except Exception as e:
        st.error(f"Error creating visualization: {e}")

# Tab 4: Seasonal Analysis
with tab4:
    st.header("Seasonal Rainfall-Wealth Relationship")

    # Check if we have seasonal data
    seasonal_cols = [col for col in unified_df.columns if 'winter' in col or 'spring' in col or 'summer' in col or 'autumn' in col]

    if seasonal_cols and 'rwi_mean' in unified_df.columns:
        # Create seasonal comparison
        seasonal_df = pd.DataFrame()
        
        for season in ['winter', 'spring', 'summer', 'autumn']:
            season_col = next((col for col in unified_df.columns if season in col), None)
            if season_col:
                # Calculate correlation with wealth
                corr = unified_df[[season_col, 'rwi_mean']].corr().iloc[0, 1]
                seasonal_df = pd.concat([seasonal_df, pd.DataFrame({
                    'Season': [season.capitalize()],
                    'Correlation': [corr],
                    'Rainfall Metric': [season_col]
                })], ignore_index=True)
        
        # Create bar chart
        fig = px.bar(
            seasonal_df,
            x='Season',
            y='Correlation',
            color='Correlation',
            color_continuous_scale='RdBu_r',
            title="Seasonal Correlation between Rainfall and Wealth",
            labels={'Correlation': 'Correlation with RWI Mean'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Find the season with the strongest relationship
        strongest_season = seasonal_df.loc[seasonal_df['Correlation'].abs().idxmax()]
        st.markdown(f"**Strongest relationship:** {strongest_season['Season']} with correlation of {strongest_season['Correlation']:.3f}")
        
        # Create scatter plot for the strongest season
        strongest_col = strongest_season['Rainfall Metric']
        
        fig = px.scatter(
            unified_df,
            x=strongest_col,
            y='rwi_mean',
            trendline="ols",
            hover_name="ADM2_EN" if "ADM2_EN" in unified_df.columns else "ADM2_PCODE",
            title=f"Relationship between {strongest_season['Season']} Rainfall and Wealth"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    elif 'seasonal_variability' in unified_df.columns and 'rwi_mean' in unified_df.columns:
        # Analyze relationship between seasonal variability and wealth
        fig = px.scatter(
            unified_df,
            x='seasonal_variability',
            y='rwi_mean',
            trendline="ols",
            hover_name="ADM2_EN" if "ADM2_EN" in unified_df.columns else "ADM2_PCODE",
            title="Relationship between Seasonal Rainfall Variability and Wealth"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        variability_corr = unified_df[['seasonal_variability', 'rwi_mean']].corr().iloc[0, 1]
        st.metric("Correlation between Seasonal Variability and Wealth", round(variability_corr, 3))
        
    else:
        st.warning("Seasonal data not available for analysis.")

# Tab 5: Data Dictionary
with tab5:
    st.header("Data Dictionary")
    st.markdown("""
    ### Unified Dataset Columns
    
    #### Identifiers and Region Information
    | Column | Description |
    | ------ | ----------- |
    | ADM2_PCODE | Region code |
    | ADM2_EN | Region name (English) |
    | ADM1_EN | Province name (English) |
    | AREA_SQKM | Area in square kilometers |
    
    #### Rainfall Metrics
    | Column | Description |
    | ------ | ----------- |
    | rfh_avg_mean | Average rainfall (mm) |
    | rfh_avg_median | Median rainfall (mm) |
    | rfh_avg_std | Standard deviation of rainfall (mm) |
    | rfh_avg_cv | Coefficient of variation for rainfall |
    | rfh_avg_winter | Average winter rainfall (mm) |
    | rfh_avg_spring | Average spring rainfall (mm) |
    | rfh_avg_summer | Average summer rainfall (mm) |
    | rfh_avg_autumn | Average autumn rainfall (mm) |
    | seasonal_variability | Measure of seasonal rainfall variability |
    
    #### Wealth Metrics
    | Column | Description |
    | ------ | ----------- |
    | rwi_mean | Mean relative wealth index |
    | rwi_median | Median relative wealth index |
    | rwi_std | Standard deviation of relative wealth index |
    | rwi_count | Number of wealth data points in region |
    """)

# Tab 6: How Did We Join the Data?
with tab6:
    st.header("How Did We Join the Data?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Step 1: What we start with")
        st.markdown("""
        🗺️ **Region Map (GeoJSON)**
        - Like a paper map of Morocco
        - Each region is a shape with boundaries
        
        📍 **Wealth Points (CSV)**
        - Each point has latitude & longitude
        - Each point has an RWI score
        """)
        
    with col2:
        st.subheader("Step 2: How we join them")
        st.markdown("""
        1. **Turn coordinates into points**
           - Convert each lat/long into a pin on the map
        
        2. **Check which region contains each point**
           - Like dropping each pin and seeing which region it lands in
           
        3. **Assign region code to each point**
           - Write down which region each pin landed in
           - Some pins might land outside all regions
        """)
    
    st.subheader("Step 3: Results")
    st.markdown("""
    After joining, for each region we:
    1. Count how many wealth points fell inside it
    2. Calculate the average RWI of all points in that region
    
    From our data:
    - Total wealth points: 5,000
    - Points matched to regions: 4,971 (99.4%)
    - Points outside any region: 29 (0.6%)
    """)

    # Add a simplified map showing the process
    try:
        if wealth_points_df is not None and admin_gdf is not None:
            st.subheader("Visual Example")
            
            # Create a simple map centered on Morocco
            m = folium.Map(
                location=[31.7917, -7.0926],
                zoom_start=6,
                tiles='CartoDB positron'
            )
            
            # Add regions with light fill
            folium.GeoJson(
                admin_gdf,
                style_function=lambda x: {
                    'fillColor': 'lightblue',
                    'color': 'black',
                    'weight': 1,
                    'fillOpacity': 0.3
                }
            ).add_to(m)
            
            # Add first 500 points as red dots for visualization
            for _, row in wealth_points_df.head(500).iterrows():
                folium.CircleMarker(
                    location=[row['latitude'], row['longitude']],
                    radius=3,
                    color='red',
                    fill=True,
                    popup=f"RWI: {row['rwi']:.3f}"
                ).add_to(m)
            
            # Display the map
            st_folium(m, width="100%", height=600, returned_objects=[])
            
            st.caption("🔴 Red dots: Sample of wealth data points")
            st.caption("🟦 Blue areas: Region boundaries")
            st.caption("Hover over points to see their RWI value")

    except Exception as e:
        st.error("Error creating visualization. Please check if the required data is loaded correctly.")

    st.markdown("""
    Think of the red dots as wealth data points, and the blue areas as regions. 
    We group the points by the region they fall into. This process is called a **spatial join**.
    """) 