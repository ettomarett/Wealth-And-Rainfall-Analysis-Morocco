"""
Functions for handling geospatial data including GeoJSON files.
"""

import pandas as pd
import geopandas as gpd
import json
from pathlib import Path


def load_admin_boundaries(file_path):
    """Load and process administrative boundaries"""
    try:
        gdf = gpd.read_file(file_path)
        # Simplify geometries for better performance
        gdf['geometry'] = gdf.geometry.simplify(0.01)
        return gdf
    except Exception as e:
        print(f"Error loading admin boundaries: {e}")
        return None


def load_aggregated_wealth(filepath):
    """
    Load pre-aggregated wealth data by region from GeoJSON file.
    
    Parameters:
    filepath (str): Path to the adm2_with_rwi.geojson file
    
    Returns:
    gpd.GeoDataFrame: GeoDataFrame with administrative boundaries and wealth metrics
    """
    try:
        # Load GeoJSON file
        gdf = gpd.read_file(filepath)
        
        # Ensure we have wealth metrics
        required_cols = ['rwi_mean', 'rwi_median', 'rwi_std', 'rwi_count']
        missing_cols = [col for col in required_cols if col not in gdf.columns]
        
        if missing_cols:
            print(f"Warning: Missing wealth metrics columns: {', '.join(missing_cols)}")
        
        # Ensure CRS is set
        if gdf.crs is None:
            gdf.crs = "EPSG:4326"
            
        print(f"Loaded aggregated wealth data for {len(gdf)} regions")
        return gdf
    except Exception as e:
        print(f"Error loading aggregated wealth data: {e}")
        return None


def extract_wealth_metrics(gdf):
    """
    Extract wealth metrics from the GeoDataFrame into a pandas DataFrame.
    
    Parameters:
    gdf (gpd.GeoDataFrame): GeoDataFrame with wealth metrics
    
    Returns:
    pd.DataFrame: DataFrame with wealth metrics by region
    """
    # Identify wealth metric columns
    wealth_cols = [col for col in gdf.columns if col.startswith('rwi_')]
    
    # Extract ADM2_PCODE (or equivalent) and wealth metrics
    id_col = None
    for candidate in ['ADM2_PCODE', 'csv_pcode', 'admin_id', 'id']:
        if candidate in gdf.columns:
            id_col = candidate
            break
    
    if id_col is None:
        raise ValueError("Could not identify region ID column in GeoJSON")
    
    # Extract to DataFrame
    wealth_df = gdf[[id_col] + wealth_cols].copy()
    
    # Rename ID column if needed
    if id_col != 'ADM2_PCODE':
        wealth_df = wealth_df.rename(columns={id_col: 'ADM2_PCODE'})
    
    return wealth_df


def merge_with_rainfall(rainfall_metrics, wealth_metrics, admin_boundaries=None):
    """
    Merge rainfall metrics with wealth metrics and optionally with admin boundaries.
    
    Parameters:
    rainfall_metrics (pd.DataFrame): DataFrame with rainfall metrics by region
    wealth_metrics (pd.DataFrame): DataFrame with wealth metrics by region
    admin_boundaries (gpd.GeoDataFrame, optional): GeoDataFrame with admin boundaries
    
    Returns:
    pd.DataFrame or gpd.GeoDataFrame: DataFrame with merged metrics
    """
    # Merge rainfall and wealth metrics
    merged = rainfall_metrics.merge(wealth_metrics, on='ADM2_PCODE', how='outer')
    
    # If admin boundaries provided, add region names and other attributes
    if admin_boundaries is not None:
        # Identify ID column in admin boundaries
        id_col = None
        for candidate in ['ADM2_PCODE', 'csv_pcode', 'admin_id', 'id']:
            if candidate in admin_boundaries.columns:
                id_col = candidate
                break
        
        if id_col is None:
            print("Warning: Could not identify region ID column in admin boundaries")
            return merged
        
        # Identify name columns
        name_cols = []
        for candidate in ['ADM2_EN', 'name_en', 'name', 'ADM2_AR']:
            if candidate in admin_boundaries.columns:
                name_cols.append(candidate)
        
        # Add area column if available
        area_col = None
        for candidate in ['AREA_SQKM', 'area', 'area_sqkm']:
            if candidate in admin_boundaries.columns:
                area_col = candidate
                break
        
        # Columns to merge from admin boundaries
        admin_cols = [id_col] + name_cols
        if area_col:
            admin_cols.append(area_col)
        
        # Extract admin data to merge
        admin_data = admin_boundaries[admin_cols].copy()
        
        # Rename ID column if needed
        if id_col != 'ADM2_PCODE':
            admin_data = admin_data.rename(columns={id_col: 'ADM2_PCODE'})
        
        # Merge with admin data
        merged = merged.merge(admin_data, on='ADM2_PCODE', how='left')
    
    return merged 


def process_wealth_points(wealth_points_df, admin_gdf):
    """Process wealth points and assign them to regions"""
    try:
        # Convert wealth points to GeoDataFrame
        wealth_gdf = gpd.GeoDataFrame(
            wealth_points_df,
            geometry=gpd.points_from_xy(wealth_points_df.longitude, wealth_points_df.latitude),
            crs="EPSG:4326"
        )
        
        # Spatial join with admin boundaries
        wealth_with_region = gpd.sjoin(
            wealth_gdf,
            admin_gdf[['ADM2_PCODE', 'geometry']],
            how="left",
            op="within"
        )
        
        return wealth_with_region
        
    except Exception as e:
        print(f"Error processing wealth points: {e}")
        return None


def process_geospatial_data():
    """Process all geospatial data"""
    # Get project root directory
    project_root = Path(__file__).parents[2]
    dataset_dir = project_root / "Datasets"
    processed_dir = project_root / "processed_data"
    
    # Create processed_data directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process administrative boundaries
    print("Processing administrative boundaries...")
    boundaries_path = dataset_dir / "merged_adm2_data.geojson"
    if boundaries_path.exists():
        admin_gdf = load_admin_boundaries(str(boundaries_path))
        if admin_gdf is not None:
            # Save simplified boundaries
            admin_gdf.to_file(processed_dir / "simplified_boundaries.geojson", driver="GeoJSON")
            print("✓ Administrative boundaries processed and saved")
            
            # Process wealth points if available
            print("\nProcessing wealth points...")
            wealth_path = dataset_dir / "morocco_relative_wealth_index.csv"
            if wealth_path.exists():
                try:
                    # Load wealth points
                    wealth_points_df = pd.read_csv(wealth_path)
                    
                    # Process wealth points
                    wealth_with_region = process_wealth_points(wealth_points_df, admin_gdf)
                    
                    if wealth_with_region is not None:
                        # Save processed wealth points
                        wealth_with_region.to_csv(
                            processed_dir / "wealth_points_with_region.csv",
                            index=False
                        )
                        print("✓ Wealth points processed and saved")
                    else:
                        print("✗ Failed to process wealth points")
                        
                except Exception as e:
                    print(f"✗ Error processing wealth points: {e}")
            else:
                print(f"✗ Wealth points file not found at {wealth_path}")
        else:
            print("✗ Failed to process administrative boundaries")
    else:
        print(f"✗ Administrative boundaries file not found at {boundaries_path}")
    
    print("\nGeospatial data processing completed!") 