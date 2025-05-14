"""
Preprocess map data and save it locally for faster map rendering.
This script should be run after data processing and before running the Streamlit app.
"""

import pandas as pd
import geopandas as gpd
import json
from pathlib import Path
import numpy as np

def prepare_wealth_features(wealth_points_df):
    """Pre-calculate wealth point features and their styling"""
    wealth_features = []
    
    if wealth_points_df is not None:
        for _, row in wealth_points_df.iterrows():
            # Set color based on RWI value
            if row['rwi'] < -1.0:
                color = 'red'
            elif row['rwi'] < 0:
                color = 'orange'
            elif row['rwi'] < 1.0:
                color = 'blue'
            else:
                color = 'green'
                
            wealth_features.append({
                'lat': float(row['latitude']),  # Convert to float for JSON serialization
                'lon': float(row['longitude']),
                'color': color,
                'rwi': float(row['rwi'])
            })
    
    return wealth_features

def prepare_region_data(admin_gdf, unified_df, rainfall_metrics):
    """Pre-calculate region features and rainfall data"""
    region_data = {}
    
    for metric in rainfall_metrics:
        if admin_gdf is not None and unified_df is not None:
            # Merge rainfall data with admin boundaries
            merged_gdf = admin_gdf.merge(
                unified_df[['ADM2_PCODE', metric, 'avg_rwi']],  # Include avg_rwi for tooltips
                on='ADM2_PCODE',
                how='left'
            )
            
            # Calculate rainfall ranges
            vmin = float(merged_gdf[metric].min())
            vmax = float(merged_gdf[metric].max())
            
            # Convert to GeoJSON with properties
            geo_data = merged_gdf.__geo_interface__
            
            # Store the data for this metric
            region_data[metric] = {
                'geo_data': geo_data,
                'vmin': vmin,
                'vmax': vmax
            }
    
    return region_data

def prepare_region_statistics(unified_df, rainfall_metrics):
    """Pre-calculate region statistics for faster loading"""
    stats = {}
    
    for metric in rainfall_metrics:
        stats[metric] = {
            'mean': float(unified_df[metric].mean()),
            'median': float(unified_df[metric].median()),
            'std': float(unified_df[metric].std()),
            'quartiles': [
                float(unified_df[metric].quantile(0.25)),
                float(unified_df[metric].quantile(0.5)),
                float(unified_df[metric].quantile(0.75))
            ]
        }
    
    return stats

def main():
    # Get project root directory
    project_root = Path(__file__).parents[2]
    processed_dir = project_root / "processed_data"
    map_data_dir = processed_dir / "map_data"
    
    # Create map_data directory if it doesn't exist
    map_data_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    
    # Load required datasets
    try:
        wealth_points_df = pd.read_csv(processed_dir / "wealth_points_with_region.csv")
        admin_gdf = gpd.read_file(processed_dir / "simplified_boundaries.geojson")
        unified_df = pd.read_csv(processed_dir / "unified_dataset.csv")
        
        # Calculate average RWI per region if not already present
        if 'avg_rwi' not in unified_df.columns and wealth_points_df is not None:
            avg_rwi = wealth_points_df.groupby('ADM2_PCODE')['rwi'].mean().reset_index()
            avg_rwi.columns = ['ADM2_PCODE', 'avg_rwi']
            unified_df = unified_df.merge(avg_rwi, on='ADM2_PCODE', how='left')
        
        # Get rainfall metrics
        rainfall_metrics = [col for col in unified_df.columns 
                          if (col.startswith('rfh_') or col.startswith('r1h_') or col.startswith('r3h_'))
                          and not col.endswith('_cv')]
        
        print("Preparing wealth features...")
        wealth_features = prepare_wealth_features(wealth_points_df)
        
        print("Preparing region data...")
        region_data = prepare_region_data(admin_gdf, unified_df, rainfall_metrics)
        
        print("Calculating region statistics...")
        region_stats = prepare_region_statistics(unified_df, rainfall_metrics)
        
        # Save wealth features
        print("Saving wealth features...")
        with open(map_data_dir / "wealth_features.json", 'w') as f:
            json.dump(wealth_features, f)
        
        # Save region data for each metric
        print("Saving region data...")
        with open(map_data_dir / "region_data.json", 'w') as f:
            json.dump(region_data, f)
        
        # Save region statistics
        print("Saving region statistics...")
        with open(map_data_dir / "region_stats.json", 'w') as f:
            json.dump(region_stats, f)
        
        # Save available metrics and other metadata
        print("Saving metadata...")
        metadata = {
            'rainfall_metrics': rainfall_metrics,
            'point_count': len(wealth_features),
            'region_count': len(admin_gdf),
            'last_updated': pd.Timestamp.now().isoformat()
        }
        with open(map_data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f)
        
        print("Successfully preprocessed and saved map data!")
        print(f"\nSummary:")
        print(f"- Processed {len(wealth_features)} wealth points")
        print(f"- Generated data for {len(rainfall_metrics)} rainfall metrics")
        print(f"- Covered {len(admin_gdf)} administrative regions")
        print(f"\nOutput files saved to: {map_data_dir}")
        
    except Exception as e:
        print(f"Error preprocessing map data: {e}")
        raise

if __name__ == "__main__":
    main() 