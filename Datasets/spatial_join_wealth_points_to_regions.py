import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# File paths with correct directory structure
wealth_points_path = 'processed_data/wealth_points_sample.csv'
regions_geojson_path = 'Datasets/merged_adm2_data.geojson'
output_path = 'processed_data/wealth_points_with_region.csv'

print("\n=== Loading Data ===")
# Load wealth points
wealth_df = pd.read_csv(wealth_points_path)
print(f"Loaded {len(wealth_df)} wealth points")

# Create geometry column from latitude and longitude
print("\n=== Creating Point Geometries ===")
geometry = [Point(xy) for xy in zip(wealth_df['longitude'], wealth_df['latitude'])]
wealth_gdf = gpd.GeoDataFrame(wealth_df, geometry=geometry, crs='EPSG:4326')
print("Sample of wealth points:")
print(wealth_gdf[['latitude', 'longitude', 'rwi']].head())

# Load region polygons
print("\n=== Loading Region Boundaries ===")
regions_gdf = gpd.read_file(regions_geojson_path)
print(f"Loaded {len(regions_gdf)} regions")
print("Sample of regions:")
print(regions_gdf[['ADM2_PCODE', 'shapeName']].head())

# Perform spatial join
print("\n=== Performing Spatial Join ===")
joined_gdf = gpd.sjoin(wealth_gdf, regions_gdf, how='left', predicate='within')

# Print join statistics
total_points = len(joined_gdf)
matched_points = joined_gdf['ADM2_PCODE'].notna().sum()
print(f"\nJoin Statistics:")
print(f"Total points: {total_points}")
print(f"Points matched to regions: {matched_points}")
print(f"Points outside regions: {total_points - matched_points}")

# Calculate average RWI per region
print("\n=== Average RWI by Region ===")
avg_rwi = joined_gdf.groupby(['ADM2_PCODE', 'shapeName'])['rwi'].agg(['mean', 'count']).reset_index()
print(avg_rwi.head())

# Keep only necessary columns from regions
joined_df = joined_gdf.drop(columns=['geometry', 'index_right'])

# Save to CSV
joined_df.to_csv(output_path, index=False)
print(f"\nSpatial join complete. Output saved to {output_path}") 