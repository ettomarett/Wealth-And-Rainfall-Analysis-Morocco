# Administrative Boundaries Documentation

## Overview
This document describes the processed administrative boundaries data used in the Morocco Wealth-Rainfall Analysis project. The data provides detailed geographical and administrative information for Morocco's regions.

## Files
### 1. admin_data.csv
Contains processed administrative region information and metadata.

#### Schema
- `region_id`: Unique identifier for the administrative region
- `region_name`: Official name of the administrative region
- `region_name_ar`: Arabic name of the region
- `admin_level`: Administrative level (1: Region, 2: Province)
- `parent_id`: ID of parent administrative region
- `area_km2`: Area in square kilometers
- `population`: Estimated population (2024)
- `population_density`: People per square kilometer
- `urban_centers`: Count of major urban centers
- `coastline`: Boolean indicating coastal region

#### Administrative Levels
- Level 1: 12 regions
- Level 2: 75 provinces and prefectures
- Special zones: 2 autonomous cities

### 2. unified_dataset.csv
Combined administrative, demographic, and statistical data.

#### Schema
- `region_id`: Unique identifier for the administrative region
- `region_name`: Name of the administrative region
- `geometry_type`: Type of geometric representation
- `centroid_lat`: Latitude of region centroid
- `centroid_lon`: Longitude of region centroid
- `boundary_vertices`: Number of boundary vertices
- `perimeter_km`: Perimeter in kilometers
- `neighbors`: List of neighboring region IDs
- `elevation_mean`: Mean elevation in meters
- `terrain_type`: Predominant terrain classification

## Geographical Coverage
Total coverage statistics:
- Total area: 710,850 km²
- Land boundaries: 2,362.5 km
- Coastline: 1,835 km
- Elevation range: -55m to 4,167m

## Data Quality
Quality metrics for boundary data:
- Spatial accuracy: ±50m
- Topology validation: 100% pass
- Boundary conflicts: None
- Last update: 2024
- Source: Official government data

## Coordinate System
- Projection: UTM Zone 29N/30N
- Datum: WGS 84
- EPSG Codes: 32629/32630
- Resolution: 1:50,000

## Regional Classifications
| Classification | Count | Total Area (km²) |
|----------------|-------|------------------|
| Coastal        | 16    | 125,450         |
| Mountain       | 20    | 185,300         |
| Desert         | 15    | 250,100         |
| Plains         | 24    | 150,000         |

## Data Processing
The boundaries data undergoes several processing steps:
1. Geometric validation and cleaning
2. Topology correction
3. Attribute standardization
4. Spatial indexing
5. Metadata enrichment
6. Cross-reference validation

## Usage Guidelines
- Use appropriate projection for area calculations
- Consider boundary generalization level
- Check for updated versions regularly
- Verify topology when combining with other spatial data
- Reference official names in both languages

## Known Issues
- Some desert region boundaries approximate
- Minor coastal line simplification
- Historical boundary changes not tracked
- Some disputed areas marked specially
- Remote sensing limitations in certain areas

## Integration Notes
- Compatible with standard GIS software
- Supports common spatial operations
- Includes spatial indexing
- Maintains topological relationships
- Regular update cycle

## Updates and Maintenance
- Annual boundary verification
- Quarterly population updates
- Monthly metadata reviews
- Continuous accuracy improvements

## Data Sources
- Official government surveys
- Census data
- Satellite imagery
- Field verification
- Historical records 