"""
Main script to process all data and prepare it for the Streamlit app.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def main():
    print("Starting data processing pipeline...")
    
    try:
        # Run data processing steps
        print("\n1. Processing raw data...")
        from src.data_processing.cleaning import process_raw_data
        process_raw_data()
        
        print("\n2. Processing geospatial data...")
        from src.data_processing.geospatial import process_geospatial_data
        process_geospatial_data()
        
        print("\n3. Creating unified dataset...")
        from src.data_processing.integration import create_unified_dataset
        create_unified_dataset()
        
        print("\n4. Preparing map data...")
        from src.data_processing.prepare_map_data import main as prepare_map_data
        prepare_map_data()
        
        print("\nAll data processing completed successfully!")
        print("\nSummary of processed data:")
        
        # Check processed files
        processed_dir = project_root / "processed_data"
        map_data_dir = processed_dir / "map_data"
        
        processed_files = {
            'Rainfall Data': processed_dir / "cleaned_rainfall_data.csv",
            'Rainfall Metrics': processed_dir / "rainfall_metrics.csv",
            'Admin Data': processed_dir / "admin_data.csv",
            'Wealth Points': processed_dir / "wealth_points_with_region.csv",
            'Unified Dataset': processed_dir / "unified_dataset.csv",
            'Map Data': {
                'Metadata': map_data_dir / "metadata.json",
                'Wealth Features': map_data_dir / "wealth_features.json",
                'Region Data': map_data_dir / "region_data.json",
                'Region Statistics': map_data_dir / "region_stats.json"
            }
        }
        
        print("\nProcessed files status:")
        for name, path in processed_files.items():
            if isinstance(path, dict):
                print(f"\n{name}:")
                for subname, subpath in path.items():
                    status = "✓ Present" if subpath.exists() else "✗ Missing"
                    print(f"  - {subname}: {status}")
            else:
                status = "✓ Present" if path.exists() else "✗ Missing"
                print(f"- {name}: {status}")
        
        print("\nYou can now run the Streamlit app.")
        
    except Exception as e:
        print(f"\nError during data processing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 