"""
Integration functions to create the unified dataset combining all processed data.
"""

import pandas as pd
from pathlib import Path

def create_unified_dataset():
    """Create unified dataset from processed data files"""
    # Get project root directory
    project_root = Path(__file__).parents[2]
    processed_dir = project_root / "processed_data"
    
    print("Creating unified dataset...")
    
    try:
        # Load processed datasets
        rainfall_metrics = pd.read_csv(processed_dir / "rainfall_metrics.csv")
        admin_data = pd.read_csv(processed_dir / "admin_data.csv")
        
        # Start with rainfall metrics
        unified_df = rainfall_metrics.copy()
        
        # Add administrative data
        if 'ADM2_PCODE' in admin_data.columns:
            # Identify columns to merge from admin data
            admin_cols = ['ADM2_PCODE']
            for col in ['ADM2_EN', 'ADM1_EN', 'AREA_SQKM']:
                if col in admin_data.columns:
                    admin_cols.append(col)
            
            unified_df = unified_df.merge(
                admin_data[admin_cols],
                on='ADM2_PCODE',
                how='left'
            )
        
        # Calculate wealth metrics from points if available
        wealth_points_path = processed_dir / "wealth_points_with_region.csv"
        if wealth_points_path.exists():
            wealth_points = pd.read_csv(wealth_points_path)
            
            # Calculate wealth metrics by region
            wealth_metrics = wealth_points.groupby('ADM2_PCODE').agg({
                'rwi': ['mean', 'median', 'std', 'count']
            }).reset_index()
            
            # Flatten column names
            wealth_metrics.columns = ['ADM2_PCODE', 'rwi_mean', 'rwi_median', 'rwi_std', 'rwi_count']
            
            # Add wealth metrics to unified dataset
            unified_df = unified_df.merge(
                wealth_metrics,
                on='ADM2_PCODE',
                how='left'
            )
        
        # Save unified dataset
        unified_df.to_csv(processed_dir / "unified_dataset.csv", index=False)
        print("✓ Unified dataset created and saved")
        
        # Print summary
        print(f"\nUnified dataset summary:")
        print(f"- Total regions: {len(unified_df)}")
        print(f"- Columns: {', '.join(unified_df.columns)}")
        
    except Exception as e:
        print(f"✗ Error creating unified dataset: {e}")
    
    print("\nData integration completed!") 