"""
Data cleaning and preprocessing functions for rainfall and other raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def load_rainfall_data(file_path):
    """Load and clean rainfall data"""
    try:
        # Read CSV with low_memory=False to handle mixed types
        df = pd.read_csv(file_path, low_memory=False)
        
        # Remove metadata rows (starting with #)
        df = df[~df['date'].astype(str).str.startswith('#')].reset_index(drop=True)
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
        
        # Ensure rainfall columns are numeric
        rainfall_cols = [col for col in df.columns if col.startswith('r') and col not in ['rfq', 'r1q', 'r3q']]
        for col in rainfall_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle any missing values
        df = df.fillna(method='ffill')  # Forward fill missing values
        
        return df
    except Exception as e:
        print(f"Error loading rainfall data: {e}")
        return None

def calculate_rainfall_metrics(rainfall_df):
    """Calculate various rainfall metrics by region"""
    metrics = []
    
    # Get rainfall columns (excluding quality flags)
    rainfall_cols = [col for col in rainfall_df.columns 
                    if col.startswith('r') and col not in ['rfq', 'r1q', 'r3q']]
    
    if not rainfall_cols:
        print("No rainfall columns found in the dataset")
        return pd.DataFrame()
    
    for region in rainfall_df['ADM2_PCODE'].unique():
        region_data = rainfall_df[rainfall_df['ADM2_PCODE'] == region]
        
        # Calculate metrics for each rainfall column
        region_metrics = {'ADM2_PCODE': region}
        
        for col in rainfall_cols:
            prefix = col.split('_')[0] if '_' in col else col  # Get prefix (rfh, r1h, r3h)
            
            # Calculate basic statistics
            mean_rainfall = region_data[col].mean()
            median_rainfall = region_data[col].median()
            std_rainfall = region_data[col].std()
            cv_rainfall = std_rainfall / mean_rainfall if mean_rainfall > 0 else 0
            
            # Add metrics with appropriate prefix
            region_metrics[f'{prefix}_mean'] = mean_rainfall
            region_metrics[f'{prefix}_median'] = median_rainfall
            region_metrics[f'{prefix}_std'] = std_rainfall
            region_metrics[f'{prefix}_cv'] = cv_rainfall
            
            # Calculate seasonal averages
            region_data['month'] = region_data['date'].dt.month
            seasonal_data = {
                'winter': region_data[region_data['month'].isin([12, 1, 2])][col].mean(),
                'spring': region_data[region_data['month'].isin([3, 4, 5])][col].mean(),
                'summer': region_data[region_data['month'].isin([6, 7, 8])][col].mean(),
                'autumn': region_data[region_data['month'].isin([9, 10, 11])][col].mean()
            }
            
            # Add seasonal metrics with appropriate prefix
            region_metrics[f'{prefix}_winter'] = seasonal_data['winter']
            region_metrics[f'{prefix}_spring'] = seasonal_data['spring']
            region_metrics[f'{prefix}_summer'] = seasonal_data['summer']
            region_metrics[f'{prefix}_autumn'] = seasonal_data['autumn']
        
        metrics.append(region_metrics)
    
    return pd.DataFrame(metrics)

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # For numeric columns, fill missing values with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    return df

def process_raw_data():
    """Process all raw data files"""
    # Get project root directory
    project_root = Path(__file__).parents[2]
    dataset_dir = project_root / "Datasets"
    processed_dir = project_root / "processed_data"
    
    # Create processed_data directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process rainfall data
    print("Processing rainfall data...")
    rainfall_path = dataset_dir / "mar-rainfall-adm2-full.csv"
    if rainfall_path.exists():
        rainfall_df = load_rainfall_data(str(rainfall_path))
        if rainfall_df is not None:
            # Calculate rainfall metrics
            rainfall_metrics = calculate_rainfall_metrics(rainfall_df)
            
            # Save processed data
            rainfall_df.to_csv(processed_dir / "cleaned_rainfall_data.csv", index=False)
            rainfall_metrics.to_csv(processed_dir / "rainfall_metrics.csv", index=False)
            print("✓ Rainfall data processed and saved")
        else:
            print("✗ Failed to process rainfall data")
    else:
        print(f"✗ Rainfall data file not found at {rainfall_path}")
    
    # Process administrative data
    print("\nProcessing administrative data...")
    admin_path = dataset_dir / "mar_adm2.csv"
    if admin_path.exists():
        try:
            admin_df = pd.read_csv(admin_path)
            admin_df = handle_missing_values(admin_df)
            admin_df.to_csv(processed_dir / "admin_data.csv", index=False)
            print("✓ Administrative data processed and saved")
        except Exception as e:
            print(f"✗ Error processing administrative data: {e}")
    else:
        print(f"✗ Administrative data file not found at {admin_path}")
    
    print("\nRaw data processing completed!") 