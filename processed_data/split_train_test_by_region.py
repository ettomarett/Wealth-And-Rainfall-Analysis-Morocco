import pandas as pd
from pathlib import Path

# Path to the cleaned minimal rainfall data
input_path = Path(__file__).parent / 'rainfall_minimal_clean.csv'
output_dir = Path(__file__).parent / 'region_splits'
output_dir.mkdir(exist_ok=True)

def main():
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['date'])
    print(f"Total records: {len(df)}")

    # Group by region
    for region, group in df.groupby('ADM2_PCODE'):
        group = group.sort_values('date')
        if len(group) > 24:
            train = group.iloc[:-12]
            test = group.iloc[-12:]
            train_file = output_dir / f"train_{region}.csv"
            test_file = output_dir / f"test_{region}.csv"
            train.to_csv(train_file, index=False)
            test.to_csv(test_file, index=False)
            print(f"Region {region}: train={len(train)}, test={len(test)} -> Saved to {train_file}, {test_file}")
        else:
            print(f"Region {region}: Not enough data to split (only {len(group)} records)")

if __name__ == "__main__":
    main() 