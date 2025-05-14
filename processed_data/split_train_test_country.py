import pandas as pd
from pathlib import Path

# User can set this: number of periods for test set
N_TEST_PERIODS = 12  # Change as needed

input_path = Path(__file__).parent / 'rainfall_minimal_clean.csv'
output_dir = Path(__file__).parent

def main():
    print(f"Reading data from {input_path}...")
    df = pd.read_csv(input_path, parse_dates=['date'])
    print(f"Total records: {len(df)}")

    # Aggregate by mean rainfall for each date (country-level)
    country_df = df.groupby('date', as_index=False)['rfh_avg'].mean()
    country_df = country_df.sort_values('date')
    print(f"Aggregated to {len(country_df)} country-level time points.")

    if len(country_df) <= N_TEST_PERIODS:
        print(f"Not enough data to split: only {len(country_df)} periods.")
        return

    train = country_df.iloc[:-N_TEST_PERIODS]
    test = country_df.iloc[-N_TEST_PERIODS:]
    train_file = output_dir / 'country_train.csv'
    test_file = output_dir / 'country_test.csv'
    train.to_csv(train_file, index=False)
    test.to_csv(test_file, index=False)
    print(f"Train set: {len(train)} rows -> {train_file}")
    print(f"Test set: {len(test)} rows -> {test_file}")

if __name__ == "__main__":
    main() 