from pathlib import Path
import pandas as pd

PROCESSED_DIR = Path("data/processed")
OUTPUT_PATH = PROCESSED_DIR / "train_processed.parquet"

files = sorted(PROCESSED_DIR.glob("train_period_*_processed.parquet"))

print('-' * 60)
print(f"COMBINING PROCESSED DATA")
print('-' * 60)

print(f"Found {len(files)} processed files")

if len(files) != 10:
    raise ValueError(f"Expected 10 processed files, but found {len(files)}")

dfs = []

for file in files:
    print(f"Read: {file.name}")
    df = pd.read_parquet(file)
    print(f"Rows: {len(df)}")

    dfs.append(df)

print("Combine datasets ...")

train_df = pd.concat(dfs , ignore_index=True)

train_df.to_parquet(OUTPUT_PATH , index=True)

print('\n' + '-' * 60)
print("COMBINATION COMPLETED")
print('-' * 60)

print(f"Total rows    : {len(train_df):,}")
print(f"Total columns : {len(train_df.columns)}")

print(f"\nSaved to:")
print(OUTPUT_PATH)