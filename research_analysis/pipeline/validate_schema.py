from pathlib import Path

from dataset_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_dataset,
)


DATA_DIR = Path("../datasets")


for file in sorted(DATA_DIR.glob("*.csv")):
    df = load_dataset(file)

    print()
    print("=" * 70)
    print(file.stem.upper())
    print("=" * 70)

    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Target:", TARGET_COLUMN)
    print("Positive:", int(df[TARGET_COLUMN].sum()))
    print("Negative:", int((~df[TARGET_COLUMN]).sum()))
    print("Missing:", int(df.isna().sum().sum()))

    print()
    print("Schema:")
    print(list(df.columns))

print()
print("All datasets passed canonical schema validation.")
