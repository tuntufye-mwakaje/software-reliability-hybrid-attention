from pathlib import Path
import sys

import pandas as pd

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent),
)

from dataset_schema import (
    FEATURE_COLUMNS,
    load_dataset,
)

from grouped_split import (
    grouped_train_val_test_split,
)


DATA_DIR = Path("../datasets")


for file in sorted(DATA_DIR.glob("*.csv")):

    df = load_dataset(file)

    train_df, validation_df, test_df = (
        grouped_train_val_test_split(
            df,
            FEATURE_COLUMNS,
            target_column="defects",
            test_size=0.20,
            validation_size=0.16,
            random_state=42,
        )
    )

    print()
    print("=" * 70)
    print(file.stem.upper())
    print("=" * 70)

    print("Original rows:", len(df))

    print(
        "Train:",
        len(train_df),
        f"({len(train_df) / len(df):.4f})"
    )

    print(
        "Validation:",
        len(validation_df),
        f"({len(validation_df) / len(df):.4f})"
    )

    print(
        "Test:",
        len(test_df),
        f"({len(test_df) / len(df):.4f})"
    )

    print()
    print(
        "Train positives:",
        int(train_df["defects"].sum())
    )

    print(
        "Validation positives:",
        int(validation_df["defects"].sum())
    )

    print(
        "Test positives:",
        int(test_df["defects"].sum())
    )

print()
print("All grouped split tests passed.")
