from pathlib import Path
import pandas as pd

from dataset_schema import (
    FEATURE_COLUMNS,
    load_dataset,
)

from grouped_split import (
    create_feature_groups,
    grouped_train_val_test_split,
)


DATA_DIR = Path("../datasets")
OUTPUT_DIR = Path("../manifests")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


for file in sorted(DATA_DIR.glob("*.csv")):

    dataset_name = file.stem.upper()

    df = load_dataset(file)

    # Stable row identifier within this raw normalized dataset.
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    # Compute the feature-vector group.
    df["feature_group_id"] = create_feature_groups(
        df,
        FEATURE_COLUMNS,
    )

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

    partition_map = {}

    for row_id in train_df["row_id"]:
        partition_map[row_id] = "train"

    for row_id in validation_df["row_id"]:
        partition_map[row_id] = "validation"

    for row_id in test_df["row_id"]:
        partition_map[row_id] = "test"

    manifest = df[
        [
            "row_id",
            "feature_group_id",
            "defects",
        ]
    ].copy()

    manifest.insert(
        0,
        "dataset",
        dataset_name,
    )

    manifest["partition"] = manifest["row_id"].map(
        partition_map
    )

    if manifest["partition"].isna().any():
        raise AssertionError(
            "Some rows were not assigned to a partition."
        )

    # Final leakage check.
    group_partition_counts = (
        manifest
        .groupby("feature_group_id")["partition"]
        .nunique()
    )

    if (group_partition_counts > 1).any():
        raise AssertionError(
            "A feature group appears in multiple partitions."
        )

    output = (
        OUTPUT_DIR
        / f"{dataset_name.lower()}_seed42_manifest.csv"
    )

    manifest.to_csv(
        output,
        index=False,
    )

    print(
        f"{dataset_name}: "
        f"{len(manifest)} rows -> {output}"
    )

print()
print("All split manifests generated successfully.")
