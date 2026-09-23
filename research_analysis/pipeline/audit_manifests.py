from pathlib import Path
import pandas as pd

from dataset_schema import (
    FEATURE_COLUMNS,
    load_dataset,
)


DATA_DIR = Path("../datasets")
MANIFEST_DIR = Path("../manifests")


for data_file in sorted(DATA_DIR.glob("*.csv")):

    dataset = data_file.stem.upper()

    manifest_file = (
        MANIFEST_DIR
        / f"{data_file.stem.lower()}_seed42_manifest.csv"
    )

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_file}"
        )

    df = load_dataset(data_file).reset_index(drop=True)
    manifest = pd.read_csv(manifest_file)

    print()
    print("=" * 70)
    print(dataset)
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Row-count check
    # ---------------------------------------------------------

    if len(df) != len(manifest):
        raise AssertionError(
            f"Row count mismatch: "
            f"dataset={len(df)}, "
            f"manifest={len(manifest)}"
        )

    print("Row count:", len(df), "OK")

    # ---------------------------------------------------------
    # 2. Row-ID uniqueness/completeness
    # ---------------------------------------------------------

    expected_ids = set(range(len(df)))
    manifest_ids = set(manifest["row_id"])

    if expected_ids != manifest_ids:
        raise AssertionError(
            "Manifest row IDs do not exactly match dataset rows."
        )

    if manifest["row_id"].duplicated().any():
        raise AssertionError(
            "Duplicate row IDs found in manifest."
        )

    print("Row IDs:", "OK")

    # ---------------------------------------------------------
    # 3. Partition validity
    # ---------------------------------------------------------

    expected_partitions = {
        "train",
        "validation",
        "test",
    }

    actual_partitions = set(
        manifest["partition"].dropna().unique()
    )

    if actual_partitions != expected_partitions:
        raise AssertionError(
            f"Unexpected partitions: {actual_partitions}"
        )

    if manifest["partition"].isna().any():
        raise AssertionError(
            "Missing partition assignments."
        )

    print("Partition labels:", "OK")

    # ---------------------------------------------------------
    # 4. Feature-group leakage check
    # ---------------------------------------------------------

    group_partition_counts = (
        manifest
        .groupby("feature_group_id")["partition"]
        .nunique()
    )

    leaking_groups = (
        group_partition_counts[
            group_partition_counts > 1
        ]
    )

    if len(leaking_groups) > 0:
        raise AssertionError(
            f"Feature-group leakage detected: "
            f"{len(leaking_groups)} groups."
        )

    print("Feature-group leakage:", "NONE")

    # ---------------------------------------------------------
    # 5. Target consistency check
    # ---------------------------------------------------------

    source_targets = (
        df["defects"]
        .astype(bool)
        .reset_index(drop=True)
    )

    manifest_targets = (
        manifest
        .sort_values("row_id")["defects"]
        .astype(bool)
        .reset_index(drop=True)
    )

    if not source_targets.equals(manifest_targets):
        raise AssertionError(
            "Manifest target values do not match source data."
        )

    print("Target consistency:", "OK")

    # ---------------------------------------------------------
    # 6. Partition summary
    # ---------------------------------------------------------

    summary = (
        manifest
        .groupby("partition")
        .agg(
            rows=("row_id", "size"),
            positive=("defects", "sum"),
        )
    )

    summary["negative"] = (
        summary["rows"] - summary["positive"]
    )

    summary["positive_rate"] = (
        summary["positive"] / summary["rows"]
    )

    print()
    print(summary)

print()
print("=" * 70)
print("ALL MANIFEST AUDITS PASSED.")
print("=" * 70)
