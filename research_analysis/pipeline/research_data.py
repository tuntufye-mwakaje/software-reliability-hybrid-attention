from pathlib import Path
import sys

import pandas as pd


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Allow imports from this pipeline directory when the file is executed
# directly with Python.
PIPELINE_DIR = Path(__file__).resolve().parent

if str(PIPELINE_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIR))


from dataset_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    CANONICAL_COLUMNS,
    canonicalize_dataset,
)


# ---------------------------------------------------------------------
# Frozen research datasets
# ---------------------------------------------------------------------

RAW_DATASETS = {
    "CM1": PROJECT_ROOT / "research_analysis" / "datasets" / "cm1.csv",
    "JM1": PROJECT_ROOT / "research_analysis" / "datasets" / "jm1.csv",
    "KC1": PROJECT_ROOT / "research_analysis" / "datasets" / "kc1.csv",
    "KC2": PROJECT_ROOT / "research_analysis" / "datasets" / "kc2.csv",
    "PC1": PROJECT_ROOT / "research_analysis" / "datasets" / "pc1.csv",
}


MANIFEST_DATASETS = {
    "CM1": PROJECT_ROOT / "research_analysis" / "manifests" / "cm1_seed42_manifest.csv",
    "JM1": PROJECT_ROOT / "research_analysis" / "manifests" / "jm1_seed42_manifest.csv",
    "KC1": PROJECT_ROOT / "research_analysis" / "manifests" / "kc1_seed42_manifest.csv",
    "KC2": PROJECT_ROOT / "research_analysis" / "manifests" / "kc2_seed42_manifest.csv",
    "PC1": PROJECT_ROOT / "research_analysis" / "manifests" / "pc1_seed42_manifest.csv",
}


EXPECTED_PARTITIONS = {
    "train",
    "validation",
    "test",
}


# ---------------------------------------------------------------------
# Raw dataset loading
# ---------------------------------------------------------------------

def load_raw_dataset(dataset_name):
    """
    Load and canonicalize one frozen research dataset.
    """

    dataset_name = dataset_name.upper()

    if dataset_name not in RAW_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Expected one of: {sorted(RAW_DATASETS)}"
        )

    path = RAW_DATASETS[dataset_name]

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    raw_df = pd.read_csv(path)

    df = canonicalize_dataset(
        raw_df,
        source_name=str(path),
    )

    # Preserve a stable row identifier corresponding to the raw CSV
    # row order. This is the identifier referenced by the frozen
    # manifest.
    df = df.reset_index(drop=True)
    df.index.name = "row_id"

    return df


# ---------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------

def load_manifest(dataset_name):
    """
    Load the frozen train/validation/test manifest.
    """

    dataset_name = dataset_name.upper()

    if dataset_name not in MANIFEST_DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Expected one of: {sorted(MANIFEST_DATASETS)}"
        )

    path = MANIFEST_DATASETS[dataset_name]

    if not path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {path}"
        )

    manifest = pd.read_csv(path)

    required_columns = {
        "dataset",
        "row_id",
        "feature_group_id",
        TARGET_COLUMN,
        "partition",
    }

    missing_columns = required_columns.difference(manifest.columns)

    if missing_columns:
        raise ValueError(
            f"Manifest {path} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    manifest = manifest.copy()

    manifest["dataset"] = manifest["dataset"].astype(str).str.upper()
    manifest["row_id"] = pd.to_numeric(
        manifest["row_id"],
        errors="raise",
    ).astype(int)

    manifest["feature_group_id"] = pd.to_numeric(
        manifest["feature_group_id"],
        errors="raise",
    ).astype(int)

    manifest[TARGET_COLUMN] = manifest[TARGET_COLUMN].astype(bool)

    manifest["partition"] = (
        manifest["partition"]
        .astype(str)
        .str.lower()
    )

    return manifest


# ---------------------------------------------------------------------
# Feature-group calculation
# ---------------------------------------------------------------------

def calculate_feature_group_ids(df):
    """
    Reconstruct feature-group IDs from the canonical feature vectors.

    A feature group represents rows with identical predictor values.
    """

    feature_hashes = pd.util.hash_pandas_object(
        df[FEATURE_COLUMNS],
        index=False,
    )

    group_ids = pd.factorize(
        feature_hashes,
        sort=False,
    )[0]

    return pd.Series(
        group_ids,
        index=df.index,
        name="feature_group_id",
    )


# ---------------------------------------------------------------------
# Manifest integrity validation
# ---------------------------------------------------------------------

def validate_manifest_integrity(
    dataset_name,
    df,
    manifest,
):
    """
    Verify that the frozen manifest still matches the raw dataset.
    """

    dataset_name = dataset_name.upper()

    # -------------------------------------------------------------
    # Dataset identity
    # -------------------------------------------------------------

    if set(manifest["dataset"]) != {dataset_name}:
        raise ValueError(
            f"{dataset_name}: manifest dataset identity mismatch."
        )

    # -------------------------------------------------------------
    # Row count
    # -------------------------------------------------------------

    if len(df) != len(manifest):
        raise ValueError(
            f"{dataset_name}: row-count mismatch. "
            f"Dataset={len(df)}, manifest={len(manifest)}"
        )

    # -------------------------------------------------------------
    # Row IDs
    # -------------------------------------------------------------

    expected_row_ids = set(range(len(df)))
    manifest_row_ids = set(manifest["row_id"])

    if manifest_row_ids != expected_row_ids:
        missing = expected_row_ids - manifest_row_ids
        extra = manifest_row_ids - expected_row_ids

        raise ValueError(
            f"{dataset_name}: row-ID mismatch. "
            f"Missing={sorted(missing)[:10]}, "
            f"Extra={sorted(extra)[:10]}"
        )

    if manifest["row_id"].duplicated().any():
        raise ValueError(
            f"{dataset_name}: duplicate row IDs found in manifest."
        )

    # -------------------------------------------------------------
    # Partition validation
    # -------------------------------------------------------------

    actual_partitions = set(manifest["partition"])

    if actual_partitions != EXPECTED_PARTITIONS:
        raise ValueError(
            f"{dataset_name}: invalid partitions. "
            f"Found={sorted(actual_partitions)}, "
            f"Expected={sorted(EXPECTED_PARTITIONS)}"
        )

    # -------------------------------------------------------------
    # Target consistency
    # -------------------------------------------------------------

    manifest_targets = (
        manifest
        .sort_values("row_id")[TARGET_COLUMN]
        .reset_index(drop=True)
    )

    dataset_targets = (
        df[TARGET_COLUMN]
        .reset_index(drop=True)
    )

    if not manifest_targets.equals(dataset_targets):
        raise ValueError(
            f"{dataset_name}: manifest target labels do not "
            f"match the canonicalized dataset."
        )

        # -------------------------------------------------------------
    # Feature-group consistency
    # -------------------------------------------------------------
    #
    # The integer feature_group_id stored in the manifest is an
    # identifier, not the scientific definition of a feature group.
    #
    # Therefore, validate the actual membership of each group rather
    # than requiring the integer IDs to be regenerated in exactly the
    # same numerical order.
    #
    # This is important because factorize() assigns integer IDs based
    # on first appearance in the DataFrame.

    calculated_groups = calculate_feature_group_ids(df)

    manifest_sorted = (
        manifest
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    calculated_sorted = (
        calculated_groups
        .reset_index(drop=True)
    )

    # Verify that every manifest group contains identical predictor
    # vectors.
    manifest_group_counts = (
        manifest_sorted
        .groupby("feature_group_id")
        .size()
    )

    for group_id in manifest_group_counts.index:

        row_ids = manifest_sorted.loc[
            manifest_sorted["feature_group_id"] == group_id,
            "row_id",
        ].tolist()

        group_features = df.loc[
            row_ids,
            FEATURE_COLUMNS,
        ]

        if len(group_features.drop_duplicates()) != 1:
            raise ValueError(
                f"{dataset_name}: manifest group {group_id} "
                f"contains different feature vectors."
            )

    # Verify the reverse direction:
    # identical feature vectors must belong to one manifest group.
    feature_to_manifest_group = {}

    for row_id in range(len(df)):

        feature_key = tuple(
            df.loc[row_id, FEATURE_COLUMNS].tolist()
        )

        manifest_group = int(
            manifest_sorted.loc[
                row_id,
                "feature_group_id",
            ]
        )

        previous_group = feature_to_manifest_group.get(
            feature_key
        )

        if previous_group is None:
            feature_to_manifest_group[feature_key] = (
                manifest_group
            )

        elif previous_group != manifest_group:
            raise ValueError(
                f"{dataset_name}: identical feature vectors were "
                f"assigned to different manifest groups."
            )

    # -------------------------------------------------------------
    # Feature-group leakage check
    # -------------------------------------------------------------

    group_partition_counts = (
        manifest
        .groupby("feature_group_id")["partition"]
        .nunique()
    )

    leaking_groups = group_partition_counts[
        group_partition_counts > 1
    ]

    if not leaking_groups.empty:
        raise ValueError(
            f"{dataset_name}: feature-group leakage detected. "
            f"Groups crossing partitions: "
            f"{leaking_groups.index.tolist()[:20]}"
        )

    # -------------------------------------------------------------
    # Canonical schema
    # -------------------------------------------------------------

    if list(df.columns) != CANONICAL_COLUMNS:
        raise ValueError(
            f"{dataset_name}: canonical schema mismatch."
        )

    return True


# ---------------------------------------------------------------------
# Frozen partition reconstruction
# ---------------------------------------------------------------------

def reconstruct_frozen_partitions(dataset_name):
    """
    Reconstruct train/validation/test DataFrames using the frozen
    manifest.

    No new random split is performed here.
    """

    dataset_name = dataset_name.upper()

    df = load_raw_dataset(dataset_name)
    manifest = load_manifest(dataset_name)

    validate_manifest_integrity(
        dataset_name,
        df,
        manifest,
    )

    manifest_indexed = (
        manifest
        .set_index("row_id")
        .loc[df.index]
    )

    train_ids = manifest_indexed.index[
        manifest_indexed["partition"] == "train"
    ]

    validation_ids = manifest_indexed.index[
        manifest_indexed["partition"] == "validation"
    ]

    test_ids = manifest_indexed.index[
        manifest_indexed["partition"] == "test"
    ]

    train_df = df.loc[train_ids].copy()
    validation_df = df.loc[validation_ids].copy()
    test_df = df.loc[test_ids].copy()

    # Preserve the original row ID as an explicit column.
    train_df.insert(
        0,
        "row_id",
        train_df.index,
    )

    validation_df.insert(
        0,
        "row_id",
        validation_df.index,
    )

    test_df.insert(
        0,
        "row_id",
        test_df.index,
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ---------------------------------------------------------------------
# Public research-dataset loader
# ---------------------------------------------------------------------

def load_research_dataset(dataset_name):
    """
    Load one complete frozen research dataset and reconstruct
    its train/validation/test partitions.
    """

    return reconstruct_frozen_partitions(
        dataset_name
    )


# ---------------------------------------------------------------------
# Command-line validation
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("Frozen research dataset loader")
    print("=" * 70)

    for dataset_name in RAW_DATASETS:

        print()
        print(f"Dataset: {dataset_name}")
        print("-" * 70)

        train_df, validation_df, test_df = (
            load_research_dataset(dataset_name)
        )

        combined = pd.concat(
            [
                train_df.assign(partition="train"),
                validation_df.assign(partition="validation"),
                test_df.assign(partition="test"),
            ],
            ignore_index=True,
        )

        print(
            f"Train:      {len(train_df):>6} rows | "
            f"positive={int(train_df[TARGET_COLUMN].sum()):>5} | "
            f"negative={int((~train_df[TARGET_COLUMN]).sum()):>5}"
        )

        print(
            f"Validation: {len(validation_df):>6} rows | "
            f"positive={int(validation_df[TARGET_COLUMN].sum()):>5} | "
            f"negative={int((~validation_df[TARGET_COLUMN]).sum()):>5}"
        )

        print(
            f"Test:       {len(test_df):>6} rows | "
            f"positive={int(test_df[TARGET_COLUMN].sum()):>5} | "
            f"negative={int((~test_df[TARGET_COLUMN]).sum()):>5}"
        )

        print(
            f"Total:      {len(combined):>6} rows"
        )

    print()
    print("=" * 70)
    print("All frozen research datasets loaded and validated successfully.")