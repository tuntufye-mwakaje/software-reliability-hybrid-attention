from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "results"
)

INPUT_FILE = (
    RESULTS_DIR
    / "hybrid_attention_repeated_seed_results.csv"
)

OUTPUT_SUMMARY_FILE = (
    RESULTS_DIR
    / "hybrid_attention_repeated_seed_summary.csv"
)

OUTPUT_LONG_FILE = (
    RESULTS_DIR
    / "hybrid_attention_repeated_seed_summary_long.csv"
)

OUTPUT_METADATA_FILE = (
    RESULTS_DIR
    / "hybrid_attention_repeated_seed_summary_metadata.json"
)

DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

EXPECTED_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "test_loss",
    "best_validation_loss",
    "best_epoch",
]


# ============================================================
# Load and validate input
# ============================================================

def load_results() -> pd.DataFrame:
    """
    Load the complete repeated-seed experiment results.
    """

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            "Repeated-seed results file was not found:\n"
            f"{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(df)} experiment rows."
    )

    return df


# ============================================================
# Validate repeated-seed experiment
# ============================================================

def validate_results(
    df: pd.DataFrame,
) -> None:
    """
    Verify that the input contains exactly the expected
    five datasets and five training seeds per dataset.
    """

    expected_rows = (
        len(DATASETS)
        * len(EXPECTED_SEEDS)
    )

    if len(df) != expected_rows:

        raise ValueError(
            "Unexpected number of experiment rows. "
            f"Expected {expected_rows}, "
            f"found {len(df)}."
        )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "dataset",
        "training_seed",
        "split_seed",
        "train_samples",
        "validation_samples",
        "test_samples",
    ] + METRICS

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required columns:\n"
            + "\n".join(missing_columns)
        )

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    actual_datasets = sorted(
        df["dataset"].unique()
    )

    expected_datasets = sorted(
        DATASETS
    )

    if actual_datasets != expected_datasets:

        raise ValueError(
            "Dataset validation failed.\n"
            f"Expected: {expected_datasets}\n"
            f"Found:    {actual_datasets}"
        )

    # --------------------------------------------------------
    # Training seed validation
    # --------------------------------------------------------

    actual_seeds = sorted(
        df["training_seed"]
        .astype(int)
        .unique()
    )

    if actual_seeds != EXPECTED_SEEDS:

        raise ValueError(
            "Training-seed validation failed.\n"
            f"Expected: {EXPECTED_SEEDS}\n"
            f"Found:    {actual_seeds}"
        )

    # --------------------------------------------------------
    # Frozen split validation
    # --------------------------------------------------------

    split_seeds = sorted(
        df["split_seed"]
        .astype(int)
        .unique()
    )

    if split_seeds != [42]:

        raise ValueError(
            "Frozen split validation failed.\n"
            f"Expected split seed [42], "
            f"found {split_seeds}"
        )

    # --------------------------------------------------------
    # Check every dataset has every training seed
    # --------------------------------------------------------

    for dataset_name in DATASETS:

        dataset_df = df[
            df["dataset"] == dataset_name
        ]

        dataset_seeds = sorted(
            dataset_df["training_seed"]
            .astype(int)
            .tolist()
        )

        if dataset_seeds != EXPECTED_SEEDS:

            raise ValueError(
                f"Seed coverage failed for {dataset_name}.\n"
                f"Expected: {EXPECTED_SEEDS}\n"
                f"Found:    {dataset_seeds}"
            )

    # --------------------------------------------------------
    # Check partition sizes are identical across seeds
    # --------------------------------------------------------

    partition_columns = [
        "train_samples",
        "validation_samples",
        "test_samples",
    ]

    for dataset_name in DATASETS:

        dataset_df = df[
            df["dataset"] == dataset_name
        ]

        for column in partition_columns:

            unique_values = (
                dataset_df[column]
                .dropna()
                .unique()
            )

            if len(unique_values) != 1:

                raise ValueError(
                    f"Partition inconsistency for "
                    f"{dataset_name}, {column}: "
                    f"{unique_values}"
                )

    # --------------------------------------------------------
    # Check metric values are numeric
    # --------------------------------------------------------

    for metric in METRICS:

        numeric_values = pd.to_numeric(
            df[metric],
            errors="coerce",
        )

        invalid_count = (
            numeric_values.isna().sum()
        )

        if invalid_count > 0:

            raise ValueError(
                f"Metric '{metric}' contains "
                f"{invalid_count} non-numeric values."
            )

    print()
    print(
        "Input validation passed."
    )

    print(
        f"Datasets: {', '.join(DATASETS)}"
    )

    print(
        "Training seeds: "
        + ", ".join(
            str(seed)
            for seed in EXPECTED_SEEDS
        )
    )

    print(
        f"Frozen split seed: {split_seeds[0]}"
    )


# ============================================================
# Calculate mean and standard deviation
# ============================================================

def calculate_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate mean and sample standard deviation across
    training seeds for every dataset and metric.

    Standard deviation uses ddof=1, which is the conventional
    sample standard deviation for repeated experimental runs.
    """

    rows = []

    for dataset_name in DATASETS:

        dataset_df = df[
            df["dataset"] == dataset_name
        ].copy()

        row = {
            "dataset": dataset_name,
            "n_seeds": len(dataset_df),
        }

        # ----------------------------------------------------
        # Partition information
        # ----------------------------------------------------

        row["train_samples"] = int(
            dataset_df["train_samples"].iloc[0]
        )

        row["validation_samples"] = int(
            dataset_df["validation_samples"].iloc[0]
        )

        row["test_samples"] = int(
            dataset_df["test_samples"].iloc[0]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        for metric in METRICS:

            values = pd.to_numeric(
                dataset_df[metric],
                errors="raise",
            ).astype(float)

            row[f"{metric}_mean"] = float(
                values.mean()
            )

            row[f"{metric}_std"] = float(
                values.std(
                    ddof=1
                )
            )

        rows.append(row)

    return pd.DataFrame(
        rows
    )


# ============================================================
# Create long-format summary
# ============================================================

def calculate_long_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a long-format table containing one row per
    dataset/metric combination.
    """

    rows = []

    for dataset_name in DATASETS:

        dataset_df = df[
            df["dataset"] == dataset_name
        ].copy()

        for metric in METRICS:

            values = pd.to_numeric(
                dataset_df[metric],
                errors="raise",
            ).astype(float)

            mean_value = float(
                values.mean()
            )

            std_value = float(
                values.std(
                    ddof=1
                )
            )

            rows.append(
                {
                    "dataset": dataset_name,
                    "metric": metric,
                    "n_seeds": len(values),
                    "mean": mean_value,
                    "std": std_value,
                    "mean_plus_minus_std": (
                        f"{mean_value:.6f} ± "
                        f"{std_value:.6f}"
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# Save metadata
# ============================================================

def save_metadata(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    """
    Save metadata describing the aggregation procedure.
    """

    metadata = {
        "experiment": (
            "repeated_seed_multi_dataset_hybrid_attention"
        ),

        "input_file": str(
            INPUT_FILE
        ),

        "datasets": DATASETS,

        "training_seeds": EXPECTED_SEEDS,

        "split_seed": 42,

        "number_of_runs": int(
            len(df)
        ),

        "number_of_datasets": len(
            DATASETS
        ),

        "training_seeds_per_dataset": len(
            EXPECTED_SEEDS
        ),

        "summary_rows": int(
            len(summary_df)
        ),

        "aggregation": {
            "location": (
                "across training seeds within each dataset"
            ),
            "central_tendency": "mean",
            "variability": (
                "sample standard deviation"
            ),
            "standard_deviation_ddof": 1,
        },

        "metrics": METRICS,

        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),

        "scaling_protocol": (
            "fit_train_only"
        ),

        "model_selection": (
            "minimum_validation_loss"
        ),

        "test_usage": (
            "final_evaluation_only"
        ),

        "interpretation": (
            "Mean and standard deviation summarize "
            "variation across independent training seeds "
            "while holding the frozen dataset partitions "
            "constant."
        ),
    }

    with open(
        OUTPUT_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "REPEATED-SEED RESULT AGGREGATION"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_results()

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_results(
        df
    )

    # --------------------------------------------------------
    # Calculate wide summary
    # --------------------------------------------------------

    summary_df = calculate_summary(
        df
    )

    # --------------------------------------------------------
    # Calculate long summary
    # --------------------------------------------------------

    long_summary_df = calculate_long_summary(
        df
    )

    # --------------------------------------------------------
    # Save wide summary
    # --------------------------------------------------------

    summary_df.to_csv(
        OUTPUT_SUMMARY_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save long summary
    # --------------------------------------------------------

    long_summary_df.to_csv(
        OUTPUT_LONG_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    save_metadata(
        df,
        summary_df,
    )

    # --------------------------------------------------------
    # Display summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "MEAN ± STANDARD DEVIATION"
    )
    print("=" * 70)

    display_columns = [
        "dataset",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
    ]

    print(
        summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Compact publication-style table
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "COMPACT METRIC SUMMARY"
    )
    print("=" * 70)

    compact_rows = []

    for _, row in summary_df.iterrows():

        compact_rows.append(
            {
                "Dataset": row["dataset"],

                "ROC-AUC": (
                    f"{row['roc_auc_mean']:.4f} ± "
                    f"{row['roc_auc_std']:.4f}"
                ),

                "PR-AUC": (
                    f"{row['pr_auc_mean']:.4f} ± "
                    f"{row['pr_auc_std']:.4f}"
                ),

                "Accuracy": (
                    f"{row['accuracy_mean']:.4f} ± "
                    f"{row['accuracy_std']:.4f}"
                ),

                "Precision": (
                    f"{row['precision_mean']:.4f} ± "
                    f"{row['precision_std']:.4f}"
                ),

                "Recall": (
                    f"{row['recall_mean']:.4f} ± "
                    f"{row['recall_std']:.4f}"
                ),

                "F1": (
                    f"{row['f1_mean']:.4f} ± "
                    f"{row['f1_std']:.4f}"
                ),
            }
        )

    compact_df = pd.DataFrame(
        compact_rows
    )

    print(
        compact_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Output locations
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "AGGREGATION COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Wide summary saved to:"
    )

    print(
        OUTPUT_SUMMARY_FILE
    )

    print()
    print(
        "Long summary saved to:"
    )

    print(
        OUTPUT_LONG_FILE
    )

    print()
    print(
        "Metadata saved to:"
    )

    print(
        OUTPUT_METADATA_FILE
    )


if __name__ == "__main__":
    main()