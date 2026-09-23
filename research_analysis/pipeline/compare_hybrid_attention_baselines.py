
from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "results"
)

HYBRID_SUMMARY_PATH = (
    RESULTS_DIR
    / "hybrid_attention_repeated_seed_summary.csv"
)

BASELINE_SUMMARY_PATH = (
    RESULTS_DIR
    / "baseline_repeated_seed_summary.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "hybrid_attention_vs_baselines_summary.csv"
)

OUTPUT_LONG_PATH = (
    RESULTS_DIR
    / "hybrid_attention_vs_baselines_summary_long.csv"
)

METADATA_PATH = (
    RESULTS_DIR
    / "hybrid_attention_vs_baselines_summary_metadata.json"
)


# ---------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------

EXPECTED_DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

EXPECTED_BASELINES = [
    "logistic_regression",
    "random_forest",
    "mlp",
]

PROPOSED_MODEL = "hybrid_attention"

EXPECTED_MODELS = [
    PROPOSED_MODEL,
    *EXPECTED_BASELINES,
]

EXPECTED_SEEDS = 5
EXPECTED_TRAINING_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

EXPECTED_SPLIT_SEED = 42


# ---------------------------------------------------------------------
# Metrics available in BOTH summaries
# ---------------------------------------------------------------------

COMPARISON_METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def validate_file(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file was not found:\n{path}"
        )


def validate_datasets(df, source_name):
    datasets = sorted(
        df["dataset"].astype(str).unique()
    )

    if datasets != sorted(EXPECTED_DATASETS):
        raise ValueError(
            f"{source_name} contains unexpected datasets.\n"
            f"Expected: {EXPECTED_DATASETS}\n"
            f"Found: {datasets}"
        )


def validate_hybrid_summary(df):
    required_columns = [
        "dataset",
        "n_seeds",
        "train_samples",
        "validation_samples",
        "test_samples",
    ]

    for metric in COMPARISON_METRICS:
        required_columns.extend(
            [
                f"{metric}_mean",
                f"{metric}_std",
            ]
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Hybrid summary is missing columns:\n"
            + "\n".join(missing)
        )

    if len(df) != len(EXPECTED_DATASETS):
        raise ValueError(
            "Hybrid summary should contain exactly "
            f"{len(EXPECTED_DATASETS)} dataset rows, "
            f"but found {len(df)}."
        )

    validate_datasets(
        df,
        "Hybrid summary"
    )

    if not (
        df["n_seeds"] == EXPECTED_SEEDS
    ).all():
        raise ValueError(
            "Hybrid summary does not contain "
            f"{EXPECTED_SEEDS} seeds for every dataset."
        )

    for column in [
        "train_samples",
        "validation_samples",
        "test_samples",
    ]:
        if df[column].isna().any():
            raise ValueError(
                "Hybrid summary contains missing "
                f"values in {column}."
            )


def validate_baseline_summary(df):
    required_columns = [
        "dataset",
        "model",
        "n_seeds",
        "split_seed",
    ]

    for metric in COMPARISON_METRICS:
        required_columns.extend(
            [
                f"test_{metric}_mean",
                f"test_{metric}_sd",
            ]
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Baseline summary is missing columns:\n"
            + "\n".join(missing)
        )

    validate_datasets(
        df,
        "Baseline summary"
    )

    models = sorted(
        df["model"].astype(str).unique()
    )

    if models != sorted(EXPECTED_BASELINES):
        raise ValueError(
            "Baseline summary contains unexpected models.\n"
            f"Expected: {EXPECTED_BASELINES}\n"
            f"Found: {models}"
        )

    expected_rows = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_BASELINES)
    )

    if len(df) != expected_rows:
        raise ValueError(
            "Baseline summary should contain exactly "
            f"{expected_rows} rows, but found {len(df)}."
        )

    if not (
        df["n_seeds"] == EXPECTED_SEEDS
    ).all():
        raise ValueError(
            "Baseline summary does not contain "
            f"{EXPECTED_SEEDS} seeds for every dataset/model."
        )

    if not (
        df["split_seed"] == EXPECTED_SPLIT_SEED
    ).all():
        raise ValueError(
            "Baseline summary contains a split seed "
            "different from the frozen seed-42 protocol."
        )


def validate_no_missing_comparison_metrics(df):
    for metric in COMPARISON_METRICS:
        mean_column = f"test_{metric}_mean"
        sd_column = f"test_{metric}_sd"

        if df[mean_column].isna().any():
            missing_rows = df.loc[
                df[mean_column].isna(),
                ["dataset", "model"]
            ]

            raise ValueError(
                f"Missing values found in {mean_column}:\n"
                f"{missing_rows.to_string(index=False)}"
            )

        if df[sd_column].isna().any():
            missing_rows = df.loc[
                df[sd_column].isna(),
                ["dataset", "model"]
            ]

            raise ValueError(
                f"Missing values found in {sd_column}:\n"
                f"{missing_rows.to_string(index=False)}"
            )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print("HYBRID ATTENTION VS BASELINE COMPARISON")
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # Check source files
    # -------------------------------------------------------------

    validate_file(
        HYBRID_SUMMARY_PATH
    )

    validate_file(
        BASELINE_SUMMARY_PATH
    )

    print(
        "Loading Hybrid Attention summary..."
    )

    hybrid_df = pd.read_csv(
        HYBRID_SUMMARY_PATH
    )

    print(
        "Loading baseline summary..."
    )

    baseline_df = pd.read_csv(
        BASELINE_SUMMARY_PATH
    )

    print()

    # -------------------------------------------------------------
    # Validate source summaries
    # -------------------------------------------------------------

    validate_hybrid_summary(
        hybrid_df
    )

    validate_baseline_summary(
        baseline_df
    )

    print(
        "Source validation: PASSED"
    )

    print()

    # -------------------------------------------------------------
    # Prepare Hybrid Attention rows
    #
    # Hybrid summary:
    #     roc_auc_mean
    #     roc_auc_std
    #
    # Baseline summary:
    #     test_roc_auc_mean
    #     test_roc_auc_sd
    #
    # We normalize names for the comparison table only.
    # No underlying result is changed.
    # -------------------------------------------------------------

    hybrid_rows = []

    for _, row in hybrid_df.iterrows():

        record = {
            "dataset": row["dataset"],
            "model": PROPOSED_MODEL,
            "n_seeds": int(row["n_seeds"]),
            "split_seed": EXPECTED_SPLIT_SEED,
            "train_samples": int(
                row["train_samples"]
            ),
            "validation_samples": int(
                row["validation_samples"]
            ),
            "test_samples": int(
                row["test_samples"]
            ),
        }

        for metric in COMPARISON_METRICS:

            record[
                f"test_{metric}_mean"
            ] = row[
                f"{metric}_mean"
            ]

            record[
                f"test_{metric}_sd"
            ] = row[
                f"{metric}_std"
            ]

        hybrid_rows.append(
            record
        )

    hybrid_comparison = pd.DataFrame(
        hybrid_rows
    )

    # -------------------------------------------------------------
    # Prepare baseline rows
    # -------------------------------------------------------------

    partition_sizes = (
        hybrid_df[
            [
                "dataset",
                "train_samples",
                "validation_samples",
                "test_samples",
            ]
        ]
        .drop_duplicates()
    )

    baseline_comparison = (
        baseline_df.copy()
    )

    baseline_comparison = (
        baseline_comparison.merge(
            partition_sizes,
            on="dataset",
            how="left",
            validate="many_to_one",
        )
    )

    # -------------------------------------------------------------
    # Combine Hybrid Attention + baselines
    # -------------------------------------------------------------

    comparison_df = pd.concat(
        [
            hybrid_comparison,
            baseline_comparison,
        ],
        ignore_index=True,
    )

    # -------------------------------------------------------------
    # Validate final row count
    # -------------------------------------------------------------

    expected_rows = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_MODELS)
    )

    if len(comparison_df) != expected_rows:
        raise ValueError(
            "Final comparison should contain exactly "
            f"{expected_rows} rows, but found "
            f"{len(comparison_df)}."
        )

    # -------------------------------------------------------------
    # Validate every dataset/model combination occurs once
    # -------------------------------------------------------------

    combination_counts = (
        comparison_df
        .groupby(
            ["dataset", "model"]
        )
        .size()
        .reset_index(
            name="count"
        )
    )

    if not (
        combination_counts["count"] == 1
    ).all():
        raise ValueError(
            "Final comparison contains duplicate "
            "dataset/model combinations."
        )

    # -------------------------------------------------------------
    # Validate expected model count
    # -------------------------------------------------------------

    observed_models = sorted(
        comparison_df[
            "model"
        ].astype(str).unique()
    )

    if observed_models != sorted(
        EXPECTED_MODELS
    ):
        raise ValueError(
            "Final comparison contains unexpected models.\n"
            f"Expected: {EXPECTED_MODELS}\n"
            f"Found: {observed_models}"
        )

    # -------------------------------------------------------------
    # Validate no missing common metrics
    # -------------------------------------------------------------

    validate_no_missing_comparison_metrics(
        comparison_df
    )

    # -------------------------------------------------------------
    # Sort and arrange columns
    # -------------------------------------------------------------

    output_columns = [
        "dataset",
        "model",
        "n_seeds",
        "split_seed",
        "train_samples",
        "validation_samples",
        "test_samples",
    ]

    for metric in COMPARISON_METRICS:

        output_columns.extend(
            [
                f"test_{metric}_mean",
                f"test_{metric}_sd",
            ]
        )

    comparison_df = (
        comparison_df[
            output_columns
        ]
        .sort_values(
            [
                "dataset",
                "model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Save wide comparison
    # -------------------------------------------------------------

    comparison_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Create long-format comparison
    # -------------------------------------------------------------

    long_records = []

    for _, row in comparison_df.iterrows():

        for metric in COMPARISON_METRICS:

            long_records.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "n_seeds": row["n_seeds"],
                    "split_seed": row["split_seed"],
                    "metric": metric,
                    "mean": row[
                        f"test_{metric}_mean"
                    ],
                    "sd": row[
                        f"test_{metric}_sd"
                    ],
                }
            )

    long_df = pd.DataFrame(
        long_records
    )

    long_df.to_csv(
        OUTPUT_LONG_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata = {
        "experiment": (
            "hybrid_attention_vs_classical_baselines"
        ),
        "datasets": EXPECTED_DATASETS,
        "models": EXPECTED_MODELS,
        "proposed_model": PROPOSED_MODEL,
        "baseline_models": EXPECTED_BASELINES,
        "training_seeds": EXPECTED_TRAINING_SEEDS,
        "n_training_seeds": EXPECTED_SEEDS,
        "split_seed": EXPECTED_SPLIT_SEED,
        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),
        "scaling_protocol": (
            "fit_train_only"
        ),
        "test_usage": (
            "final_evaluation_only"
        ),
        "comparison_metrics": (
            COMPARISON_METRICS
        ),
        "hybrid_attention_input": (
            "semantic feature-group tokens "
            "with 4 groups padded to width 8"
        ),
        "baseline_input": (
            "21-dimensional standardized "
            "feature vectors"
        ),
        "excluded_metrics": {
            "balanced_accuracy": (
                "not present in the Hybrid "
                "repeated-seed summary"
            ),
            "youden_j": (
                "not present in the Hybrid "
                "repeated-seed summary"
            ),
        },
        "note": (
            "Only metrics available in both "
            "source summaries are included. "
            "No missing Hybrid Attention "
            "metrics are reconstructed or inferred."
        ),
        "output_rows": int(
            len(comparison_df)
        ),
        "source_files": {
            "hybrid_attention": str(
                HYBRID_SUMMARY_PATH
            ),
            "baselines": str(
                BASELINE_SUMMARY_PATH
            ),
        },
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # -------------------------------------------------------------
    # Console validation summary
    # -------------------------------------------------------------

    print(
        "Final comparison validation: PASSED"
    )

    print()

    print(
        f"TOTAL COMPARISON ROWS: "
        f"{len(comparison_df)}"
    )

    print()

    print(
        "MODEL COUNTS:"
    )

    print(
        comparison_df[
            "model"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "DATASET COUNTS:"
    )

    print(
        comparison_df[
            "dataset"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "DATASET/MODEL COMBINATIONS:"
    )

    print(
        comparison_df[
            [
                "dataset",
                "model",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "COMMON TEST METRICS:"
    )

    print(
        ", ".join(
            COMPARISON_METRICS
        )
    )

    print()

    print(
        "COMPARISON SUMMARY:"
    )

    print(
        comparison_df[
            [
                "dataset",
                "model",
                "test_roc_auc_mean",
                "test_roc_auc_sd",
                "test_pr_auc_mean",
                "test_pr_auc_sd",
                "test_f1_mean",
                "test_f1_sd",
            ]
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "SAVED FILES:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        OUTPUT_LONG_PATH
    )

    print(
        METADATA_PATH
    )

    print()

    print("=" * 70)
    print(
        "COMPARISON COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()