from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Paths
# ============================================================

PIPELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PIPELINE_DIR.parent / "results"

INPUT_FILE = (
    RESULTS_DIR
    / "baseline_repeated_seed_results.csv"
)

OUTPUT_SUMMARY = (
    RESULTS_DIR
    / "baseline_repeated_seed_summary.csv"
)

OUTPUT_LONG = (
    RESULTS_DIR
    / "baseline_repeated_seed_summary_long.csv"
)

OUTPUT_METADATA = (
    RESULTS_DIR
    / "baseline_repeated_seed_summary_metadata.json"
)


# ============================================================
# Experimental configuration
# ============================================================

EXPECTED_DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

EXPECTED_MODELS = [
    "logistic_regression",
    "random_forest",
    "mlp",
]

EXPECTED_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

EXPECTED_SPLIT_SEED = 42


# ============================================================
# Metrics to aggregate
# ============================================================

METRICS = [
    "test_roc_auc",
    "test_pr_auc",
    "test_accuracy",
    "test_balanced_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_youden_j",
]


# ============================================================
# Validation
# ============================================================

def validate_input(df):
    """Validate the repeated-seed baseline results."""

    required_columns = [
        "dataset",
        "model",
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
            f"Missing required columns: {missing_columns}"
        )

    expected_rows = (
        len(EXPECTED_DATASETS)
        * len(EXPECTED_MODELS)
        * len(EXPECTED_SEEDS)
    )

    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"found {len(df)}."
        )

    # ---------------------------------------------------------
    # Dataset validation
    # ---------------------------------------------------------

    actual_datasets = sorted(
        df["dataset"].unique().tolist()
    )

    if actual_datasets != sorted(EXPECTED_DATASETS):
        raise ValueError(
            "Unexpected dataset set: "
            f"{actual_datasets}"
        )

    # ---------------------------------------------------------
    # Model validation
    # ---------------------------------------------------------

    actual_models = sorted(
        df["model"].unique().tolist()
    )

    if actual_models != sorted(EXPECTED_MODELS):
        raise ValueError(
            "Unexpected model set: "
            f"{actual_models}"
        )

    # ---------------------------------------------------------
    # Seed validation
    # ---------------------------------------------------------

    actual_seeds = sorted(
        df["training_seed"].unique().tolist()
    )

    if actual_seeds != EXPECTED_SEEDS:
        raise ValueError(
            "Unexpected training seeds: "
            f"{actual_seeds}"
        )

    actual_split_seeds = sorted(
        df["split_seed"].unique().tolist()
    )

    if actual_split_seeds != [
        EXPECTED_SPLIT_SEED
    ]:
        raise ValueError(
            "Unexpected split seed values: "
            f"{actual_split_seeds}"
        )

    # ---------------------------------------------------------
    # Dataset/model/seed completeness
    # ---------------------------------------------------------

    counts = (
        df.groupby(
            ["dataset", "model"]
        )
        .size()
        .reset_index(name="count")
    )

    invalid_counts = counts[
        counts["count"] != len(EXPECTED_SEEDS)
    ]

    if not invalid_counts.empty:
        raise ValueError(
            "Some dataset/model combinations do not "
            "contain exactly five training seeds:\n"
            f"{invalid_counts}"
        )

    # ---------------------------------------------------------
    # Frozen partition sizes
    # ---------------------------------------------------------

    expected_partitions = {
        "CM1": (309, 84, 105),
        "JM1": (8491, 2190, 2523),
        "KC1": (1370, 307, 432),
        "KC2": (346, 66, 110),
        "PC1": (722, 171, 216),
    }

    for dataset, expected in expected_partitions.items():

        subset = df[
            df["dataset"] == dataset
        ]

        actual = (
            subset[
                [
                    "train_samples",
                    "validation_samples",
                    "test_samples",
                ]
            ]
            .drop_duplicates()
            .to_numpy()
        )

        if len(actual) != 1:
            raise ValueError(
                f"{dataset}: partition sizes are "
                "not consistent across runs."
            )

        actual_tuple = tuple(
            actual[0].tolist()
        )

        if actual_tuple != expected:
            raise ValueError(
                f"{dataset}: expected partition sizes "
                f"{expected}, found {actual_tuple}."
            )

    # ---------------------------------------------------------
    # Missing metric validation
    # ---------------------------------------------------------

    missing = df[METRICS].isna().sum()

    if missing.sum() > 0:
        raise ValueError(
            "Missing metric values found:\n"
            f"{missing[missing > 0]}"
        )


# ============================================================
# Aggregation
# ============================================================

def aggregate_results(df):
    """
    Calculate mean and sample standard deviation across
    training seeds for each dataset/model combination.
    """

    rows = []

    grouped = df.groupby(
        ["dataset", "model"],
        sort=True,
    )

    for (
        dataset,
        model,
    ), group in grouped:

        row = {
            "dataset": dataset,
            "model": model,
            "n_seeds": len(group),
            "split_seed": int(
                group["split_seed"].iloc[0]
            ),
        }

        # -----------------------------------------------------
        # Aggregate each test metric
        # -----------------------------------------------------

        for metric in METRICS:

            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            row[
                f"{metric}_mean"
            ] = float(
                np.mean(values)
            )

            row[
                f"{metric}_sd"
            ] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Long-format aggregation
# ============================================================

def create_long_format(summary):
    """Create publication-friendly long-format results."""

    rows = []

    for _, row in summary.iterrows():

        for metric in METRICS:

            rows.append(
                {
                    "dataset": row["dataset"],
                    "model": row["model"],
                    "n_seeds": row["n_seeds"],
                    "split_seed": row["split_seed"],
                    "metric": metric,
                    "mean": row[
                        f"{metric}_mean"
                    ],
                    "sd": row[
                        f"{metric}_sd"
                    ],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Metadata
# ============================================================

def create_metadata(df):
    """Create reproducibility metadata."""

    metadata = {
        "experiment": (
            "repeated_seed_classical_baselines"
        ),
        "input_file": INPUT_FILE.name,
        "datasets": EXPECTED_DATASETS,
        "models": EXPECTED_MODELS,
        "training_seeds": EXPECTED_SEEDS,
        "split_seed": EXPECTED_SPLIT_SEED,
        "total_runs": int(len(df)),
        "aggregation": (
            "mean_and_sample_standard_deviation"
        ),
        "standard_deviation_ddof": 1,
        "threshold": 0.5,
        "feature_representation": (
            "21_dimensional_standardized_feature_vector"
        ),
        "scaling_protocol": (
            "fit_train_only"
        ),
        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),
        "test_usage": (
            "final_evaluation_only"
        ),
        "models": {
            "logistic_regression": {
                "max_iter": 2000,
                "class_weight": None,
                "random_state": (
                    "training_seed"
                ),
            },
            "random_forest": {
                "n_estimators": 300,
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "class_weight": None,
                "n_jobs": -1,
                "random_state": (
                    "training_seed"
                ),
            },
            "mlp": {
                "hidden_layers": [
                    64,
                    32,
                ],
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "batch_size": 32,
                "learning_rate_init": 0.001,
                "max_iter": 50,
                "early_stopping": False,
                "random_state": (
                    "training_seed"
                ),
            },
        },
        "metrics": METRICS,
    }

    return metadata


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("BASELINE REPEATED-SEED AGGREGATION")
    print("=" * 70)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(
        f"\nReading:\n{INPUT_FILE}"
    )

    df = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"\nInput rows: {len(df)}"
    )

    print(
        "\nValidating experiment structure..."
    )

    validate_input(df)

    print(
        "Validation passed."
    )

    print(
        "\nAggregating mean ± sample SD "
        "across training seeds..."
    )

    summary = aggregate_results(
        df
    )

    long_summary = create_long_format(
        summary
    )

    metadata = create_metadata(
        df
    )

    # ---------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------

    summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
    )

    long_summary.to_csv(
        OUTPUT_LONG,
        index=False,
    )

    import json

    with open(
        OUTPUT_METADATA,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Console summary
    # ---------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "AGGREGATED TEST RESULTS"
    )

    print(
        "=" * 70
    )

    display_columns = [
        "dataset",
        "model",
        "test_roc_auc_mean",
        "test_roc_auc_sd",
        "test_pr_auc_mean",
        "test_pr_auc_sd",
        "test_f1_mean",
        "test_f1_sd",
    ]

    display_df = summary[
        display_columns
    ].copy()

    print(
        display_df.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.6f}"
            ),
        )
    )

    print(
        "\nSummary saved to:"
    )
    print(
        OUTPUT_SUMMARY
    )

    print(
        "\nLong-format summary saved to:"
    )
    print(
        OUTPUT_LONG
    )

    print(
        "\nMetadata saved to:"
    )
    print(
        OUTPUT_METADATA
    )

    print(
        "\n" + "=" * 70
    )


if __name__ == "__main__":
    main()