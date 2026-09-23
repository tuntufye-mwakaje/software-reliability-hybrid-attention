from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "ablation_repeated_seed_results.csv"

AGGREGATED_FILE = RESULTS_DIR / "ablation_repeated_seed_aggregate.csv"
METADATA_FILE = RESULTS_DIR / "ablation_repeated_seed_aggregate_metadata.json"


# ============================================================
# Frozen experimental protocol
# ============================================================

EXPECTED_VARIANTS = [
    "A0_full_hybrid_attention",
    "A1_no_self_attention",
    "A2_no_semantic_tokenization",
    "A3_single_head_attention",
]

EXPECTED_DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

EXPECTED_SEEDS = [42, 43, 44, 45, 46]

EXPECTED_SPLIT_SEED = 42

EXPECTED_PARTITIONS = {
    "CM1": (309, 84, 105),
    "JM1": (8491, 2190, 2523),
    "KC1": (1370, 307, 432),
    "KC2": (346, 66, 110),
    "PC1": (722, 171, 216),
}

METRICS = [
    "roc_auc",
    "pr_auc",
    "accuracy",
    "precision",
    "recall",
    "f1",
]


# ============================================================
# Validation
# ============================================================

def validate_results(df: pd.DataFrame) -> None:
    print("=" * 70)
    print("VALIDATING ABLATION RESULTS")
    print("=" * 70)

    required_columns = [
        "variant",
        "dataset",
        "training_seed",
        "split_seed",
        "train_samples",
        "validation_samples",
        "test_samples",
        "parameter_count",
        "input_dimension",
        "num_heads",
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

    if len(df) != 100:
        raise ValueError(
            f"Expected exactly 100 rows, found {len(df)}."
        )

    variants = sorted(df["variant"].unique())

    if variants != sorted(EXPECTED_VARIANTS):
        raise ValueError(
            f"Unexpected variants: {variants}"
        )

    datasets = sorted(df["dataset"].unique())

    if datasets != sorted(EXPECTED_DATASETS):
        raise ValueError(
            f"Unexpected datasets: {datasets}"
        )

    seeds = sorted(df["training_seed"].unique())

    if seeds != EXPECTED_SEEDS:
        raise ValueError(
            f"Unexpected training seeds: {seeds}"
        )

    split_seeds = sorted(df["split_seed"].unique())

    if split_seeds != [EXPECTED_SPLIT_SEED]:
        raise ValueError(
            f"Unexpected split seeds: {split_seeds}"
        )

    duplicate_count = df.duplicated(
        ["variant", "dataset", "training_seed"]
    ).sum()

    if duplicate_count != 0:
        raise ValueError(
            f"Found {duplicate_count} duplicate "
            "variant/dataset/seed rows."
        )

    group_sizes = (
        df.groupby(["variant", "dataset"])
        .size()
    )

    if not (group_sizes == 5).all():
        raise ValueError(
            "Every variant/dataset combination must contain exactly "
            "five training seeds."
        )

    missing_metrics = int(
        df[METRICS].isna().sum().sum()
    )

    if missing_metrics != 0:
        raise ValueError(
            f"Found {missing_metrics} missing metric values."
        )

    # --------------------------------------------------------
    # Frozen partition validation
    # --------------------------------------------------------

    for dataset, expected_sizes in EXPECTED_PARTITIONS.items():

        subset = df[df["dataset"] == dataset]

        actual_sizes = (
            subset[
                [
                    "train_samples",
                    "validation_samples",
                    "test_samples",
                ]
            ]
            .drop_duplicates()
            .values
            .tolist()
        )

        if actual_sizes != [list(expected_sizes)]:
            raise ValueError(
                f"Frozen partition mismatch for {dataset}: "
                f"expected {expected_sizes}, "
                f"found {actual_sizes}"
            )

    # --------------------------------------------------------
    # Architecture validation
    # --------------------------------------------------------

    expected_architecture = {
        "A0_full_hybrid_attention": {
            "parameter_count": 9985,
            "input_dimension": 8,
            "num_heads": 4,
        },
        "A1_no_self_attention": {
            "parameter_count": 1441,
            "input_dimension": 8,
            "num_heads": 0,
        },
        "A2_no_semantic_tokenization": {
            "parameter_count": 1857,
            "input_dimension": 21,
            "num_heads": 0,
        },
        "A3_single_head_attention": {
            "parameter_count": 9985,
            "input_dimension": 8,
            "num_heads": 1,
        },
    }

    for variant, expected in expected_architecture.items():

        subset = df[df["variant"] == variant]

        for column, expected_value in expected.items():

            actual_values = sorted(
                subset[column].unique().tolist()
            )

            if actual_values != [expected_value]:
                raise ValueError(
                    f"{variant}: unexpected {column}: "
                    f"{actual_values}; "
                    f"expected {expected_value}"
                )

    print("Rows:", len(df))
    print("Variants:", variants)
    print("Datasets:", datasets)
    print("Training seeds:", seeds)
    print("Split seed:", split_seeds)
    print("Duplicate rows:", duplicate_count)
    print("Missing metric values:", missing_metrics)

    print("\nFrozen partition validation:")

    for dataset, sizes in EXPECTED_PARTITIONS.items():
        print(
            f"  {dataset}: "
            f"train={sizes[0]}, "
            f"validation={sizes[1]}, "
            f"test={sizes[2]}"
        )

    print("\nArchitecture validation:")

    for variant, values in expected_architecture.items():
        print(
            f"  {variant}: "
            f"parameters={values['parameter_count']}, "
            f"input_dimension={values['input_dimension']}, "
            f"heads={values['num_heads']}"
        )

    print("\nVALIDATION PASSED")


# ============================================================
# Five-seed aggregation
# ============================================================

def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for variant in EXPECTED_VARIANTS:

        for dataset in EXPECTED_DATASETS:

            subset = df[
                (df["variant"] == variant)
                & (df["dataset"] == dataset)
            ].copy()

            if len(subset) != 5:
                raise ValueError(
                    f"{variant} / {dataset}: "
                    f"expected 5 rows, found {len(subset)}"
                )

            result = {
                "variant": variant,
                "dataset": dataset,
            }

            for metric in METRICS:

                values = subset[metric].astype(float)

                result[f"{metric}_mean"] = values.mean()
                result[f"{metric}_sd"] = values.std(
                    ddof=1
                )

            rows.append(result)

    aggregate = pd.DataFrame(rows)

    return aggregate


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    validate_results(df)

    aggregate = aggregate_results(df)

    aggregate.to_csv(
        AGGREGATED_FILE,
        index=False,
        float_format="%.6f",
    )

    metadata = {
        "experiment": "four_variant_repeated_seed_ablation",
        "input_file": str(INPUT_FILE),
        "output_file": str(AGGREGATED_FILE),
        "aggregation": (
            "mean_and_sample_standard_deviation_"
            "across_training_seeds"
        ),
        "training_seeds": EXPECTED_SEEDS,
        "split_seed": EXPECTED_SPLIT_SEED,
        "datasets": EXPECTED_DATASETS,
        "variants": EXPECTED_VARIANTS,
        "number_of_runs": 100,
        "runs_per_variant_dataset": 5,
        "metrics": METRICS,
        "standard_deviation": (
            "sample_standard_deviation_ddof_1"
        ),
        "frozen_partitions": EXPECTED_PARTITIONS,
    }

    METADATA_FILE.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("FIVE-SEED AGGREGATION")
    print("=" * 70)

    display_columns = [
        "variant",
        "dataset",
        "roc_auc_mean",
        "roc_auc_sd",
        "pr_auc_mean",
        "pr_auc_sd",
        "f1_mean",
        "f1_sd",
    ]

    print(
        aggregate[display_columns]
        .to_string(index=False)
    )

    print("\nAggregated results saved to:")
    print(AGGREGATED_FILE)

    print("\nAggregation metadata saved to:")
    print(METADATA_FILE)


if __name__ == "__main__":
    main()