
from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "results"
)

HYBRID_RESULTS_PATH = (
    RESULTS_DIR
    / "hybrid_attention_repeated_seed_results.csv"
)

BASELINE_RESULTS_PATH = (
    RESULTS_DIR
    / "baseline_repeated_seed_results.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "hybrid_attention_statistical_comparison.csv"
)

OUTPUT_LONG_PATH = (
    RESULTS_DIR
    / "hybrid_attention_statistical_comparison_long.csv"
)

METADATA_PATH = (
    RESULTS_DIR
    / "hybrid_attention_statistical_comparison_metadata.json"
)


# ---------------------------------------------------------------------
# Experimental design
# ---------------------------------------------------------------------

DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

BASELINES = [
    "logistic_regression",
    "random_forest",
    "mlp",
]

TRAINING_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

SPLIT_SEED = 42

PRIMARY_METRICS = [
    "roc_auc",
    "pr_auc",
    "f1",
]

SUPPLEMENTARY_METRICS = [
    "accuracy",
    "precision",
    "recall",
]

ALL_METRICS = (
    PRIMARY_METRICS
    + SUPPLEMENTARY_METRICS
)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_source_files():
    if not HYBRID_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Hybrid result file not found:\n"
            f"{HYBRID_RESULTS_PATH}"
        )

    if not BASELINE_RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Baseline result file not found:\n"
            f"{BASELINE_RESULTS_PATH}"
        )


def validate_hybrid_results(df):
    required_columns = [
        "dataset",
        "training_seed",
        "split_seed",
    ]

    required_columns.extend(
        ALL_METRICS
    )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Hybrid results are missing columns:\n"
            + "\n".join(missing)
        )

    if len(df) != 25:
        raise ValueError(
            "Hybrid result file should contain "
            f"25 rows, but contains {len(df)}."
        )

    datasets = sorted(
        df["dataset"].unique()
    )

    if datasets != sorted(DATASETS):
        raise ValueError(
            "Unexpected Hybrid datasets.\n"
            f"Expected: {DATASETS}\n"
            f"Found: {datasets}"
        )

    seeds = sorted(
        df["training_seed"].unique()
    )

    if seeds != TRAINING_SEEDS:
        raise ValueError(
            "Unexpected Hybrid training seeds.\n"
            f"Expected: {TRAINING_SEEDS}\n"
            f"Found: {seeds}"
        )

    if not (
        df["split_seed"] == SPLIT_SEED
    ).all():
        raise ValueError(
            "Hybrid results contain a split seed "
            "other than 42."
        )

    counts = (
        df.groupby("dataset")[
            "training_seed"
        ]
        .count()
    )

    if not (
        counts == len(TRAINING_SEEDS)
    ).all():
        raise ValueError(
            "Every Hybrid dataset must contain "
            "exactly five training seeds."
        )

    if df[ALL_METRICS].isna().any().any():
        raise ValueError(
            "Hybrid results contain missing "
            "statistical metric values."
        )


def validate_baseline_results(df):
    required_columns = [
        "dataset",
        "model",
        "training_seed",
        "split_seed",
    ]

    for metric in ALL_METRICS:
        required_columns.append(
            f"test_{metric}"
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Baseline results are missing columns:\n"
            + "\n".join(missing)
        )

    if len(df) != 75:
        raise ValueError(
            "Baseline result file should contain "
            f"75 rows, but contains {len(df)}."
        )

    datasets = sorted(
        df["dataset"].unique()
    )

    if datasets != sorted(DATASETS):
        raise ValueError(
            "Unexpected baseline datasets.\n"
            f"Expected: {DATASETS}\n"
            f"Found: {datasets}"
        )

    models = sorted(
        df["model"].unique()
    )

    if models != sorted(BASELINES):
        raise ValueError(
            "Unexpected baseline models.\n"
            f"Expected: {BASELINES}\n"
            f"Found: {models}"
        )

    seeds = sorted(
        df["training_seed"].unique()
    )

    if seeds != TRAINING_SEEDS:
        raise ValueError(
            "Unexpected baseline training seeds.\n"
            f"Expected: {TRAINING_SEEDS}\n"
            f"Found: {seeds}"
        )

    if not (
        df["split_seed"] == SPLIT_SEED
    ).all():
        raise ValueError(
            "Baseline results contain a split seed "
            "other than 42."
        )

    counts = (
        df.groupby(
            ["dataset", "model"]
        )["training_seed"]
        .count()
    )

    if not (
        counts == len(TRAINING_SEEDS)
    ).all():
        raise ValueError(
            "Every dataset/model combination "
            "must contain exactly five training seeds."
        )

    metric_columns = [
        f"test_{metric}"
        for metric in ALL_METRICS
    ]

    if df[metric_columns].isna().any().any():
        raise ValueError(
            "Baseline results contain missing "
            "statistical metric values."
        )


# ---------------------------------------------------------------------
# Exact paired sign-flip permutation test
# ---------------------------------------------------------------------

def exact_sign_flip_pvalue(
    differences
):
    """
    Exact two-sided paired sign-flip permutation test.

    With five paired observations there are only
    2^5 = 32 possible sign assignments, so all
    permutations are evaluated exactly.

    Test statistic:
        absolute mean paired difference
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) == 0:
        return np.nan

    observed = abs(
        np.mean(differences)
    )

    permutation_statistics = []

    for signs in itertools.product(
        [-1.0, 1.0],
        repeat=len(differences),
    ):

        signed = (
            differences
            * np.asarray(signs)
        )

        permutation_statistics.append(
            abs(np.mean(signed))
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    p_value = (
        np.sum(
            permutation_statistics >= observed
        )
        / len(permutation_statistics)
    )

    return float(p_value)


# ---------------------------------------------------------------------
# Paired rank-biserial correlation
# ---------------------------------------------------------------------

def paired_rank_biserial(
    differences
):
    """
    Paired rank-biserial effect size.

    Positive values indicate that the Hybrid
    Attention model tends to have larger values
    than the baseline.

    Zero differences are excluded from ranking.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    differences = differences[
        np.isfinite(differences)
    ]

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:
        return 0.0

    absolute_values = np.abs(
        differences
    )

    ranks = pd.Series(
        absolute_values
    ).rank(
        method="average"
    ).to_numpy()

    positive_rank_sum = np.sum(
        ranks[differences > 0]
    )

    negative_rank_sum = np.sum(
        ranks[differences < 0]
    )

    total_rank_sum = (
        positive_rank_sum
        + negative_rank_sum
    )

    if total_rank_sum == 0:
        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / total_rank_sum
    )


# ---------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------

def holm_adjust(
    p_values
):
    """
    Holm-Bonferroni adjustment.

    Returns adjusted p-values in the original
    order of the supplied values.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    adjusted = np.full(
        len(p_values),
        np.nan,
        dtype=float
    )

    valid_indices = [
        i
        for i, p in enumerate(p_values)
        if np.isfinite(p)
    ]

    if not valid_indices:
        return adjusted

    ordered_indices = sorted(
        valid_indices,
        key=lambda i: p_values[i]
    )

    running_max = 0.0

    m = len(
        ordered_indices
    )

    for rank, index in enumerate(
        ordered_indices
    ):

        adjusted_value = (
            m - rank
        ) * p_values[index]

        running_max = max(
            running_max,
            adjusted_value
        )

        adjusted[index] = min(
            running_max,
            1.0
        )

    return adjusted


# ---------------------------------------------------------------------
# Statistical comparison for one paired comparison
# ---------------------------------------------------------------------

def calculate_comparison(
    hybrid_values,
    baseline_values,
):
    hybrid_values = np.asarray(
        hybrid_values,
        dtype=float
    )

    baseline_values = np.asarray(
        baseline_values,
        dtype=float
    )

    if len(hybrid_values) != len(
        baseline_values
    ):
        raise ValueError(
            "Hybrid and baseline arrays "
            "must have the same length."
        )

    differences = (
        hybrid_values
        - baseline_values
    )

    mean_difference = float(
        np.mean(differences)
    )

    sd_difference = float(
        np.std(
            differences,
            ddof=1
        )
    )

    # Exact Wilcoxon signed-rank test.
    #
    # Zero differences are omitted because they
    # contribute no directional information.
    try:
        wilcoxon_result = wilcoxon(
            differences,
            zero_method="wilcox",
            alternative="two-sided",
            method="exact",
        )

        wilcoxon_statistic = float(
            wilcoxon_result.statistic
        )

        wilcoxon_p = float(
            wilcoxon_result.pvalue
        )

    except ValueError:
        wilcoxon_statistic = np.nan
        wilcoxon_p = np.nan

    permutation_p = (
        exact_sign_flip_pvalue(
            differences
        )
    )

    effect_size = (
        paired_rank_biserial(
            differences
        )
    )

    return {
        "n_pairs": int(
            len(differences)
        ),
        "hybrid_mean": float(
            np.mean(hybrid_values)
        ),
        "baseline_mean": float(
            np.mean(baseline_values)
        ),
        "mean_difference_hybrid_minus_baseline":
            mean_difference,
        "sd_paired_difference":
            sd_difference,
        "wilcoxon_statistic":
            wilcoxon_statistic,
        "wilcoxon_p_value":
            wilcoxon_p,
        "exact_sign_flip_p_value":
            permutation_p,
        "paired_rank_biserial":
            effect_size,
    }


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "SEED-LEVEL STATISTICAL COMPARISON"
    )
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    validate_source_files()

    print(
        "Loading Hybrid Attention seed-level results..."
    )

    hybrid_df = pd.read_csv(
        HYBRID_RESULTS_PATH
    )

    print(
        "Loading baseline seed-level results..."
    )

    baseline_df = pd.read_csv(
        BASELINE_RESULTS_PATH
    )

    print()

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_hybrid_results(
        hybrid_df
    )

    validate_baseline_results(
        baseline_df
    )

    print(
        "Source validation: PASSED"
    )

    print(
        f"Hybrid rows: {len(hybrid_df)}"
    )

    print(
        f"Baseline rows: {len(baseline_df)}"
    )

    print()

    # -------------------------------------------------------------
    # Check seed alignment explicitly
    # -------------------------------------------------------------

    expected_seed_set = set(
        TRAINING_SEEDS
    )

    for dataset in DATASETS:

        hybrid_seeds = set(
            hybrid_df.loc[
                hybrid_df["dataset"] == dataset,
                "training_seed",
            ]
        )

        if hybrid_seeds != expected_seed_set:
            raise ValueError(
                f"Hybrid seed mismatch for {dataset}."
            )

        for model in BASELINES:

            baseline_seeds = set(
                baseline_df.loc[
                    (
                        baseline_df["dataset"]
                        == dataset
                    )
                    & (
                        baseline_df["model"]
                        == model
                    ),
                    "training_seed",
                ]
            )

            if (
                baseline_seeds
                != expected_seed_set
            ):
                raise ValueError(
                    f"Baseline seed mismatch for "
                    f"{dataset}/{model}."
                )

    print(
        "Seed alignment validation: PASSED"
    )

    print()

    # -------------------------------------------------------------
    # Build paired statistical comparisons
    # -------------------------------------------------------------

    records = []

    for dataset in DATASETS:

        hybrid_dataset = (
            hybrid_df[
                hybrid_df["dataset"]
                == dataset
            ]
            .sort_values(
                "training_seed"
            )
        )

        for model in BASELINES:

            baseline_dataset = (
                baseline_df[
                    (
                        baseline_df["dataset"]
                        == dataset
                    )
                    & (
                        baseline_df["model"]
                        == model
                    )
                ]
                .sort_values(
                    "training_seed"
                )
            )

            # Verify exact seed ordering.
            if not np.array_equal(
                hybrid_dataset[
                    "training_seed"
                ].to_numpy(),
                baseline_dataset[
                    "training_seed"
                ].to_numpy(),
            ):
                raise ValueError(
                    "Seed ordering mismatch for "
                    f"{dataset}/{model}."
                )

            for metric in ALL_METRICS:

                hybrid_values = (
                    hybrid_dataset[
                        metric
                    ].to_numpy(
                        dtype=float
                    )
                )

                baseline_values = (
                    baseline_dataset[
                        f"test_{metric}"
                    ].to_numpy(
                        dtype=float
                    )
                )

                statistics = (
                    calculate_comparison(
                        hybrid_values,
                        baseline_values,
                    )
                )

                record = {
                    "dataset": dataset,
                    "baseline_model": model,
                    "metric": metric,
                    "analysis_group": (
                        "primary"
                        if metric
                        in PRIMARY_METRICS
                        else "supplementary"
                    ),
                    "training_seeds": (
                        "42,43,44,45,46"
                    ),
                    "split_seed": SPLIT_SEED,
                }

                record.update(
                    statistics
                )

                records.append(
                    record
                )

    results_df = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------
    # Holm correction
    #
    # Correction is applied separately within
    # each metric across the 15 dataset/baseline
    # comparisons.
    # -------------------------------------------------------------

    results_df[
        "wilcoxon_p_holm_by_metric"
    ] = np.nan

    results_df[
        "sign_flip_p_holm_by_metric"
    ] = np.nan

    for metric in ALL_METRICS:

        mask = (
            results_df["metric"]
            == metric
        )

        wilcoxon_values = (
            results_df.loc[
                mask,
                "wilcoxon_p_value",
            ]
            .to_numpy()
        )

        sign_flip_values = (
            results_df.loc[
                mask,
                "exact_sign_flip_p_value",
            ]
            .to_numpy()
        )

        results_df.loc[
            mask,
            "wilcoxon_p_holm_by_metric",
        ] = holm_adjust(
            wilcoxon_values
        )

        results_df.loc[
            mask,
            "sign_flip_p_holm_by_metric",
        ] = holm_adjust(
            sign_flip_values
        )

    # -------------------------------------------------------------
    # Add directional interpretation
    #
    # This is descriptive, not a significance claim.
    # -------------------------------------------------------------

    def direction(value):

        if value > 0:
            return "hybrid_attention_higher"

        if value < 0:
            return "hybrid_attention_lower"

        return "equal"

    results_df[
        "observed_direction"
    ] = results_df[
        "mean_difference_hybrid_minus_baseline"
    ].apply(
        direction
    )

    # -------------------------------------------------------------
    # Significance flags
    # -------------------------------------------------------------

    results_df[
        "wilcoxon_holm_significant_alpha_0.05"
    ] = (
        results_df[
            "wilcoxon_p_holm_by_metric"
        ]
        < 0.05
    )

    results_df[
        "sign_flip_holm_significant_alpha_0.05"
    ] = (
        results_df[
            "sign_flip_p_holm_by_metric"
        ]
        < 0.05
    )

    # -------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------

    results_df = (
        results_df
        .sort_values(
            [
                "metric",
                "dataset",
                "baseline_model",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Save full statistical table
    # -------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Long-format manuscript-friendly table
    # -------------------------------------------------------------

    long_columns = [
        "dataset",
        "baseline_model",
        "metric",
        "analysis_group",
        "n_pairs",
        "hybrid_mean",
        "baseline_mean",
        "mean_difference_hybrid_minus_baseline",
        "sd_paired_difference",
        "wilcoxon_statistic",
        "wilcoxon_p_value",
        "wilcoxon_p_holm_by_metric",
        "exact_sign_flip_p_value",
        "sign_flip_p_holm_by_metric",
        "paired_rank_biserial",
        "observed_direction",
    ]

    long_df = results_df[
        long_columns
    ].copy()

    long_df.to_csv(
        OUTPUT_LONG_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------

    metadata = {
        "experiment": (
            "seed_level_statistical_comparison"
        ),
        "datasets": DATASETS,
        "baseline_models": BASELINES,
        "proposed_model": "hybrid_attention",
        "training_seeds": TRAINING_SEEDS,
        "n_training_seeds": len(
            TRAINING_SEEDS
        ),
        "split_seed": SPLIT_SEED,
        "pairing": (
            "same dataset and corresponding "
            "training seed across models"
        ),
        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),
        "scaling_protocol": (
            "fit_train_only"
        ),
        "test_usage": (
            "final_evaluation_only"
        ),
        "primary_metrics": PRIMARY_METRICS,
        "supplementary_metrics": (
            SUPPLEMENTARY_METRICS
        ),
        "primary_statistical_test": (
            "paired Wilcoxon signed-rank test"
        ),
        "secondary_exact_test": (
            "exact paired sign-flip permutation "
            "test using all 2^5 sign assignments"
        ),
        "effect_size": (
            "paired rank-biserial correlation"
        ),
        "multiple_comparison_correction": (
            "Holm-Bonferroni adjustment separately "
            "within each metric across 15 "
            "dataset-baseline comparisons"
        ),
        "alpha": 0.05,
        "note": (
            "Statistical inference uses the five "
            "seed-level observations per paired "
            "dataset/model comparison, not aggregated "
            "means or standard deviations."
        ),
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
    # Console summary
    # -------------------------------------------------------------

    print(
        "Statistical comparison validation: PASSED"
    )

    print()

    print(
        f"TOTAL STATISTICAL COMPARISONS: "
        f"{len(results_df)}"
    )

    print(
        "Expected: "
        f"{len(DATASETS)} datasets × "
        f"{len(BASELINES)} baselines × "
        f"{len(ALL_METRICS)} metrics = "
        f"{len(DATASETS) * len(BASELINES) * len(ALL_METRICS)}"
    )

    print()

    print(
        "PRIMARY METRIC SUMMARY:"
    )

    print(
        results_df[
            results_df["metric"].isin(
                PRIMARY_METRICS
            )
        ][
            [
                "dataset",
                "baseline_model",
                "metric",
                "hybrid_mean",
                "baseline_mean",
                "mean_difference_hybrid_minus_baseline",
                "wilcoxon_p_value",
                "wilcoxon_p_holm_by_metric",
                "exact_sign_flip_p_value",
                "sign_flip_p_holm_by_metric",
                "paired_rank_biserial",
                "observed_direction",
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
        "STATISTICAL COMPARISON COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
