
from pathlib import Path
import json
import itertools

import numpy as np
import pandas as pd
from scipy.stats import t


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

STATISTICAL_RESULTS_PATH = (
    RESULTS_DIR
    / "hybrid_attention_statistical_comparison.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "hybrid_attention_effect_size_confidence_analysis.csv"
)

OUTPUT_LONG_PATH = (
    RESULTS_DIR
    / "hybrid_attention_effect_size_confidence_analysis_long.csv"
)

METADATA_PATH = (
    RESULTS_DIR
    / "hybrid_attention_effect_size_confidence_analysis_metadata.json"
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

def validate_files():
    required_files = [
        HYBRID_RESULTS_PATH,
        BASELINE_RESULTS_PATH,
        STATISTICAL_RESULTS_PATH,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file was not found:\n{path}"
            )


def validate_hybrid(df):

    required = [
        "dataset",
        "training_seed",
        "split_seed",
    ]

    required += ALL_METRICS

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Hybrid results are missing columns:\n"
            + "\n".join(missing)
        )

    if len(df) != 25:

        raise ValueError(
            "Expected 25 Hybrid Attention rows, "
            f"found {len(df)}."
        )

    if sorted(df["dataset"].unique()) != sorted(DATASETS):

        raise ValueError(
            "Unexpected Hybrid datasets."
        )

    if sorted(
        df["training_seed"].unique()
    ) != TRAINING_SEEDS:

        raise ValueError(
            "Unexpected Hybrid training seeds."
        )

    if not (
        df["split_seed"] == SPLIT_SEED
    ).all():

        raise ValueError(
            "Hybrid results contain a split seed "
            "other than 42."
        )

    if df[ALL_METRICS].isna().any().any():

        raise ValueError(
            "Hybrid results contain missing metric values."
        )


def validate_baseline(df):

    required = [
        "dataset",
        "model",
        "training_seed",
        "split_seed",
    ]

    required += [
        f"test_{metric}"
        for metric in ALL_METRICS
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            "Baseline results are missing columns:\n"
            + "\n".join(missing)
        )

    if len(df) != 75:

        raise ValueError(
            "Expected 75 baseline rows, "
            f"found {len(df)}."
        )

    if sorted(df["dataset"].unique()) != sorted(DATASETS):

        raise ValueError(
            "Unexpected baseline datasets."
        )

    if sorted(df["model"].unique()) != sorted(BASELINES):

        raise ValueError(
            "Unexpected baseline models."
        )

    if sorted(
        df["training_seed"].unique()
    ) != TRAINING_SEEDS:

        raise ValueError(
            "Unexpected baseline training seeds."
        )

    if not (
        df["split_seed"] == SPLIT_SEED
    ).all():

        raise ValueError(
            "Baseline results contain a split seed "
            "other than 42."
        )


# ---------------------------------------------------------------------
# Effect size calculations
# ---------------------------------------------------------------------

def paired_cohens_d(differences):
    """
    Cohen's d for paired observations.

    d = mean(differences) / SD(differences)

    Positive values indicate a higher Hybrid Attention
    metric value than the corresponding baseline.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    mean_difference = np.mean(
        differences
    )

    sd_difference = np.std(
        differences,
        ddof=1
    )

    if sd_difference == 0:

        if mean_difference > 0:
            return np.inf

        if mean_difference < 0:
            return -np.inf

        return 0.0

    return float(
        mean_difference
        / sd_difference
    )


def confidence_interval_mean_difference(
    differences,
    confidence_level=0.95,
):
    """
    Student-t confidence interval for the paired
    mean difference.

    The calculation is based on the five paired
    seed-level differences.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    n = len(differences)

    mean_difference = np.mean(
        differences
    )

    if n < 2:

        return (
            np.nan,
            np.nan,
        )

    sd_difference = np.std(
        differences,
        ddof=1
    )

    if sd_difference == 0:

        return (
            float(mean_difference),
            float(mean_difference),
        )

    standard_error = (
        sd_difference
        / np.sqrt(n)
    )

    alpha = (
        1.0
        - confidence_level
    )

    critical_value = t.ppf(
        1.0 - alpha / 2.0,
        df=n - 1,
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        float(
            mean_difference - margin
        ),
        float(
            mean_difference + margin
        ),
    )


def exact_sign_flip_pvalue(
    differences
):
    """
    Exact two-sided paired sign-flip permutation
    test.

    With five seeds there are exactly 32 possible
    sign assignments.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

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

    return float(
        np.mean(
            permutation_statistics
            >= observed
        )
    )


def paired_rank_biserial(
    differences
):
    """
    Paired rank-biserial effect size.

    Positive:
        Hybrid Attention tends to be higher.

    Negative:
        Hybrid Attention tends to be lower.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    differences = differences[
        differences != 0
    ]

    if len(differences) == 0:

        return 0.0

    ranks = pd.Series(
        np.abs(differences)
    ).rank(
        method="average"
    ).to_numpy()

    positive_rank_sum = np.sum(
        ranks[differences > 0]
    )

    negative_rank_sum = np.sum(
        ranks[differences < 0]
    )

    denominator = (
        positive_rank_sum
        + negative_rank_sum
    )

    if denominator == 0:

        return 0.0

    return float(
        (
            positive_rank_sum
            - negative_rank_sum
        )
        / denominator
    )


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "EFFECT SIZE AND CONFIDENCE INTERVAL ANALYSIS"
    )
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # Load files
    # -------------------------------------------------------------

    validate_files()

    print(
        "Loading Hybrid Attention results..."
    )

    hybrid_df = pd.read_csv(
        HYBRID_RESULTS_PATH
    )

    print(
        "Loading baseline results..."
    )

    baseline_df = pd.read_csv(
        BASELINE_RESULTS_PATH
    )

    print(
        "Loading statistical comparison results..."
    )

    statistical_df = pd.read_csv(
        STATISTICAL_RESULTS_PATH
    )

    print()

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_hybrid(
        hybrid_df
    )

    validate_baseline(
        baseline_df
    )

    print(
        "Source validation: PASSED"
    )

    # -------------------------------------------------------------
    # Verify statistical comparison size
    # -------------------------------------------------------------

    expected_statistical_rows = (
        len(DATASETS)
        * len(BASELINES)
        * len(ALL_METRICS)
    )

    if len(statistical_df) != (
        expected_statistical_rows
    ):

        raise ValueError(
            "Statistical comparison file should "
            f"contain {expected_statistical_rows} rows, "
            f"found {len(statistical_df)}."
        )

    print(
        "Statistical comparison validation: PASSED"
    )

    print()

    # -------------------------------------------------------------
    # Calculate effect sizes and CIs
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

            # Exact seed alignment.
            hybrid_seeds = (
                hybrid_dataset[
                    "training_seed"
                ].to_numpy()
            )

            baseline_seeds = (
                baseline_dataset[
                    "training_seed"
                ].to_numpy()
            )

            if not np.array_equal(
                hybrid_seeds,
                baseline_seeds,
            ):

                raise ValueError(
                    "Training seed alignment failed "
                    f"for {dataset}/{model}."
                )

            for metric in ALL_METRICS:

                hybrid_values = (
                    hybrid_dataset[
                        metric
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                baseline_values = (
                    baseline_dataset[
                        f"test_{metric}"
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                differences = (
                    hybrid_values
                    - baseline_values
                )

                mean_difference = (
                    np.mean(differences)
                )

                sd_difference = (
                    np.std(
                        differences,
                        ddof=1
                    )
                )

                ci_lower, ci_upper = (
                    confidence_interval_mean_difference(
                        differences
                    )
                )

                d_paired = (
                    paired_cohens_d(
                        differences
                    )
                )

                rank_biserial = (
                    paired_rank_biserial(
                        differences
                    )
                )

                sign_flip_p = (
                    exact_sign_flip_pvalue(
                        differences
                    )
                )

                # Retrieve the already calculated
                # Wilcoxon statistics.
                matching = statistical_df[
                    (
                        statistical_df["dataset"]
                        == dataset
                    )
                    & (
                        statistical_df[
                            "baseline_model"
                        ]
                        == model
                    )
                    & (
                        statistical_df["metric"]
                        == metric
                    )
                ]

                if len(matching) != 1:

                    raise ValueError(
                        "Could not uniquely match "
                        "statistical comparison for "
                        f"{dataset}/{model}/{metric}."
                    )

                statistical_row = (
                    matching.iloc[0]
                )

                records.append(
                    {
                        "dataset": dataset,
                        "baseline_model": model,
                        "metric": metric,
                        "analysis_group": (
                            "primary"
                            if metric
                            in PRIMARY_METRICS
                            else "supplementary"
                        ),
                        "n_pairs": len(
                            differences
                        ),
                        "training_seeds": (
                            "42,43,44,45,46"
                        ),
                        "split_seed": SPLIT_SEED,
                        "hybrid_mean": np.mean(
                            hybrid_values
                        ),
                        "baseline_mean": np.mean(
                            baseline_values
                        ),
                        "mean_difference_hybrid_minus_baseline":
                            mean_difference,
                        "sd_paired_difference":
                            sd_difference,
                        "ci95_lower":
                            ci_lower,
                        "ci95_upper":
                            ci_upper,
                        "paired_cohens_d":
                            d_paired,
                        "paired_rank_biserial":
                            rank_biserial,
                        "wilcoxon_statistic":
                            statistical_row[
                                "wilcoxon_statistic"
                            ],
                        "wilcoxon_p_value":
                            statistical_row[
                                "wilcoxon_p_value"
                            ],
                        "wilcoxon_p_holm_by_metric":
                            statistical_row[
                                "wilcoxon_p_holm_by_metric"
                            ],
                        "exact_sign_flip_p_value":
                            sign_flip_p,
                        "sign_flip_p_holm_by_metric":
                            statistical_row[
                                "sign_flip_p_holm_by_metric"
                            ],
                        "observed_direction": (
                            "hybrid_attention_higher"
                            if mean_difference > 0
                            else (
                                "hybrid_attention_lower"
                                if mean_difference < 0
                                else "equal"
                            )
                        ),
                    }
                )

    results_df = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------
    # Validate output
    # -------------------------------------------------------------

    expected_rows = (
        len(DATASETS)
        * len(BASELINES)
        * len(ALL_METRICS)
    )

    if len(results_df) != expected_rows:

        raise ValueError(
            "Effect-size analysis should contain "
            f"{expected_rows} rows, "
            f"found {len(results_df)}."
        )

    # -------------------------------------------------------------
    # Save complete table
    # -------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # -------------------------------------------------------------
    # Manuscript-oriented long table
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
        "ci95_lower",
        "ci95_upper",
        "paired_cohens_d",
        "paired_rank_biserial",
        "wilcoxon_statistic",
        "wilcoxon_p_value",
        "wilcoxon_p_holm_by_metric",
        "exact_sign_flip_p_value",
        "sign_flip_p_holm_by_metric",
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
            "paired_effect_size_and_confidence_analysis"
        ),
        "datasets": DATASETS,
        "baseline_models": BASELINES,
        "training_seeds": TRAINING_SEEDS,
        "n_pairs_per_comparison": len(
            TRAINING_SEEDS
        ),
        "split_seed": SPLIT_SEED,
        "primary_metrics": PRIMARY_METRICS,
        "supplementary_metrics": (
            SUPPLEMENTARY_METRICS
        ),
        "paired_difference_definition": (
            "Hybrid Attention minus baseline"
        ),
        "confidence_interval": (
            "95 percent Student-t confidence interval "
            "for the mean paired difference"
        ),
        "effect_size_1": (
            "paired Cohen d"
        ),
        "effect_size_2": (
            "paired rank-biserial correlation"
        ),
        "statistical_tests_reused": [
            "paired Wilcoxon signed-rank test",
            "exact paired sign-flip permutation test",
        ],
        "multiple_comparison_correction": (
            "Holm-Bonferroni adjustment within metric"
        ),
        "test_data_usage": (
            "existing locked test results only; "
            "no retraining or test-based model selection"
        ),
        "important_limitation": (
            "Only five paired training seeds are available "
            "per comparison; confidence intervals and "
            "inferential results should therefore be "
            "interpreted conservatively."
        ),
        "output_rows": len(
            results_df
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
    # Console output
    # -------------------------------------------------------------

    print(
        "Effect-size analysis validation: PASSED"
    )

    print()

    print(
        f"TOTAL EFFECT-SIZE ROWS: "
        f"{len(results_df)}"
    )

    print()

    print(
        "PRIMARY METRIC EFFECT-SIZE SUMMARY:"
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
                "mean_difference_hybrid_minus_baseline",
                "ci95_lower",
                "ci95_upper",
                "paired_cohens_d",
                "paired_rank_biserial",
                "wilcoxon_p_holm_by_metric",
                "sign_flip_p_holm_by_metric",
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
        "EFFECT-SIZE ANALYSIS COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
