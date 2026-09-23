from pathlib import Path
import itertools
import json

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "results"
)

ABLATION_RESULTS_PATH = (
    RESULTS_DIR
    / "ablation_repeated_seed_results.csv"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ablation_statistical_comparison.csv"
)

OUTPUT_LONG_PATH = (
    RESULTS_DIR
    / "ablation_statistical_comparison_long.csv"
)

METADATA_PATH = (
    RESULTS_DIR
    / "ablation_statistical_comparison_metadata.json"
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

REFERENCE_VARIANT = (
    "A0_full_hybrid_attention"
)

ABLATION_VARIANTS = [
    "A1_no_self_attention",
    "A2_no_semantic_tokenization",
    "A3_single_head_attention",
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
# Descriptive labels
# ---------------------------------------------------------------------

ABLATION_DESCRIPTIONS = {
    "A1_no_self_attention": (
        "Self-attention removed"
    ),
    "A2_no_semantic_tokenization": (
        "Semantic feature-group tokenization removed"
    ),
    "A3_single_head_attention": (
        "Four attention heads changed to one head"
    ),
}


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_source_file():

    if not ABLATION_RESULTS_PATH.exists():

        raise FileNotFoundError(
            "Ablation result file not found:\n"
            f"{ABLATION_RESULTS_PATH}"
        )


def validate_results(df):

    required_columns = [
        "variant",
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
            "Ablation results are missing columns:\n"
            + "\n".join(missing)
        )

    expected_variants = [
        REFERENCE_VARIANT,
        *ABLATION_VARIANTS,
    ]

    variants = sorted(
        df["variant"].astype(str).unique()
    )

    if variants != sorted(
        expected_variants
    ):

        raise ValueError(
            "Unexpected ablation variants.\n"
            f"Expected: {expected_variants}\n"
            f"Found: {variants}"
        )

    datasets = sorted(
        df["dataset"].astype(str).unique()
    )

    if datasets != sorted(DATASETS):

        raise ValueError(
            "Unexpected datasets.\n"
            f"Expected: {DATASETS}\n"
            f"Found: {datasets}"
        )

    seeds = sorted(
        df["training_seed"].unique()
    )

    if seeds != TRAINING_SEEDS:

        raise ValueError(
            "Unexpected training seeds.\n"
            f"Expected: {TRAINING_SEEDS}\n"
            f"Found: {seeds}"
        )

    if not (
        df["split_seed"] == SPLIT_SEED
    ).all():

        raise ValueError(
            "Ablation results contain a split seed "
            "other than 42."
        )

    expected_rows = (
        len(expected_variants)
        * len(DATASETS)
        * len(TRAINING_SEEDS)
    )

    if len(df) != expected_rows:

        raise ValueError(
            "Ablation result file should contain "
            f"{expected_rows} rows, "
            f"but contains {len(df)}."
        )

    duplicate_keys = (
        df.groupby(
            [
                "variant",
                "dataset",
                "training_seed",
            ]
        )
        .size()
    )

    duplicate_keys = duplicate_keys[
        duplicate_keys > 1
    ]

    if len(duplicate_keys) > 0:

        raise ValueError(
            "Duplicate variant/dataset/training_seed "
            "records were found:\n"
            f"{duplicate_keys}"
        )

    if df[ALL_METRICS].isna().any().any():

        raise ValueError(
            "Ablation results contain missing "
            "statistical metric values."
        )

    # -------------------------------------------------------------
    # Verify every variant/dataset combination has five seeds.
    # -------------------------------------------------------------

    counts = (
        df.groupby(
            [
                "variant",
                "dataset",
            ]
        )["training_seed"]
        .count()
    )

    if not (
        counts == len(TRAINING_SEEDS)
    ).all():

        raise ValueError(
            "Every variant/dataset combination must "
            "contain exactly five training seeds."
        )


# ---------------------------------------------------------------------
# Exact paired sign-flip permutation test
# ---------------------------------------------------------------------

def exact_sign_flip_pvalue(
    differences
):

    """
    Exact two-sided paired sign-flip permutation test.

    All possible sign assignments are evaluated.

    With five paired observations:

        2^5 = 32

    possible sign assignments exist.

    Difference convention:

        ablation - reference

    Therefore:

        negative difference
            -> ablation metric is lower

        positive difference
            -> ablation metric is higher
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
            abs(
                np.mean(signed)
            )
        )

    permutation_statistics = np.asarray(
        permutation_statistics
    )

    p_value = (
        np.sum(
            permutation_statistics
            >= observed
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

    Positive:
        ablation tends to have higher values
        than the full Hybrid Attention model.

    Negative:
        ablation tends to have lower values
        than the full Hybrid Attention model.

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
# Paired Cohen's d
# ---------------------------------------------------------------------

def paired_cohens_d(
    differences
):

    """
    Cohen's d for paired observations.

        d = mean(difference) / SD(difference)

    Difference convention:

        ablation - reference

    Positive:
        ablation higher

    Negative:
        ablation lower
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


# ---------------------------------------------------------------------
# 95% Student-t confidence interval
# ---------------------------------------------------------------------

def confidence_interval_mean_difference(
    differences,
    confidence_level=0.95,
):

    """
    Student-t confidence interval for the paired
    mean difference.

    The interval is based on the five paired
    training-seed differences.
    """

    differences = np.asarray(
        differences,
        dtype=float
    )

    differences = differences[
        np.isfinite(differences)
    ]

    n = len(differences)

    if n < 2:

        return (
            np.nan,
            np.nan,
        )

    mean_difference = np.mean(
        differences
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


# ---------------------------------------------------------------------
# Calculate one paired comparison
# ---------------------------------------------------------------------

def calculate_comparison(
    reference_values,
    ablation_values,
):

    reference_values = np.asarray(
        reference_values,
        dtype=float
    )

    ablation_values = np.asarray(
        ablation_values,
        dtype=float
    )

    if len(reference_values) != len(
        ablation_values
    ):

        raise ValueError(
            "Reference and ablation arrays "
            "must have the same length."
        )

    differences = (
        ablation_values
        - reference_values
    )

    if not np.isfinite(
        differences
    ).all():

        raise ValueError(
            "Non-finite paired differences detected."
        )

    n_pairs = len(
        differences
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

    # -------------------------------------------------------------
    # Exact Wilcoxon signed-rank test
    # -------------------------------------------------------------

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

    # -------------------------------------------------------------
    # Exact paired sign-flip test
    # -------------------------------------------------------------

    sign_flip_p = (
        exact_sign_flip_pvalue(
            differences
        )
    )

    # -------------------------------------------------------------
    # Effect sizes
    # -------------------------------------------------------------

    rank_biserial = (
        paired_rank_biserial(
            differences
        )
    )

    cohens_d = (
        paired_cohens_d(
            differences
        )
    )

    # -------------------------------------------------------------
    # Confidence interval
    # -------------------------------------------------------------

    ci_lower, ci_upper = (
        confidence_interval_mean_difference(
            differences
        )
    )

    return {
        "n_pairs": int(
            n_pairs
        ),
        "reference_mean": float(
            np.mean(reference_values)
        ),
        "ablation_mean": float(
            np.mean(ablation_values)
        ),
        "mean_difference_ablation_minus_reference":
            mean_difference,
        "sd_paired_difference":
            sd_difference,
        "ci95_lower":
            ci_lower,
        "ci95_upper":
            ci_upper,
        "paired_cohens_d":
            cohens_d,
        "paired_rank_biserial":
            rank_biserial,
        "wilcoxon_statistic":
            wilcoxon_statistic,
        "wilcoxon_p_value":
            wilcoxon_p,
        "exact_sign_flip_p_value":
            sign_flip_p,
    }


# ---------------------------------------------------------------------
# Direction
# ---------------------------------------------------------------------

def observed_direction(
    value
):

    if value > 0:

        return (
            "ablation_higher"
        )

    if value < 0:

        return (
            "ablation_lower"
        )

    return "equal"


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def main():

    print("=" * 70)
    print(
        "ABLATION SEED-LEVEL STATISTICAL ANALYSIS"
    )
    print("=" * 70)
    print()

    # -------------------------------------------------------------
    # Load
    # -------------------------------------------------------------

    validate_source_file()

    print(
        "Loading ablation seed-level results..."
    )

    df = pd.read_csv(
        ABLATION_RESULTS_PATH
    )

    print()

    # -------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------

    validate_results(
        df
    )

    print(
        "Source validation: PASSED"
    )

    print(
        f"Rows: {len(df)}"
    )

    print(
        f"Variants: {df['variant'].unique().tolist()}"
    )

    print(
        f"Datasets: {df['dataset'].unique().tolist()}"
    )

    print(
        f"Training seeds: "
        f"{sorted(df['training_seed'].unique())}"
    )

    print(
        f"Split seed: "
        f"{sorted(df['split_seed'].unique())}"
    )

    print()

    # -------------------------------------------------------------
    # Explicit seed alignment validation
    # -------------------------------------------------------------

    expected_seed_set = set(
        TRAINING_SEEDS
    )

    for dataset in DATASETS:

        reference_seeds = set(
            df.loc[
                (
                    df["variant"]
                    == REFERENCE_VARIANT
                )
                & (
                    df["dataset"]
                    == dataset
                ),
                "training_seed",
            ]
        )

        if reference_seeds != (
            expected_seed_set
        ):

            raise ValueError(
                "Reference seed mismatch for "
                f"{dataset}."
            )

        for variant in ABLATION_VARIANTS:

            ablation_seeds = set(
                df.loc[
                    (
                        df["variant"]
                        == variant
                    )
                    & (
                        df["dataset"]
                        == dataset
                    ),
                    "training_seed",
                ]
            )

            if ablation_seeds != (
                expected_seed_set
            ):

                raise ValueError(
                    "Ablation seed mismatch for "
                    f"{dataset}/{variant}."
                )

    print(
        "Seed alignment validation: PASSED"
    )

    print()

    # -------------------------------------------------------------
    # Build paired comparisons
    # -------------------------------------------------------------

    records = []

    for dataset in DATASETS:

        reference_dataset = (
            df[
                (
                    df["variant"]
                    == REFERENCE_VARIANT
                )
                & (
                    df["dataset"]
                    == dataset
                )
            ]
            .sort_values(
                "training_seed"
            )
        )

        for variant in ABLATION_VARIANTS:

            ablation_dataset = (
                df[
                    (
                        df["variant"]
                        == variant
                    )
                    & (
                        df["dataset"]
                        == dataset
                    )
                ]
                .sort_values(
                    "training_seed"
                )
            )

            # -----------------------------------------------------
            # Verify exact seed ordering.
            # -----------------------------------------------------

            reference_seeds = (
                reference_dataset[
                    "training_seed"
                ].to_numpy()
            )

            ablation_seeds = (
                ablation_dataset[
                    "training_seed"
                ].to_numpy()
            )

            if not np.array_equal(
                reference_seeds,
                ablation_seeds,
            ):

                raise ValueError(
                    "Training seed ordering mismatch "
                    f"for {dataset}/{variant}."
                )

            for metric in ALL_METRICS:

                reference_values = (
                    reference_dataset[
                        metric
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                ablation_values = (
                    ablation_dataset[
                        metric
                    ]
                    .to_numpy(
                        dtype=float
                    )
                )

                statistics = (
                    calculate_comparison(
                        reference_values,
                        ablation_values,
                    )
                )

                record = {
                    "dataset": dataset,
                    "reference_variant": (
                        REFERENCE_VARIANT
                    ),
                    "ablation_variant": variant,
                    "ablation_description": (
                        ABLATION_DESCRIPTIONS[
                            variant
                        ]
                    ),
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

                record[
                    "observed_direction"
                ] = observed_direction(
                    statistics[
                        "mean_difference_ablation_minus_reference"
                    ]
                )

                records.append(
                    record
                )

    results_df = pd.DataFrame(
        records
    )

    # -------------------------------------------------------------
    # Validate expected comparison count
    #
    # 5 datasets
    # Ã— 3 ablations
    # Ã— 6 metrics
    # = 90 comparisons
    # -------------------------------------------------------------

    expected_rows = (
        len(DATASETS)
        * len(ABLATION_VARIANTS)
        * len(ALL_METRICS)
    )

    if len(results_df) != (
        expected_rows
    ):

        raise ValueError(
            "Unexpected statistical comparison "
            f"row count. Expected {expected_rows}, "
            f"found {len(results_df)}."
        )

    print(
        "Comparison construction validation: PASSED"
    )

    print(
        f"Total paired comparisons: "
        f"{len(results_df)}"
    )

    print()

    # -------------------------------------------------------------
    # Holm correction
    #
    # Correction is applied separately within
    # each metric across the 15 ablation/dataset
    # comparisons:
    #
    # 5 datasets Ã— 3 ablations = 15.
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

        # ---------------------------------------------------------
        # Holm-Bonferroni correction.
        # ---------------------------------------------------------

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
                "ablation_variant",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Save complete statistical table
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
        "reference_variant",
        "ablation_variant",
        "ablation_description",
        "metric",
        "analysis_group",
        "n_pairs",
        "reference_mean",
        "ablation_mean",
        "mean_difference_ablation_minus_reference",
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
            "paired_ablation_seed_level_statistical_analysis"
        ),
        "reference_variant": (
            REFERENCE_VARIANT
        ),
        "ablation_variants": (
            ABLATION_VARIANTS
        ),
        "ablation_descriptions": (
            ABLATION_DESCRIPTIONS
        ),
        "datasets": DATASETS,
        "training_seeds": TRAINING_SEEDS,
        "n_training_seeds": len(
            TRAINING_SEEDS
        ),
        "split_seed": SPLIT_SEED,
        "primary_metrics": (
            PRIMARY_METRICS
        ),
        "supplementary_metrics": (
            SUPPLEMENTARY_METRICS
        ),
        "paired_difference_definition": (
            "ablation minus full Hybrid Attention"
        ),
        "statistical_tests": [
            "paired Wilcoxon signed-rank test",
            "exact paired sign-flip permutation test",
        ],
        "exact_sign_flip_permutations": (
            "all 2^5 = 32 sign assignments"
        ),
        "effect_size_1": (
            "paired Cohen d"
        ),
        "effect_size_2": (
            "paired rank-biserial correlation"
        ),
        "confidence_interval": (
            "95 percent Student-t confidence interval "
            "for the mean paired difference"
        ),
        "multiple_comparison_correction": (
            "Holm-Bonferroni adjustment separately "
            "within each metric across 15 "
            "dataset-ablation comparisons"
        ),
        "comparisons_per_metric": (
            len(DATASETS)
            * len(ABLATION_VARIANTS)
        ),
        "alpha": 0.05,
        "pairing": (
            "same dataset and corresponding training "
            "seed between A0 and each ablation"
        ),
        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),
        "scaling_protocol": (
            "fit_train_only"
        ),
        "threshold_protocol": (
            "fixed threshold 0.50"
        ),
        "test_usage": (
            "final evaluation results only; "
            "no threshold tuning or model selection "
            "performed using these ablation comparisons"
        ),
        "inference_unit": (
            "five paired training-seed observations "
            "per dataset-ablation comparison"
        ),
        "important_limitation": (
            "Only five paired training seeds are available "
            "per comparison. Exact permutation p-values, "
            "confidence intervals, and effect sizes should "
            "therefore be interpreted conservatively."
        ),
        "zero_f1_interpretation": (
            "Zero F1 is retained as an observed fixed-"
            "threshold result and is not treated as a "
            "missing value or analysis failure."
        ),
        "output_rows": int(
            len(results_df)
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
        "Statistical analysis validation: PASSED"
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
                "ablation_variant",
                "metric",
                "reference_mean",
                "ablation_mean",
                "mean_difference_ablation_minus_reference",
                "ci95_lower",
                "ci95_upper",
                "paired_cohens_d",
                "paired_rank_biserial",
                "wilcoxon_p_value",
                "wilcoxon_p_holm_by_metric",
                "exact_sign_flip_p_value",
                "sign_flip_p_holm_by_metric",
                "observed_direction",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print()

    print(
        "SIGNIFICANCE COUNTS:"
    )

    print(
        "Wilcoxon Holm-significant "
        f"(alpha=0.05): "
        f"{int(results_df['wilcoxon_holm_significant_alpha_0.05'].sum())}"
    )

    print(
        "Exact sign-flip Holm-significant "
        f"(alpha=0.05): "
        f"{int(results_df['sign_flip_holm_significant_alpha_0.05'].sum())}"
    )

    print()

    print(
        "ZERO-F1 COMPARISONS:"
    )

    zero_f1 = results_df[
        results_df["metric"] == "f1"
    ]

    zero_f1 = zero_f1[
        (
            zero_f1["reference_mean"] == 0
        )
        |
        (
            zero_f1["ablation_mean"] == 0
        )
    ]

    if len(zero_f1) == 0:

        print(
            "None."
        )

    else:

        print(
            zero_f1[
                [
                    "dataset",
                    "ablation_variant",
                    "reference_mean",
                    "ablation_mean",
                    "mean_difference_ablation_minus_reference",
                ]
            ]
            .to_string(
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
        "ABLATION STATISTICAL ANALYSIS COMPLETE"
    )
    print("=" * 70)


# ---------------------------------------------------------------------
# Holm-Bonferroni correction
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
# Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()
