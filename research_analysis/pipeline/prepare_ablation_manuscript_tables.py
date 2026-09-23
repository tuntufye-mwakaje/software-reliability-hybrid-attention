from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(r"D:\Projects\software-reliability-hybrid-attention")

RESULTS_DIR = PROJECT_ROOT / "research_analysis" / "results"

AGGREGATE_PATH = RESULTS_DIR / "ablation_repeated_seed_aggregate.csv"
STATS_PATH = RESULTS_DIR / "ablation_statistical_comparison.csv"
ZERO_F1_PATH = RESULTS_DIR / "ablation_zero_f1_diagnostics.csv"

OUTPUT_TABLE = RESULTS_DIR / "ablation_manuscript_primary_table.csv"
OUTPUT_STATS = RESULTS_DIR / "ablation_manuscript_statistical_table.csv"
OUTPUT_ZERO = RESULTS_DIR / "ablation_manuscript_zero_f1_table.csv"
OUTPUT_SUMMARY = RESULTS_DIR / "ablation_manuscript_component_summary.csv"
OUTPUT_METADATA = RESULTS_DIR / "ablation_manuscript_tables_metadata.json"


# ============================================================
# CONSTANTS
# ============================================================

VARIANT_ORDER = [
    "A0_full_hybrid_attention",
    "A1_no_self_attention",
    "A2_no_semantic_tokenization",
    "A3_single_head_attention",
]

VARIANT_LABELS = {
    "A0_full_hybrid_attention": "A0 Full Hybrid Attention",
    "A1_no_self_attention": "A1 No Self-Attention",
    "A2_no_semantic_tokenization": "A2 No Semantic Tokenization",
    "A3_single_head_attention": "A3 Single-Head Attention",
}

DATASET_ORDER = ["CM1", "JM1", "KC1", "KC2", "PC1"]

PRIMARY_METRICS = ["roc_auc", "pr_auc", "f1"]

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "f1": "F1",
}


# ============================================================
# HELPERS
# ============================================================

def check_exists(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def variant_label(variant):
    return VARIANT_LABELS.get(variant, variant)


def build_primary_performance_table(aggregate):
    """
    Convert the validated wide-format aggregate file into a
    manuscript-friendly long-format table.
    """

    rows = []

    for _, record in aggregate.iterrows():

        for metric in PRIMARY_METRICS:

            mean_column = f"{metric}_mean"
            sd_column = f"{metric}_sd"

            if mean_column not in aggregate.columns:
                raise ValueError(
                    f"Missing aggregate column: {mean_column}"
                )

            if sd_column not in aggregate.columns:
                raise ValueError(
                    f"Missing aggregate column: {sd_column}"
                )

            mean_value = float(record[mean_column])
            sd_value = float(record[sd_column])

            rows.append({
                "dataset": record["dataset"],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "variant": record["variant"],
                "variant_label": variant_label(record["variant"]),
                "mean": mean_value,
                "sd": sd_value,
                "mean_sd": f"{mean_value:.6f} ± {sd_value:.6f}",
            })

    result = pd.DataFrame(rows)

    variant_order = {
        value: index
        for index, value in enumerate(VARIANT_ORDER)
    }

    dataset_order = {
        value: index
        for index, value in enumerate(DATASET_ORDER)
    }

    result["dataset_order"] = result["dataset"].map(dataset_order)
    result["variant_order"] = result["variant"].map(variant_order)

    result = result.sort_values(
        [
            "dataset_order",
            "metric",
            "variant_order",
        ]
    )

    result = result.drop(
        columns=["dataset_order", "variant_order"]
    )

    return result


def prepare_statistical_table(stats):

    required_columns = {
        "dataset",
        "reference_variant",
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
    }

    missing = required_columns - set(stats.columns)

    if missing:
        raise ValueError(
            "Statistical file is missing required columns: "
            + str(sorted(missing))
        )

    result = stats[
        stats["metric"].isin(PRIMARY_METRICS)
    ].copy()

    result["ablation_label"] = result[
        "ablation_variant"
    ].map(variant_label)

    result["metric_label"] = result[
        "metric"
    ].map(
        lambda x: METRIC_LABELS.get(x, x)
    )

    result["ci95"] = (
        "["
        + result["ci95_lower"].map(
            lambda x: f"{float(x):.6f}"
        )
        + ", "
        + result["ci95_upper"].map(
            lambda x: f"{float(x):.6f}"
        )
        + "]"
    )

    variant_order = {
        "A1_no_self_attention": 1,
        "A2_no_semantic_tokenization": 2,
        "A3_single_head_attention": 3,
    }

    dataset_order = {
        value: index
        for index, value in enumerate(DATASET_ORDER)
    }

    result["dataset_order"] = result["dataset"].map(dataset_order)

    result["variant_order"] = result[
        "ablation_variant"
    ].map(variant_order)

    result = result.sort_values(
        [
            "dataset_order",
            "metric",
            "variant_order",
        ]
    )

    columns = [
        "dataset",
        "metric_label",
        "ablation_label",
        "reference_mean",
        "ablation_mean",
        "mean_difference_ablation_minus_reference",
        "ci95",
        "paired_cohens_d",
        "paired_rank_biserial",
        "wilcoxon_p_value",
        "wilcoxon_p_holm_by_metric",
        "exact_sign_flip_p_value",
        "sign_flip_p_holm_by_metric",
        "observed_direction",
    ]

    result = result[columns]

    return result


def prepare_component_summary(stats):

    rows = []

    for variant in [
        "A1_no_self_attention",
        "A2_no_semantic_tokenization",
        "A3_single_head_attention",
    ]:

        variant_subset = stats[
            stats["ablation_variant"] == variant
        ]

        row = {
            "ablation_variant": variant,
            "ablation_label": variant_label(variant),
        }

        for metric in PRIMARY_METRICS:

            subset = variant_subset[
                variant_subset["metric"] == metric
            ].copy()

            if subset.empty:
                continue

            deltas = subset[
                "mean_difference_ablation_minus_reference"
            ].astype(float)

            row[f"{metric}_mean_delta"] = deltas.mean()
            row[f"{metric}_median_delta"] = deltas.median()
            row[f"{metric}_min_delta"] = deltas.min()
            row[f"{metric}_max_delta"] = deltas.max()

            row[f"{metric}_negative_count"] = int(
                (deltas < 0).sum()
            )

            row[f"{metric}_positive_count"] = int(
                (deltas > 0).sum()
            )

            row[f"{metric}_zero_count"] = int(
                (deltas == 0).sum()
            )

            row[f"{metric}_holm_significant_count"] = int(
                (
                    subset["wilcoxon_p_holm_by_metric"].astype(float)
                    < 0.05
                ).sum()
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("PREPARING MANUSCRIPT-READY ABLATION TABLES")
    print("=" * 70)

    # --------------------------------------------------------
    # Check source files
    # --------------------------------------------------------

    for path in [
        AGGREGATE_PATH,
        STATS_PATH,
        ZERO_F1_PATH,
    ]:
        check_exists(path)

    print("\nLoading validated ablation outputs...")

    aggregate = pd.read_csv(AGGREGATE_PATH)
    stats = pd.read_csv(STATS_PATH)
    zero_f1 = pd.read_csv(ZERO_F1_PATH)

    # --------------------------------------------------------
    # Source validation
    # --------------------------------------------------------

    expected_variants = set(VARIANT_ORDER)
    expected_datasets = set(DATASET_ORDER)

    actual_variants = set(aggregate["variant"])
    actual_datasets = set(aggregate["dataset"])

    if actual_variants != expected_variants:
        raise ValueError(
            "Unexpected variants in aggregate file: "
            f"{sorted(actual_variants)}"
        )

    if actual_datasets != expected_datasets:
        raise ValueError(
            "Unexpected datasets in aggregate file: "
            f"{sorted(actual_datasets)}"
        )

    if len(aggregate) != 20:
        raise ValueError(
            f"Expected 20 aggregate rows, found {len(aggregate)}"
        )

    print("Aggregate schema validation: PASSED")
    print(f"Aggregate rows: {len(aggregate)}")

    print("Statistical rows:", len(stats))
    print("Zero-F1 rows:", len(zero_f1))

    # --------------------------------------------------------
    # Primary table
    # --------------------------------------------------------

    primary = build_primary_performance_table(
        aggregate
    )

    primary.to_csv(
        OUTPUT_TABLE,
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Statistical table
    # --------------------------------------------------------

    stats_primary = prepare_statistical_table(
        stats
    )

    stats_primary.to_csv(
        OUTPUT_STATS,
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Zero-F1 table
    # --------------------------------------------------------

    zero_output = zero_f1.copy()

    if (
        "ablation_variant"
        in zero_output.columns
    ):
        zero_output["ablation_label"] = zero_output[
            "ablation_variant"
        ].map(variant_label)

    zero_output.to_csv(
        OUTPUT_ZERO,
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Component summary
    # --------------------------------------------------------

    component_summary = prepare_component_summary(
        stats
    )

    component_summary.to_csv(
        OUTPUT_SUMMARY,
        index=False,
        float_format="%.6f",
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "analysis":
            "manuscript_ready_ablation_table_preparation",

        "reference_variant":
            "A0_full_hybrid_attention",

        "ablations": [
            "A1_no_self_attention",
            "A2_no_semantic_tokenization",
            "A3_single_head_attention",
        ],

        "datasets": DATASET_ORDER,

        "training_seeds": [
            42,
            43,
            44,
            45,
            46,
        ],

        "split_seed": 42,

        "primary_metrics": PRIMARY_METRICS,

        "difference_definition":
            "ablation minus A0 reference",

        "classification_threshold": 0.50,

        "statistical_correction":
            "Holm correction separately within each metric",

        "zero_f1_policy":
            "Retained as observed results; not treated as missing values",

        "aggregate_format":
            "wide format with metric-specific mean and SD columns",

        "source_files": {
            "aggregate": str(AGGREGATE_PATH),
            "statistics": str(STATS_PATH),
            "zero_f1": str(ZERO_F1_PATH),
        },

        "output_files": {
            "primary": str(OUTPUT_TABLE),
            "statistics": str(OUTPUT_STATS),
            "zero_f1": str(OUTPUT_ZERO),
            "component_summary": str(OUTPUT_SUMMARY),
        },
    }

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

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\nPRIMARY PERFORMANCE TABLE:")
    print(
        primary.to_string(
            index=False,
            max_rows=100,
        )
    )

    print("\nSTATISTICAL TABLE:")
    print(
        stats_primary.to_string(
            index=False,
            max_rows=100,
        )
    )

    print("\nZERO-F1 TABLE:")
    print(
        zero_output.to_string(
            index=False,
            max_rows=100,
        )
    )

    print("\nCOMPONENT SUMMARY:")
    print(
        component_summary.to_string(
            index=False
        )
    )

    print("\nSAVED FILES:")
    print(OUTPUT_TABLE)
    print(OUTPUT_STATS)
    print(OUTPUT_ZERO)
    print(OUTPUT_SUMMARY)
    print(OUTPUT_METADATA)

    print("\n" + "=" * 70)
    print("MANUSCRIPT ABLATION TABLE PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()