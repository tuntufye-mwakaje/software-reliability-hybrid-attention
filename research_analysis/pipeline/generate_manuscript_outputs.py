from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PIPELINE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PIPELINE_DIR.parent / "results"
OUTPUT_DIR = RESULTS_DIR / "manuscript"

BASELINE_SUMMARY = RESULTS_DIR / "hybrid_attention_vs_baselines_summary.csv"
BASELINE_STATS = RESULTS_DIR / "hybrid_attention_statistical_comparison.csv"
BASELINE_EFFECTS = RESULTS_DIR / "hybrid_attention_effect_size_confidence_analysis.csv"

ABLATION_AGGREGATE = RESULTS_DIR / "ablation_repeated_seed_aggregate.csv"
ABLATION_STATS = RESULTS_DIR / "ablation_statistical_comparison.csv"
ABLATION_COMPONENT = RESULTS_DIR / "ablation_manuscript_component_summary.csv"
ABLATION_ZERO_F1 = RESULTS_DIR / "ablation_manuscript_zero_f1_table.csv"


# ============================================================
# CONSTANTS
# ============================================================

DATASETS = ["CM1", "JM1", "KC1", "KC2", "PC1"]

PRIMARY_METRICS = ["roc_auc", "pr_auc", "f1"]

SUPPLEMENTARY_METRICS = [
    "accuracy",
    "precision",
    "recall",
]

ALL_METRICS = PRIMARY_METRICS + SUPPLEMENTARY_METRICS

METRIC_LABELS = {
    "roc_auc": "ROC-AUC",
    "pr_auc": "PR-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
}

BASELINE_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "mlp": "MLP",
}

ABLATION_LABELS = {
    "A0_full_hybrid_attention": "A0 Full Hybrid Attention",
    "A1_no_self_attention": "A1 No Self-Attention",
    "A2_no_semantic_tokenization": "A2 No Semantic Tokenization",
    "A3_single_head_attention": "A3 Single-Head Attention",
}


# ============================================================
# HELPERS
# ============================================================

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file was not found:\n{path}"
        )


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_datasets(df: pd.DataFrame, name: str) -> None:
    if "dataset" not in df.columns:
        raise ValueError(f"{name}: missing 'dataset' column.")

    found = sorted(df["dataset"].dropna().unique().tolist())

    if found != sorted(DATASETS):
        raise ValueError(
            f"{name}: unexpected datasets.\n"
            f"Expected: {DATASETS}\n"
            f"Found: {found}"
        )


def round_numeric(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    result = df.copy()

    for column in result.columns:
        if pd.api.types.is_numeric_dtype(result[column]):
            result[column] = result[column].round(decimals)

    return result


def mean_sd_text(mean_value, sd_value) -> str:
    if pd.isna(mean_value) or pd.isna(sd_value):
        return "NA"

    return f"{float(mean_value):.4f} ± {float(sd_value):.4f}"


# ============================================================
# TABLE 1
# HYBRID VS BASELINES PERFORMANCE
# ============================================================

def build_baseline_performance_table() -> pd.DataFrame:
    df = pd.read_csv(BASELINE_SUMMARY)

    validate_datasets(df, "Baseline summary")

    required = {
        "dataset",
        "model",
        "n_seeds",
        "split_seed",
        "test_roc_auc_mean",
        "test_roc_auc_sd",
        "test_pr_auc_mean",
        "test_pr_auc_sd",
        "test_accuracy_mean",
        "test_accuracy_sd",
        "test_precision_mean",
        "test_precision_sd",
        "test_recall_mean",
        "test_recall_sd",
        "test_f1_mean",
        "test_f1_sd",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Baseline summary is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    rows = []

    for dataset in DATASETS:
        subset = df[df["dataset"] == dataset].copy()

        for _, row in subset.iterrows():
            model = row["model"]

            if model not in BASELINE_LABELS and model != "hybrid_attention":
                continue

            model_label = (
                "Hybrid Attention"
                if model == "hybrid_attention"
                else BASELINE_LABELS[model]
            )

            rows.append(
                {
                    "Dataset": dataset,
                    "Model": model_label,
                    "Seeds": int(row["n_seeds"]),
                    "Split Seed": int(row["split_seed"]),
                    "ROC-AUC": mean_sd_text(
                        row["test_roc_auc_mean"],
                        row["test_roc_auc_sd"],
                    ),
                    "PR-AUC": mean_sd_text(
                        row["test_pr_auc_mean"],
                        row["test_pr_auc_sd"],
                    ),
                    "Accuracy": mean_sd_text(
                        row["test_accuracy_mean"],
                        row["test_accuracy_sd"],
                    ),
                    "Precision": mean_sd_text(
                        row["test_precision_mean"],
                        row["test_precision_sd"],
                    ),
                    "Recall": mean_sd_text(
                        row["test_recall_mean"],
                        row["test_recall_sd"],
                    ),
                    "F1": mean_sd_text(
                        row["test_f1_mean"],
                        row["test_f1_sd"],
                    ),
                }
            )

    result = pd.DataFrame(rows)

    result["model_order"] = result["Model"].map(
        {
            "Hybrid Attention": 0,
            "Logistic Regression": 1,
            "Random Forest": 2,
            "MLP": 3,
        }
    )

    result = (
        result
        .sort_values(["Dataset", "model_order"])
        .drop(columns=["model_order"])
        .reset_index(drop=True)
    )

    return result


# ============================================================
# TABLE 2
# BASELINE STATISTICAL COMPARISON
# PRIMARY METRICS ONLY
# ============================================================

def build_baseline_statistical_table() -> pd.DataFrame:
    stats = pd.read_csv(BASELINE_STATS)

    validate_datasets(stats, "Baseline statistical comparison")

    required = {
        "dataset",
        "baseline_model",
        "metric",
        "n_pairs",
        "hybrid_mean",
        "baseline_mean",
        "mean_difference_hybrid_minus_baseline",
        "wilcoxon_p_value",
        "wilcoxon_p_holm_by_metric",
        "exact_sign_flip_p_value",
        "sign_flip_p_holm_by_metric",
        "paired_rank_biserial",
        "observed_direction",
    }

    missing = required - set(stats.columns)

    if missing:
        raise ValueError(
            "Baseline statistical comparison is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    stats = stats[
        stats["metric"].isin(PRIMARY_METRICS)
    ].copy()

    rows = []

    for _, row in stats.iterrows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Baseline": BASELINE_LABELS.get(
                    row["baseline_model"],
                    row["baseline_model"],
                ),
                "Metric": METRIC_LABELS[row["metric"]],
                "n": int(row["n_pairs"]),
                "Hybrid Mean": float(row["hybrid_mean"]),
                "Baseline Mean": float(row["baseline_mean"]),
                "Δ Hybrid − Baseline": float(
                    row["mean_difference_hybrid_minus_baseline"]
                ),
                "Wilcoxon p": float(row["wilcoxon_p_value"]),
                "Wilcoxon Holm p": float(
                    row["wilcoxon_p_holm_by_metric"]
                ),
                "Exact Sign-Flip p": float(
                    row["exact_sign_flip_p_value"]
                ),
                "Sign-Flip Holm p": float(
                    row["sign_flip_p_holm_by_metric"]
                ),
                "Rank-Biserial": float(
                    row["paired_rank_biserial"]
                ),
                "Observed Direction": row["observed_direction"],
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["Metric", "Dataset", "Baseline"]
    ).reset_index(drop=True)


# ============================================================
# TABLE 3
# BASELINE EFFECT SIZE + CI
# PRIMARY METRICS ONLY
# ============================================================

def build_baseline_effect_table() -> pd.DataFrame:
    effects = pd.read_csv(BASELINE_EFFECTS)

    validate_datasets(effects, "Baseline effect-size analysis")

    required = {
        "dataset",
        "baseline_model",
        "metric",
        "n_pairs",
        "hybrid_mean",
        "baseline_mean",
        "mean_difference_hybrid_minus_baseline",
        "ci95_lower",
        "ci95_upper",
        "paired_cohens_d",
        "paired_rank_biserial",
    }

    missing = required - set(effects.columns)

    if missing:
        raise ValueError(
            "Baseline effect-size analysis is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    effects = effects[
        effects["metric"].isin(PRIMARY_METRICS)
    ].copy()

    rows = []

    for _, row in effects.iterrows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Baseline": BASELINE_LABELS.get(
                    row["baseline_model"],
                    row["baseline_model"],
                ),
                "Metric": METRIC_LABELS[row["metric"]],
                "n": int(row["n_pairs"]),
                "Hybrid Mean": float(row["hybrid_mean"]),
                "Baseline Mean": float(row["baseline_mean"]),
                "Mean Δ": float(
                    row["mean_difference_hybrid_minus_baseline"]
                ),
                "95% CI Lower": float(row["ci95_lower"]),
                "95% CI Upper": float(row["ci95_upper"]),
                "Paired Cohen's d": float(
                    row["paired_cohens_d"]
                ),
                "Rank-Biserial": float(
                    row["paired_rank_biserial"]
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["Metric", "Dataset", "Baseline"]
    ).reset_index(drop=True)


# ============================================================
# TABLE 4
# ABLATION PERFORMANCE
# ============================================================

def build_ablation_performance_table() -> pd.DataFrame:
    df = pd.read_csv(ABLATION_AGGREGATE)

    validate_datasets(df, "Ablation aggregate")

    required = {
        "variant",
        "dataset",
    }

    for metric in PRIMARY_METRICS:
        required.add(f"{metric}_mean")
        required.add(f"{metric}_sd")

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Ablation aggregate is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    rows = []

    for _, row in df.iterrows():
        variant = row["variant"]

        for metric in PRIMARY_METRICS:
            rows.append(
                {
                    "Dataset": row["dataset"],
                    "Variant": ABLATION_LABELS.get(
                        variant,
                        variant,
                    ),
                    "Metric": METRIC_LABELS[metric],
                    "Mean": float(row[f"{metric}_mean"]),
                    "SD": float(row[f"{metric}_sd"]),
                    "Mean ± SD": mean_sd_text(
                        row[f"{metric}_mean"],
                        row[f"{metric}_sd"],
                    ),
                }
            )

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["Metric", "Dataset", "Variant"]
    ).reset_index(drop=True)


# ============================================================
# TABLE 5
# ABLATION STATISTICAL COMPARISON
# ============================================================

def build_ablation_statistical_table() -> pd.DataFrame:
    stats = pd.read_csv(ABLATION_STATS)

    validate_datasets(stats, "Ablation statistical comparison")

    required = {
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
    }

    missing = required - set(stats.columns)

    if missing:
        raise ValueError(
            "Ablation statistical comparison is missing columns:\n"
            + "\n".join(sorted(missing))
        )

    stats = stats[
        stats["metric"].isin(PRIMARY_METRICS)
    ].copy()

    rows = []

    for _, row in stats.iterrows():
        rows.append(
            {
                "Dataset": row["dataset"],
                "Ablation": ABLATION_LABELS.get(
                    row["ablation_variant"],
                    row["ablation_variant"],
                ),
                "Metric": METRIC_LABELS[row["metric"]],
                "Full Hybrid Mean": float(
                    row["reference_mean"]
                ),
                "Ablation Mean": float(
                    row["ablation_mean"]
                ),
                "Δ Ablation − Full": float(
                    row["mean_difference_ablation_minus_reference"]
                ),
                "95% CI Lower": float(row["ci95_lower"]),
                "95% CI Upper": float(row["ci95_upper"]),
                "Paired Cohen's d": float(
                    row["paired_cohens_d"]
                ),
                "Rank-Biserial": float(
                    row["paired_rank_biserial"]
                ),
                "Wilcoxon p": float(
                    row["wilcoxon_p_value"]
                ),
                "Wilcoxon Holm p": float(
                    row["wilcoxon_p_holm_by_metric"]
                ),
                "Exact Sign-Flip p": float(
                    row["exact_sign_flip_p_value"]
                ),
                "Sign-Flip Holm p": float(
                    row["sign_flip_p_holm_by_metric"]
                ),
                "Observed Direction": row["observed_direction"],
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        ["Metric", "Dataset", "Ablation"]
    ).reset_index(drop=True)


# ============================================================
# TABLE 6
# ABLATION COMPONENT SUMMARY
# ============================================================

def build_component_summary() -> pd.DataFrame:
    df = pd.read_csv(ABLATION_COMPONENT)

    if "variant" in df.columns:
        df["Variant"] = df["variant"].map(
            lambda x: ABLATION_LABELS.get(x, x)
        )

    return df


# ============================================================
# TABLE 7
# ZERO-F1 DIAGNOSTICS
# ============================================================

def build_zero_f1_table() -> pd.DataFrame:
    df = pd.read_csv(ABLATION_ZERO_F1)

    if "variant" in df.columns:
        df["variant"] = df["variant"].map(
            lambda x: ABLATION_LABELS.get(x, x)
        )
        df = df.rename(columns={"variant": "Variant"})

    return df


# ============================================================
# FIGURE 1
# HYBRID VS BASELINES ROC-AUC
# ============================================================

def plot_baseline_performance() -> Path:
    df = pd.read_csv(BASELINE_SUMMARY)

    rows = []

    for _, row in df.iterrows():
        model = row["model"]

        if model not in BASELINE_LABELS and model != "hybrid_attention":
            continue

        rows.append(
            {
                "dataset": row["dataset"],
                "model": (
                    "Hybrid Attention"
                    if model == "hybrid_attention"
                    else BASELINE_LABELS[model]
                ),
                "mean": row["test_roc_auc_mean"],
                "sd": row["test_roc_auc_sd"],
            }
        )

    plot_df = pd.DataFrame(rows)

    models = [
        "Hybrid Attention",
        "Logistic Regression",
        "Random Forest",
        "MLP",
    ]

    x = np.arange(len(DATASETS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, model in enumerate(models):
        subset = (
            plot_df[plot_df["model"] == model]
            .set_index("dataset")
            .reindex(DATASETS)
        )

        positions = x + (i - 1.5) * width

        ax.bar(
            positions,
            subset["mean"].values,
            width,
            yerr=subset["sd"].values,
            capsize=3,
            label=model,
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel("ROC-AUC (mean ± SD)")
    ax.set_title(
        "Hybrid Attention and Conventional Baselines Across Datasets"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()

    path = OUTPUT_DIR / "figure_1_model_performance.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


# ============================================================
# ABLATION FIGURES
# ============================================================

def plot_ablation_metric(metric: str, figure_number: int) -> Path:
    df = pd.read_csv(ABLATION_AGGREGATE)

    variants = [
        "A0_full_hybrid_attention",
        "A1_no_self_attention",
        "A2_no_semantic_tokenization",
        "A3_single_head_attention",
    ]

    x = np.arange(len(DATASETS))
    width = 0.18

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, variant in enumerate(variants):
        subset = (
            df[df["variant"] == variant]
            .set_index("dataset")
            .reindex(DATASETS)
        )

        positions = x + (i - 1.5) * width

        ax.bar(
            positions,
            subset[f"{metric}_mean"].values,
            width,
            yerr=subset[f"{metric}_sd"].values,
            capsize=3,
            label=ABLATION_LABELS[variant],
        )

    ax.set_xlabel("Dataset")
    ax.set_ylabel(f"{METRIC_LABELS[metric]} (mean ± SD)")
    ax.set_title(
        f"Ablation Analysis: {METRIC_LABELS[metric]}"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    fig.tight_layout()

    path = OUTPUT_DIR / (
        f"figure_{figure_number}_ablation_{metric}.png"
    )

    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


# ============================================================
# ABLATION DELTA FIGURE
# ============================================================

def plot_ablation_deltas() -> Path:
    df = pd.read_csv(ABLATION_STATS)

    df = df[
        df["metric"].isin(PRIMARY_METRICS)
    ].copy()

    variants = [
        "A1_no_self_attention",
        "A2_no_semantic_tokenization",
        "A3_single_head_attention",
    ]

    fig, axes = plt.subplots(
        len(PRIMARY_METRICS),
        1,
        figsize=(11, 12),
        sharex=True,
    )

    for ax, metric in zip(axes, PRIMARY_METRICS):
        metric_df = df[df["metric"] == metric]

        for variant in variants:
            subset = (
                metric_df[
                    metric_df["ablation_variant"] == variant
                ]
                .set_index("dataset")
                .reindex(DATASETS)
            )

            ax.plot(
                DATASETS,
                subset[
                    "mean_difference_ablation_minus_reference"
                ],
                marker="o",
                label=ABLATION_LABELS[variant],
            )

        ax.axhline(
            0,
            linewidth=1,
            linestyle="--",
        )

        ax.set_ylabel(
            f"Δ {METRIC_LABELS[metric]}"
        )
        ax.grid(axis="y", alpha=0.25)

    axes[-1].set_xlabel("Dataset")

    axes[0].legend(
        fontsize=8,
        loc="best",
    )

    fig.suptitle(
        "Ablation Effects Relative to the Full Hybrid Attention Model",
        y=0.995,
    )

    fig.tight_layout()

    path = OUTPUT_DIR / "figure_5_ablation_deltas.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 70)
    print("GENERATING MANUSCRIPT-READY TABLES AND FIGURES")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        BASELINE_SUMMARY,
        BASELINE_STATS,
        BASELINE_EFFECTS,
        ABLATION_AGGREGATE,
        ABLATION_STATS,
        ABLATION_COMPONENT,
        ABLATION_ZERO_F1,
    ]

    print("\nChecking required source files...")

    for path in required_files:
        require_file(path)
        print(f"FOUND: {path.name}")

    print("\nBuilding manuscript tables...")

    table_1 = build_baseline_performance_table()
    save_csv(
        table_1,
        OUTPUT_DIR / "table_1_model_performance.csv",
    )
    print(
        f"Table 1: {len(table_1)} rows"
    )

    table_2 = build_baseline_statistical_table()
    save_csv(
        table_2,
        OUTPUT_DIR / "table_2_baseline_statistics.csv",
    )
    print(
        f"Table 2: {len(table_2)} rows"
    )

    table_3 = build_baseline_effect_table()
    save_csv(
        table_3,
        OUTPUT_DIR / "table_3_baseline_effect_sizes.csv",
    )
    print(
        f"Table 3: {len(table_3)} rows"
    )

    table_4 = build_ablation_performance_table()
    save_csv(
        table_4,
        OUTPUT_DIR / "table_4_ablation_performance.csv",
    )
    print(
        f"Table 4: {len(table_4)} rows"
    )

    table_5 = build_ablation_statistical_table()
    save_csv(
        table_5,
        OUTPUT_DIR / "table_5_ablation_statistics.csv",
    )
    print(
        f"Table 5: {len(table_5)} rows"
    )

    table_6 = build_component_summary()
    save_csv(
        table_6,
        OUTPUT_DIR / "table_6_component_summary.csv",
    )
    print(
        f"Table 6: {len(table_6)} rows"
    )

    table_7 = build_zero_f1_table()
    save_csv(
        table_7,
        OUTPUT_DIR / "table_7_zero_f1_diagnostics.csv",
    )
    print(
        f"Table 7: {len(table_7)} rows"
    )

    print("\nGenerating figures...")

    figure_paths = []

    figure_paths.append(
        plot_baseline_performance()
    )

    figure_paths.append(
        plot_ablation_metric("roc_auc", 2)
    )

    figure_paths.append(
        plot_ablation_metric("pr_auc", 3)
    )

    figure_paths.append(
        plot_ablation_metric("f1", 4)
    )

    figure_paths.append(
        plot_ablation_deltas()
    )

    for path in figure_paths:
        print(f"FIGURE: {path.name}")

    metadata = {
        "source_files": [
            str(path)
            for path in required_files
        ],
        "datasets": DATASETS,
        "primary_metrics": PRIMARY_METRICS,
        "supplementary_metrics": SUPPLEMENTARY_METRICS,
        "baseline_models": list(BASELINE_LABELS.keys()),
        "ablation_variants": list(ABLATION_LABELS.keys()),
        "output_directory": str(OUTPUT_DIR),
        "tables": [
            "table_1_model_performance.csv",
            "table_2_baseline_statistics.csv",
            "table_3_baseline_effect_sizes.csv",
            "table_4_ablation_performance.csv",
            "table_5_ablation_statistics.csv",
            "table_6_component_summary.csv",
            "table_7_zero_f1_diagnostics.csv",
        ],
        "figures": [
            path.name
            for path in figure_paths
        ],
        "methodological_note": (
            "Baseline and ablation inferential comparisons use "
            "five paired training seeds (42-46) under the fixed "
            "seed-42 dataset partitions. Primary metrics are ROC-AUC, "
            "PR-AUC, and F1. Accuracy, precision, and recall are "
            "treated as supplementary metrics."
        ),
    }

    metadata_path = (
        OUTPUT_DIR /
        "manuscript_outputs_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(f"METADATA: {metadata_path.name}")

    print("\n" + "=" * 70)
    print("MANUSCRIPT OUTPUT GENERATION COMPLETED")
    print("=" * 70)
    print(f"Output directory:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()