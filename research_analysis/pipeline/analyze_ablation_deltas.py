from pathlib import Path
import json
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PIPELINE_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = PIPELINE_DIR.parent
RESULTS_DIR = RESEARCH_DIR / "results"

INPUT_CSV = RESULTS_DIR / "ablation_repeated_seed_aggregate.csv"

DELTA_CSV = RESULTS_DIR / "ablation_component_deltas.csv"
ZERO_F1_CSV = RESULTS_DIR / "ablation_zero_f1_diagnostics.csv"
VARIABILITY_CSV = RESULTS_DIR / "ablation_seed_variability_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "ablation_diagnostic_summary.json"


# ============================================================
# CONFIGURATION
# ============================================================

REFERENCE_VARIANT = "A0_full_hybrid_attention"

ABLATION_VARIANTS = {
    "A1_no_self_attention": "Self-attention removed",
    "A2_no_semantic_tokenization": "Semantic tokenization removed",
    "A3_single_head_attention": "Four heads changed to one head",
}

METRICS = [
    "roc_auc",
    "pr_auc",
    "f1",
]


# ============================================================
# LOAD
# ============================================================

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Required aggregate file was not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

required_columns = [
    "variant",
    "dataset",
    "roc_auc_mean",
    "roc_auc_sd",
    "pr_auc_mean",
    "pr_auc_sd",
    "f1_mean",
    "f1_sd",
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# BASIC VALIDATION
# ============================================================

expected_variants = {
    REFERENCE_VARIANT,
    *ABLATION_VARIANTS.keys(),
}

actual_variants = set(df["variant"].unique())

if actual_variants != expected_variants:
    raise ValueError(
        "Unexpected variants detected.\n"
        f"Expected: {sorted(expected_variants)}\n"
        f"Found: {sorted(actual_variants)}"
    )

datasets = sorted(df["dataset"].unique())

if len(datasets) != 5:
    raise ValueError(
        f"Expected 5 datasets, found {len(datasets)}: {datasets}"
    )


# ============================================================
# COMPONENT DELTAS
# ============================================================
#
# Positive delta:
#     ablation > full model
#
# Negative delta:
#     ablation < full model
#
# This deliberately avoids calling one variant "better".
# ============================================================

reference = (
    df[df["variant"] == REFERENCE_VARIANT]
    .set_index("dataset")
)

delta_rows = []

for variant, description in ABLATION_VARIANTS.items():

    ablation = (
        df[df["variant"] == variant]
        .set_index("dataset")
    )

    for dataset in datasets:

        row = {
            "variant": variant,
            "component_change": description,
            "dataset": dataset,
        }

        for metric in METRICS:

            mean_col = f"{metric}_mean"
            sd_col = f"{metric}_sd"

            reference_mean = reference.loc[dataset, mean_col]
            ablation_mean = ablation.loc[dataset, mean_col]

            reference_sd = reference.loc[dataset, sd_col]
            ablation_sd = ablation.loc[dataset, sd_col]

            row[f"{metric}_full"] = reference_mean
            row[f"{metric}_ablation"] = ablation_mean

            row[f"{metric}_delta"] = (
                ablation_mean - reference_mean
            )

            row[f"{metric}_abs_delta"] = abs(
                ablation_mean - reference_mean
            )

            row[f"{metric}_sd_full"] = reference_sd
            row[f"{metric}_sd_ablation"] = ablation_sd

        delta_rows.append(row)

delta_df = pd.DataFrame(delta_rows)

delta_df.to_csv(DELTA_CSV, index=False)


# ============================================================
# ZERO-F1 DIAGNOSTICS
# ============================================================

zero_f1_rows = []

for _, row in df.iterrows():

    if np.isclose(row["f1_mean"], 0.0):

        zero_f1_rows.append({
            "variant": row["variant"],
            "dataset": row["dataset"],
            "f1_mean": row["f1_mean"],
            "f1_sd": row["f1_sd"],
            "roc_auc_mean": row["roc_auc_mean"],
            "pr_auc_mean": row["pr_auc_mean"],
            "pr_auc_sd": row["pr_auc_sd"],
        })

zero_f1_df = pd.DataFrame(zero_f1_rows)

zero_f1_df.to_csv(ZERO_F1_CSV, index=False)


# ============================================================
# VARIABILITY SUMMARY
# ============================================================

variability_rows = []

for _, row in df.iterrows():

    variability_rows.append({
        "variant": row["variant"],
        "dataset": row["dataset"],

        "roc_auc_mean": row["roc_auc_mean"],
        "roc_auc_sd": row["roc_auc_sd"],
        "roc_auc_cv_percent": (
            abs(row["roc_auc_sd"] / row["roc_auc_mean"]) * 100
            if row["roc_auc_mean"] != 0 else np.nan
        ),

        "pr_auc_mean": row["pr_auc_mean"],
        "pr_auc_sd": row["pr_auc_sd"],
        "pr_auc_cv_percent": (
            abs(row["pr_auc_sd"] / row["pr_auc_mean"]) * 100
            if row["pr_auc_mean"] != 0 else np.nan
        ),

        "f1_mean": row["f1_mean"],
        "f1_sd": row["f1_sd"],
        "f1_cv_percent": (
            abs(row["f1_sd"] / row["f1_mean"]) * 100
            if row["f1_mean"] != 0 else np.nan
        ),
    })

variability_df = pd.DataFrame(variability_rows)

variability_df.to_csv(VARIABILITY_CSV, index=False)


# ============================================================
# AGGREGATED DIAGNOSTICS
# ============================================================

summary = {
    "reference_variant": REFERENCE_VARIANT,
    "datasets": datasets,
    "ablation_variants": ABLATION_VARIANTS,
    "number_of_datasets": len(datasets),
    "number_of_variants": len(expected_variants),
    "zero_f1_rows": int(len(zero_f1_df)),
}

for variant in ABLATION_VARIANTS:

    subset = delta_df[
        delta_df["variant"] == variant
    ]

    summary[variant] = {
        "roc_auc_delta_mean_across_datasets":
            float(subset["roc_auc_delta"].mean()),

        "roc_auc_delta_sd_across_datasets":
            float(subset["roc_auc_delta"].std(ddof=1)),

        "pr_auc_delta_mean_across_datasets":
            float(subset["pr_auc_delta"].mean()),

        "pr_auc_delta_sd_across_datasets":
            float(subset["pr_auc_delta"].std(ddof=1)),

        "f1_delta_mean_across_datasets":
            float(subset["f1_delta"].mean()),

        "f1_delta_sd_across_datasets":
            float(subset["f1_delta"].std(ddof=1)),

        "datasets_with_negative_roc_auc_delta":
            int((subset["roc_auc_delta"] < 0).sum()),

        "datasets_with_positive_roc_auc_delta":
            int((subset["roc_auc_delta"] > 0).sum()),

        "datasets_with_negative_pr_auc_delta":
            int((subset["pr_auc_delta"] < 0).sum()),

        "datasets_with_positive_pr_auc_delta":
            int((subset["pr_auc_delta"] > 0).sum()),

        "datasets_with_negative_f1_delta":
            int((subset["f1_delta"] < 0).sum()),

        "datasets_with_positive_f1_delta":
            int((subset["f1_delta"] > 0).sum()),
    }


with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)


# ============================================================
# CONSOLE REPORT
# ============================================================

print("=" * 70)
print("ABLATION COMPONENT DELTA ANALYSIS")
print("=" * 70)

print(f"Reference: {REFERENCE_VARIANT}")
print(f"Datasets: {datasets}")
print()

print("Interpretation:")
print("  delta = ablation mean - full-model mean")
print("  negative delta -> ablation metric is lower")
print("  positive delta -> ablation metric is higher")
print()

for variant in ABLATION_VARIANTS:

    print("-" * 70)
    print(variant)
    print(ABLATION_VARIANTS[variant])
    print("-" * 70)

    subset = delta_df[
        delta_df["variant"] == variant
    ]

    display_cols = [
        "dataset",
        "roc_auc_delta",
        "pr_auc_delta",
        "f1_delta",
    ]

    print(
        subset[display_cols]
        .to_string(index=False, float_format=lambda x: f"{x:.6f}")
    )

    print()

    print(
        f"Mean ROC-AUC delta: "
        f"{subset['roc_auc_delta'].mean():.6f}"
    )

    print(
        f"Mean PR-AUC delta:   "
        f"{subset['pr_auc_delta'].mean():.6f}"
    )

    print(
        f"Mean F1 delta:       "
        f"{subset['f1_delta'].mean():.6f}"
    )

print()
print("=" * 70)
print("ZERO-F1 DIAGNOSTICS")
print("=" * 70)

if zero_f1_df.empty:
    print("No zero-F1 rows detected.")
else:
    print(
        zero_f1_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

print()
print("=" * 70)
print("FILES GENERATED")
print("=" * 70)

print(DELTA_CSV)
print(ZERO_F1_CSV)
print(VARIABILITY_CSV)
print(SUMMARY_JSON)

print()
print("DIAGNOSTIC ANALYSIS COMPLETE")