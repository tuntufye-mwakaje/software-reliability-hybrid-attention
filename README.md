# Research Results and Manuscript Artifacts

This directory contains the validated public results and manuscript artifacts for the multi-dataset software reliability study.

## Current Study Scope

Datasets: CM1, JM1, KC1, KC2, PC1
Training seeds: 42, 43, 44, 45, 46
Frozen partition seed: 42
Total seed-level experimental runs: 200

- Hybrid Attention: 25 runs
- Conventional baselines: 75 runs
- Ablation experiments: 100 runs

The experiments use frozen grouped-feature train/validation/test partitions and train-only standardization. The canonical feature space contains 21 predictors.

## Manuscript Tables

- tables/table_1_model_performance.csv
- tables/table_2_baseline_statistics.csv
- tables/table_3_baseline_effect_sizes.csv
- tables/table_4_ablation_performance.csv
- tables/table_5_ablation_statistics.csv
- tables/table_6_component_summary.csv
- tables/table_7_zero_f1_diagnostics.csv

## Manuscript Figures

- figures/manuscript/figure_1_model_performance.png
- figures/manuscript/figure_2_ablation_roc_auc.png
- figures/manuscript/figure_3_ablation_pr_auc.png
- figures/manuscript/figure_4_ablation_f1.png
- figures/manuscript/figure_5_ablation_deltas.png

## Statistical Analysis

Primary metrics are ROC-AUC, PR-AUC, and F1. Accuracy, precision, and recall are supplementary metrics.

Paired comparisons use the five controlled training seeds. The analysis includes exact Wilcoxon signed-rank tests, exact sign-flip tests, Holm correction, paired Cohen's d, rank-biserial correlation, and 95% Student-t confidence intervals.

No primary ablation comparison is Holm-significant at alpha = 0.05. The five training seeds are repeated controlled runs and should not be interpreted as five independent populations.

## Numerical Validation

Tables 4, 5, and 7 were reconciled against their validated source files. All corresponding values matched within 1e-9.

## Legacy Prototype

The legacy_kc1_single_run directory contains the earlier KC1-only prototype results. These artifacts are retained for provenance but are not the primary results of the current five-dataset study.

## Evidence Boundary

The artifacts document the implemented experiments and their observed results. They should not be interpreted as establishing universal superiority of the proposed architecture. Dataset-level variation, threshold-dependent F1 behavior, and the limited number of repeated training seeds remain important considerations.
