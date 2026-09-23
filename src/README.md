# Source Code

This directory contains the validated implementation of the multi-dataset software reliability prediction study.

## Current Study

The public source implements the experimental methodology used for CM1, JM1, KC1, KC2, and PC1.

### Core Modules

| Module | Purpose |
|---|---|
| dataset_schema.py | Defines and validates the canonical 21-feature schema and dataset-specific target normalization |
| grouped_split.py | Implements grouped, stratified partitioning with predictor-group leakage controls |
| research_data.py | Loads datasets and validates frozen seed-42 manifests and partitions |
| research_preprocessing.py | Applies train-only standardization and semantic feature-group tokenization |
| research_models.py | Defines the Hybrid Attention architecture |
| research_train.py | Trains the Hybrid Attention model across controlled seeds 42-46 |
| baseline_experiments.py | Runs Logistic Regression, Random Forest, and MLP baselines using the same partitions |
| ablation_experiments.py | Runs A0-A3 component ablation experiments |
| statistical_comparison.py | Performs paired seed-level statistical comparisons and Holm correction |
| effect_size_confidence_analysis.py | Computes paired effect sizes and confidence intervals |

## Experimental Design

- Datasets: CM1, JM1, KC1, KC2, PC1
- Frozen grouped train/validation/test partitions generated with split seed 42
- Train-only feature standardization
- 21 standardized predictors organized into four semantic feature groups
- Hybrid Attention input representation: four semantic tokens with width eight
- Controlled training seeds: 42, 43, 44, 45, 46
- Primary metrics: ROC-AUC, PR-AUC, and F1
- Supplementary metrics: accuracy, precision, and recall
- Hybrid Attention checkpoint selection uses validation loss before locked final test evaluation
- Baselines use the same frozen partitions and standardized predictors without validation-based hyperparameter tuning

## Ablation Variants

- A0: Full Hybrid Attention model
- A1: No self-attention
- A2: No semantic tokenization
- A3: Single-head attention

## Reproducibility

The source code is designed to reproduce the controlled experimental workflow documented in the repository methodology and reproducibility documentation. Raw datasets, frozen manifests, generated checkpoints, and private research outputs are intentionally kept outside the public source tree.
