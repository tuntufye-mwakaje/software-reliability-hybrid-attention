# Hybrid Attention-Based Software Reliability Prediction

A reproducible multi-dataset research implementation for software reliability prediction using semantic feature-group tokenization and a Hybrid Attention architecture.

## Research Overview

This repository contains the validated implementation, experimental evidence, statistical analysis, and manuscript artifacts for a software reliability prediction study across five NASA software defect datasets:

* **CM1**
* **JM1**
* **KC1**
* **KC2**
* **PC1**

The current implementation extends the original KC1-only prototype into a controlled multi-dataset experimental pipeline.

The study evaluates a Hybrid Attention model against conventional machine-learning baselines and controlled architectural ablations under a common experimental protocol.

## Experimental Design

The validated experimental workflow uses:

* Five datasets: CM1, JM1, KC1, KC2, PC1
* Frozen grouped train/validation/test partitions generated with split seed **42**
* Train-only feature standardization
* **21 canonical software-metric predictors**
* Four semantic feature groups
* Four semantic tokens with width eight
* Controlled training seeds: **42, 43, 44, 45, 46**
* Primary metrics: **ROC-AUC, PR-AUC, F1**
* Supplementary metrics: accuracy, precision, and recall
* Validation-loss-based checkpoint selection for Hybrid Attention
* Locked final test evaluation
* Common frozen partitions and standardized predictors for conventional baselines

The five training seeds represent repeated controlled runs rather than five independent populations.

## Experimental Matrix

The completed seed-level experimental matrix contains **200 runs**:

| Experiment             |    Runs |
| ---------------------- | ------: |
| Hybrid Attention       |      25 |
| Conventional baselines |      75 |
| Ablation experiments   |     100 |
| **Total**              | **200** |

### Conventional baselines

The baseline experiments include:

* Logistic Regression
* Random Forest
* Multilayer Perceptron (MLP)

### Hybrid Attention ablations

| Variant | Description                 |
| ------- | --------------------------- |
| A0      | Full Hybrid Attention model |
| A1      | No self-attention           |
| A2      | No semantic tokenization    |
| A3      | Single-head attention       |

The ablations are designed to isolate the contribution of major architectural components rather than to constitute unrelated alternative models.

## Hybrid Attention Architecture

The canonical implementation represents the 21 standardized predictors through four semantic feature groups:

| Feature group        | Features |           Width |
| -------------------- | -------: | --------------: |
| Size                 |        5 | 8 after padding |
| Complexity           |        4 | 8 after padding |
| Halstead             |        8 |               8 |
| Operators / operands |        4 | 8 after padding |

The resulting representation has shape:

```text
(samples, 4 tokens, 8 features)
```

The Hybrid Attention model applies:

1. Feature-group tokenization
2. Linear embedding
3. Multi-head self-attention
4. Layer normalization
5. Feed-forward transformation
6. Mean token pooling
7. Binary classification head

The current implementation contains **9,985 trainable parameters**.

## Repository Structure

```text
software-reliability-hybrid-attention/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── notebooks/
│   └── README.md
│
├── research_analysis/
│   ├── datasets/
│   ├── manifests/
│   ├── models/
│   ├── pipeline/
│   └── results/
│
├── results/
│   ├── figures/
│   │   └── manuscript/
│   ├── tables/
│   └── legacy_kc1_single_run/
│
├── src/
│   ├── dataset_schema.py
│   ├── grouped_split.py
│   ├── research_data.py
│   ├── research_preprocessing.py
│   ├── research_models.py
│   ├── research_train.py
│   ├── baseline_experiments.py
│   ├── ablation_experiments.py
│   ├── statistical_comparison.py
│   └── effect_size_confidence_analysis.py
│
├── tests/
│   ├── test_pipeline.py
│   └── test_pipeline.py.legacy
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### Directory roles

**`src/`**
Canonical reusable implementation of the validated research components.

**`research_analysis/`**
Research execution and evidence layer containing dataset provenance, frozen manifests, experiment checkpoints, analysis scripts, audits, and generated research outputs.

**`results/`**
Public manuscript tables, figures, statistical artifacts, and clearly separated legacy prototype results.

**`tests/`**
Automated regression tests for the canonical implementation, together with the retained historical KC1-only test file.

**`data/`, `models/`, `notebooks/`**
Supporting documentation and repository organization for the public research package.

## Canonical Source Modules

The main implementation is documented in [`src/README.md`](src/README.md).

Important modules include:

| Module                               | Purpose                                                                      |
| ------------------------------------ | ---------------------------------------------------------------------------- |
| `dataset_schema.py`                  | Validates the canonical 21-feature schema and target normalization           |
| `grouped_split.py`                   | Creates grouped, stratified partitions with predictor-group leakage controls |
| `research_data.py`                   | Loads datasets and validates frozen manifests                                |
| `research_preprocessing.py`          | Applies train-only standardization and feature-group tokenization            |
| `research_models.py`                 | Defines the Hybrid Attention architecture                                    |
| `research_train.py`                  | Trains Hybrid Attention across controlled seeds                              |
| `baseline_experiments.py`            | Runs conventional baseline experiments                                       |
| `ablation_experiments.py`            | Runs A0-A3 ablation experiments                                              |
| `statistical_comparison.py`          | Performs paired statistical comparisons                                      |
| `effect_size_confidence_analysis.py` | Computes effect sizes and confidence intervals                               |

## Research Analysis Pipeline

The `research_analysis/pipeline/` directory contains the executable research workflow.

Major stages include:

```text
Dataset validation
      ↓
Dataset provenance and audits
      ↓
Frozen grouped partitions
      ↓
Train-only standardization
      ↓
Semantic feature-group tokenization
      ↓
Hybrid Attention training
      ↓
Baseline experiments
      ↓
Ablation experiments
      ↓
Repeated-seed aggregation
      ↓
Statistical comparison
      ↓
Effect-size / confidence analysis
      ↓
Manuscript tables and figures
```

Relevant pipeline scripts include:

* `validate_schema.py`
* `audit_manifests.py`
* `audit_research_results.py`
* `generate_manifests.py`
* `research_train.py`
* `baseline_experiments.py`
* `ablation_experiments.py`
* `aggregate_repeated_seeds.py`
* `aggregate_baseline_repeated_seeds.py`
* `aggregate_ablation_repeated_seeds.py`
* `statistical_comparison.py`
* `effect_size_confidence_analysis.py`
* `generate_manuscript_outputs.py`

Completed experimental artifacts are retained in the repository; the completed experiments do not need to be rerun merely to reproduce the repository migration.

## Installation

The repository uses pinned Python dependencies:

```text
numpy==1.26.4
pandas==2.2.3
scikit-learn==1.6.1
matplotlib==3.10.0
torch==2.8.0
pytest==9.1.1
```

Create and activate a virtual environment, then install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Validation

The canonical regression suite can be executed with:

```powershell
python -m pytest ".\tests\test_pipeline.py" -q
```

Current validation result:

```text
11 passed in 1.96s
```

The tests cover, among other components:

* Canonical dataset schema
* Feature-group definitions
* Column canonicalization
* Target validation
* Grouped split leakage controls
* Train-only standardization
* Semantic token shape and dtype
* Hybrid Attention forward-pass shape
* Model parameter count
* Deterministic evaluation behavior
* Identical-row grouping behavior

A separate grouped-split validation script is also retained under:

```text
research_analysis/pipeline/test_grouped_split.py
```

## Results and Manuscript Artifacts

The public manuscript artifacts are located under [`results/`](results/).

### Tables

* [`table_1_model_performance.csv`](results/tables/table_1_model_performance.csv)
* [`table_2_baseline_statistics.csv`](results/tables/table_2_baseline_statistics.csv)
* [`table_3_baseline_effect_sizes.csv`](results/tables/table_3_baseline_effect_sizes.csv)
* [`table_4_ablation_performance.csv`](results/tables/table_4_ablation_performance.csv)
* [`table_5_ablation_statistics.csv`](results/tables/table_5_ablation_statistics.csv)
* [`table_6_component_summary.csv`](results/tables/table_6_component_summary.csv)
* [`table_7_zero_f1_diagnostics.csv`](results/tables/table_7_zero_f1_diagnostics.csv)

### Manuscript figures

* [`figure_1_model_performance.png`](results/figures/manuscript/figure_1_model_performance.png)
* [`figure_2_ablation_roc_auc.png`](results/figures/manuscript/figure_2_ablation_roc_auc.png)
* [`figure_3_ablation_pr_auc.png`](results/figures/manuscript/figure_3_ablation_pr_auc.png)
* [`figure_4_ablation_f1.png`](results/figures/manuscript/figure_4_ablation_f1.png)
* [`figure_5_ablation_deltas.png`](results/figures/manuscript/figure_5_ablation_deltas.png)

The results documentation is available in [`results/README.md`](results/README.md).

## Statistical Analysis

The primary evaluation metrics are:

* ROC-AUC
* PR-AUC
* F1

Accuracy, precision, and recall are treated as supplementary metrics.

The repeated-seed analysis includes:

* Exact Wilcoxon signed-rank tests
* Exact sign-flip tests
* Holm multiple-comparison correction
* Paired Cohen's *d*
* Rank-biserial correlation
* 95% Student-*t* confidence intervals

The manuscript artifacts report the resulting observed comparisons and effect estimates. The five controlled training seeds should not be interpreted as five independent populations.

## Numerical Reconciliation

The manuscript tables were reconciled against their validated source artifacts.

Tables 4, 5, and 7 were specifically reconciled with corresponding source files, with reported values matching within `1e-9`.

## Reproducibility and Evidence

The repository is organized to preserve the distinction between:

1. **Implementation** — canonical source code in `src/`
2. **Research execution** — analysis pipeline and artifacts in `research_analysis/`
3. **Observed evidence** — tables, figures, statistical outputs, and diagnostics in `results/`
4. **Validation** — automated tests in `tests/`
5. **Historical provenance** — legacy KC1-only prototype artifacts

This separation is intended to make the experimental evidence traceable without treating the earlier prototype as equivalent to the validated five-dataset study.

## Legacy Prototype

The earlier KC1-only implementation is retained under:

```text
results/legacy_kc1_single_run/
```

These files are preserved for provenance and historical comparison. They are not the primary results of the current five-dataset study.

The historical test file is retained separately as:

```text
tests/test_pipeline.py.legacy
```

The current `tests/test_pipeline.py` is the authoritative regression suite for the validated implementation.

## Evidence Boundary

The repository documents the implemented experimental methodology and its observed results.

The evidence should not be interpreted as establishing universal superiority of the proposed architecture. Results can vary across datasets, evaluation thresholds, and repeated training runs. The limited number of controlled training seeds is also an important consideration when interpreting statistical results.

The repository therefore emphasizes reproducibility, controlled comparisons, transparent statistical analysis, and explicit separation between observed results and broader claims.

## Research Status

Current repository checkpoint:

* Five datasets validated
* Frozen seed-42 partitions retained
* Train-only standardization implemented
* 21-feature canonical schema implemented
* Four semantic feature groups implemented
* Hybrid Attention implementation validated
* Conventional baselines implemented
* Four ablation variants implemented
* 200 seed-level experimental runs completed
* Statistical and effect-size analyses completed
* Manuscript tables and figures generated
* Automated regression suite passing
* Repository migration committed and pushed to GitHub

Latest migration commit:

```text
69486e6 Migrate repository to validated multi-dataset research pipeline
```
