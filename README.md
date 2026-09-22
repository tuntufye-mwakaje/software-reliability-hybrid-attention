# Hybrid Attention-Based Software Reliability Prediction

A reproducible deep learning project for software defect prediction using the NASA PROMISE KC1 dataset. The project investigates whether semantic groups of software metrics can be represented as tokens and modeled with multi-head self-attention for software reliability analysis.

> \*\*Project status:\*\* Reproducible experimental implementation with documented evaluation, baseline comparisons, threshold analysis, and attention visualization.

## Overview

Software reliability prediction aims to identify software modules that are more likely to contain defects. This project develops a hybrid attention-based neural architecture that groups KC1 software metrics into semantic feature groups, embeds those groups as tokens, and applies multi-head self-attention before binary classification.

The implementation emphasizes experimental validity:

* Data splitting is performed before fitted preprocessing.
* The feature scaler is fitted only on the training partition.
* The final test set remains untouched during threshold selection.
* The primary experiment does not use synthetic target-derived signals.
* The primary experiment does not use data augmentation.
* Class imbalance is explicitly reported.
* Conventional machine-learning baselines are evaluated using the same final test partition.
* Model behavior is analyzed through the learned attention matrix.

## Research Context

**Research topic**

> Hybrid Attention-Based Deep Learning Framework for Software Reliability Prediction in AI-Integrated Systems

**Dataset**

NASA PROMISE KC1 software defect dataset, accessed through OpenML dataset ID `1067`.

The dataset contains software engineering metrics describing software modules and a binary `defects` target.

## Dataset

The project uses 21 numeric software metrics and one binary target:

* **Rows:** 2,109
* **Features:** 21
* **Target:** `defects`
* **Non-defective modules:** 1,783
* **Defective modules:** 326
* **Defective proportion:** approximately 15.46%

The raw dataset is stored locally under:

```text
data/raw/kc1.csv
```

Raw data is excluded from Git tracking through `.gitignore`.

Dataset acquisition and validation are implemented in:

```text
src/data\_loader.py
```

The loader validates the expected schema, row count, and target distribution after loading the dataset.

See [`data/README.md`](data/README.md) for dataset provenance, acquisition, validation, and data-handling details.

## Methodology

### 1\. Feature selection and grouping

The 21 software metrics are organized into four project-defined semantic groups:

|Feature group|Metrics|
|-|-|
|Size|`loc`, `lOCode`, `lOComment`, `lOBlank`, `locCodeAndComment`|
|Complexity|`v(g)`, `ev(g)`, `iv(g)`, `branchCount`|
|Halstead|`n`, `v`, `l`, `d`, `i`, `e`, `b`, `t`|
|Operators / Operands|`uniq\_Op`, `uniq\_Opnd`, `total\_Op`, `total\_Opnd`|

These groups are a representation design introduced by this project. They should not be interpreted as an official NASA/PROMISE taxonomy.

### 2\. Leakage-aware data preparation

The data is divided into three partitions:

```text
Full dataset
    |
    +-- Final test set: 20%
    |
    +-- Development set: 80%
             |
             +-- Training set: 80% of development data
             |
             +-- Validation set: 20% of development data
```

The resulting sample counts are:

|Partition|Samples|
|-|-:|
|Training|1,349|
|Validation|338|
|Final test|422|

All splits are stratified using random seed `42`.

`StandardScaler` is fitted only on the training partition and then applied to validation and final test data. This prevents information from the evaluation partitions from influencing preprocessing.

### 3\. Feature tokens

Each semantic feature group is represented as a token.

Because the four groups contain different numbers of features, smaller groups are zero-padded to the maximum group width of eight features.

The resulting tensor representation is:

```text
(samples, 4, 8)
```

where:

* `4` = semantic feature-group tokens
* `8` = maximum feature width after padding

### 4\. Hybrid attention architecture

The model consists of:

```text
Grouped KC1 metrics
        |
        v
Standardized feature groups
        |
        v
Feature-group tokens
        |
        v
Learned feature-group embedding
        |
        v
Multi-head self-attention
        |
        v
Feed-forward transformation
        |
        v
Mean pooling across tokens
        |
        v
Binary classification head
        |
        v
Defect prediction
```

The implemented model uses:

* Input dimension: `8`
* Embedding dimension: `32`
* Attention heads: `4`
* Dropout: `0.1`
* Trainable parameters: `9,985`

The implementation is located in:

```text
src/models.py
```

The attention layer uses PyTorch `nn.MultiheadAttention`.

The exported attention matrix is the default averaged attention across the four heads. It should therefore not be interpreted as separate per-head attention maps.

## Training

The primary experiment uses:

* Optimizer: Adam
* Learning rate: `0.001`
* Weight decay: `0.0001`
* Batch size: `32`
* Maximum epochs: `50`
* Early-stopping patience: `10`
* Learning-rate reduction factor: `0.5`
* Learning-rate scheduler patience: `3`
* Minimum learning rate: `1e-6`
* Gradient clipping: `1.0`
* Random seed: `42`

The best checkpoint is selected using validation loss.

Training stopped early at epoch 16.

Best validation loss:

```text
0.345243
```

Training implementation:

```text
src/train.py
```

The trained checkpoint is stored locally at:

```text
models/hybrid\_attention\_best.pt
```

## Evaluation

The final test set contains 422 samples:

* 357 non-defective
* 65 defective

### Hybrid attention model â€” threshold 0.50

|Metric|Value|
|-|-:|
|ROC AUC|0.796639|
|PR AUC|0.422107|
|Accuracy|0.855450|
|Precision|0.750000|
|Recall|0.092308|
|F1|0.164384|

Confusion matrix:

```text
                  Predicted
                Non-defect  Defect
Actual Non-defect    355       2
Actual Defect        59        6
```

The default threshold demonstrates an important property of this imbalanced classification problem: high accuracy can coexist with poor defective-module recall.

### Validation-selected threshold

A decision threshold was selected using the validation partition rather than the final test set.

The selected threshold was:

```text
0.18
```

The final test performance at this threshold was:

|Metric|Value|
|-|-:|
|ROC AUC|0.796639|
|PR AUC|0.422107|
|Accuracy|0.748815|
|Precision|0.348148|
|Recall|0.723077|
|F1|0.470000|

Confusion matrix:

```text
                  Predicted
                Non-defect  Defect
Actual Non-defect    269      88
Actual Defect         18      47
```

The threshold comparison illustrates the tradeoff between precision and recall. The threshold was chosen using validation data and then applied once to the untouched final test set.

## Baseline comparison

Two conventional models were evaluated on the same final test partition:

* Logistic Regression
* Random Forest

At their default `0.50` operating threshold:

|Model|ROC AUC|PR AUC|Accuracy|Precision|Recall|F1|
|-|-:|-:|-:|-:|-:|-:|
|Logistic Regression|0.789528|0.479546|0.755924|0.346774|0.661538|0.455026|
|Random Forest|0.809761|0.489432|0.872038|0.666667|0.338462|0.448980|
|Hybrid Attention|0.796639|0.422107|0.855450|0.750000|0.092308|0.164384|

The results do **not** establish that the hybrid attention model universally outperforms the baselines. In particular, the Random Forest has higher ROC AUC and PR AUC in this experiment.

The hybrid model's validation-selected threshold provides a different operating point, with substantially higher defective-module recall and an F1 score of `0.470000`. Because this threshold differs from the baseline operating points, these threshold-specific values should be interpreted as an operating-point analysis rather than a direct model-ranking claim.

## Attention Analysis

The model exposes its learned attention weights for analysis.

The average attention matrix indicates that the **Halstead** token receives substantial attention from the other feature-group tokens. The largest average attention value in the current experiment is:

```text
Complexity -> Halstead: 0.425351
```

The complete average attention matrix is available in:

```text
results/attention\_weights.csv
```

and visualized in:

```text
results/figures/attention\_heatmap.png
```

Attention weights describe model behavior in this architecture. They should not be interpreted as proof of causal relationships or conventional statistical feature importance.

## Evaluation Figures

The repository contains generated figures for the current experiment:

|Figure|Purpose|
|-|-|
|`training\_history.png`|Training and validation loss|
|`roc\_curve.png`|ROC curve|
|`precision\_recall\_curve.png`|Precision-recall curve|
|`test\_confusion\_matrix.png`|Final test confusion matrix|
|`model\_comparison.png`|Baseline/model metric comparison|
|`validation\_f1\_threshold.png`|Validation F1 across thresholds|
|`validation\_precision\_recall\_threshold.png`|Validation precision-recall tradeoff|
|`attention\_heatmap.png`|Average feature-group attention|

## Repository Structure

```text
software-reliability-hybrid-attention/
â”‚
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ raw/
â”‚   â”‚   â””â”€â”€ kc1.csv                  # Local raw dataset, ignored by Git
â”‚   â””â”€â”€ README.md                    # Dataset documentation
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ hybrid\_attention\_best.pt     # Trained model checkpoint
â”‚   â””â”€â”€ README.md
â”‚
â”œâ”€â”€ notebooks/
â”‚   â”œâ”€â”€ complete\_workflow\_cells.ipynb
â”‚   â””â”€â”€ README.md
â”‚
â”œâ”€â”€ results/
â”‚   â”œâ”€â”€ attention\_weights.csv
â”‚   â”œâ”€â”€ baseline\_results.csv
â”‚   â”œâ”€â”€ hybrid\_attention\_results.csv
â”‚   â”œâ”€â”€ hybrid\_attention\_threshold\_selected.csv
â”‚   â”œâ”€â”€ model\_comparison.csv
â”‚   â”œâ”€â”€ threshold\_analysis.csv
â”‚   â”œâ”€â”€ training\_history.csv
â”‚   â”œâ”€â”€ README.md
â”‚   â””â”€â”€ figures/
â”‚       â”œâ”€â”€ attention\_heatmap.png
â”‚       â”œâ”€â”€ model\_comparison.png
â”‚       â”œâ”€â”€ precision\_recall\_curve.png
â”‚       â”œâ”€â”€ roc\_curve.png
â”‚       â”œâ”€â”€ test\_confusion\_matrix.png
â”‚       â”œâ”€â”€ training\_history.png
â”‚       â”œâ”€â”€ validation\_f1\_threshold.png
â”‚       â””â”€â”€ validation\_precision\_recall\_threshold.png
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ attention\_analysis.py
â”‚   â”œâ”€â”€ baseline\_models.py
â”‚   â”œâ”€â”€ compare\_models.py
â”‚   â”œâ”€â”€ data\_loader.py
â”‚   â”œâ”€â”€ evaluate.py
â”‚   â”œâ”€â”€ feature\_profile.py
â”‚   â”œâ”€â”€ models.py
â”‚   â”œâ”€â”€ plot\_evaluation.py
â”‚   â”œâ”€â”€ preprocessing.py
â”‚   â”œâ”€â”€ threshold\_analysis.py
â”‚   â”œâ”€â”€ train.py
â”‚   â””â”€â”€ README.md
â”‚
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test\_pipeline.py
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ LICENSE
â””â”€â”€ README.md
```

## Reproducibility

### 1\. Clone the repository

```bash
git clone https://github.com/tuntufye-mwakaje/software-reliability-hybrid-attention.git
cd software-reliability-hybrid-attention
```

### 2\. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3\. Install dependencies

```bash

pip install -r requirements.txt
```

### 4\. Run the training pipeline

From the repository root:

```bash
python -m src.train
```

### 5\. Run evaluation

```bash
python -m src.evaluate
```

### 6\. Run baseline comparison

```bash
python -m src.baseline\_models
python -m src.compare\_models
```

### 7\. Run threshold analysis

```bash
python -m src.threshold\_analysis
```

### 8\. Generate attention analysis

```bash
python -m src.attention\_analysis
```

### 9\. Generate evaluation figures

```bash
python -m src.plot\_evaluation
```

### 10\. Run tests

```bash
pytest
```

The current automated test suite contains 9 tests covering dataset validation, preprocessing, feature-token construction, model output shapes, checkpoint loading, and expected result artifacts.

## Experimental Integrity

The current primary experiment was revised to address methodological problems identified during development.

The current pipeline avoids:

* preprocessing fitted on the full dataset before splitting
* target-derived synthetic runtime signals
* augmentation performed before train/test separation
* validation-based threshold selection using the final test set
* treating placeholder embeddings as meaningful learned representations
* presenting the old experimental results as final evidence

This separation is important because software defect datasets are relatively small and imbalanced, making leakage and evaluation design particularly important.

## Current Limitations

This project is an experimental research implementation and has several limitations:

1. **Single dataset:** The current evaluation uses KC1 only.
2. **Class imbalance:** Defective modules represent approximately 15.46% of the dataset.
3. **Single random split:** The current primary experiment uses one stratified split with seed `42`.
4. **Representation design:** The four feature groups are project-defined rather than an externally validated taxonomy.
5. **Token padding:** Groups with fewer than eight features are zero-padded after standardization.
6. **Attention interpretation:** Attention weights indicate model interaction patterns, not causal importance.
7. **Limited baselines:** The current comparison includes Logistic Regression and Random Forest.
8. **No external validation:** Results have not yet been validated on additional NASA/PROMISE datasets.
9. **Threshold dependence:** Classification metrics such as precision, recall, and F1 depend on the selected decision threshold.

## Future Improvements

Potential next-stage improvements include:

* evaluating additional NASA/PROMISE datasets
* repeated stratified cross-validation
* broader baseline coverage
* ablation studies for feature grouping and attention
* comparison of alternative imbalance-handling strategies
* group-specific projection layers instead of zero-padding
* calibration analysis
* confidence intervals and statistical comparison across repeated runs
* systematic hyperparameter search
* experiment tracking and configuration management
* improved documentation and automated CI checks

## Testing

Run:

```bash
pytest
```

Current status:

```text
9 passed
```

The tests are intended to protect the reproducibility of the data-processing, model, and result-generation pipeline.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Author

**Tuntufye Ulimbakisya Mwakaje**

GitHub: [@tuntufye-mwakaje](https://github.com/tuntufye-mwakaje)

LinkedIn: [tuntufye-mwakaje](https://www.linkedin.com/in/tuntufye-mwakaje/)
