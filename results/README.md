# Results and Evaluation Artifacts

This directory contains the reproducible outputs from the current clean experiment for the **Hybrid Attention-Based Deep Learning Framework for Software Reliability Prediction** using the NASA/PROMISE KC1 software defect dataset.

The artifacts document model training, final test evaluation, threshold analysis, baseline comparison, and attention analysis.

---

## 1. Experimental Protocol

The current experiment uses:

- Dataset: NASA/PROMISE KC1
- OpenML dataset ID: 1067
- Samples: 2,109
- Input features: 21
- Target: `defects`
- Positive class: 326 defective instances
- Negative class: 1,783 non-defective instances
- Random seed: 42

The data is divided using stratified sampling:

| Partition | Samples | Defective | Non-defective |
|---|---:|---:|---:|
| Training | 1,349 | 209 | 1,140 |
| Validation | 338 | 52 | 286 |
| Final test | 422 | 65 | 357 |

The final test set is kept untouched during model training and threshold selection.

Feature scaling is fitted using the training partition only and subsequently applied to the validation and final test partitions.

The model represents the 21 software metrics as four semantic feature groups:

1. Size
2. Complexity
3. Halstead
4. Operators / Operands

These groups are a representation design used by this project rather than an official KC1 taxonomy.

---

## 2. Hybrid Attention Model

The model uses:

- Four semantic feature-group tokens
- Input token width: 8
- Learned embedding dimension: 32
- Multi-head self-attention
- Four attention heads
- Feed-forward residual block
- Layer normalization
- GELU activations
- Dropout: 0.1
- Binary classification output

Total trainable parameters:

````text
9,985
````

The model receives tensors with shape:

````text
(batch_size, 4, 8)
````

The four tokens correspond to the project's semantic feature groups. Self-attention is applied across these group-level representations before classification.

---

## 3. Training

Training configuration:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Initial learning rate | 0.001 |
| Weight decay | 0.0001 |
| Batch size | 32 |
| Maximum epochs | 50 |
| Early stopping patience | 10 |
| Gradient clipping | 1.0 |
| LR scheduler | ReduceLROnPlateau |
| Scheduler factor | 0.5 |
| Scheduler patience | 3 |
| Minimum learning rate | 0.000001 |
| Random seed | 42 |
| Loss | BCEWithLogitsLoss |

Training stopped early at epoch 16.

The best validation loss was:

````text
0.345243
````

The complete epoch-by-epoch training history is stored in:

````text
training_history.csv
````

The corresponding visualization is:

````text
figures/training_history.png
````

---

## 4. Hybrid Attention Ã¢â‚¬â€ Default Threshold

At the conventional classification threshold of `0.50`, the untouched final test set produced:

| Metric | Result |
|---|---:|
| ROC AUC | 0.796639 |
| PR AUC | 0.422107 |
| Accuracy | 0.855450 |
| Precision | 0.750000 |
| Recall | 0.092308 |
| F1 | 0.164384 |

Confusion matrix:

````text
[[355   2]
 [ 59   6]]
````

This operating point produces high precision but low recall for the defective class.

The results are stored in:

````text
hybrid_attention_results.csv
````

---

## 5. Validation-Based Threshold Selection

Because KC1 is imbalanced, the project also evaluates the effect of changing the classification threshold.

Threshold selection is performed **only on the validation partition**.

The threshold producing the highest validation F1 was:

````text
0.18
````

Validation performance at this threshold:

| Metric | Validation Result |
|---|---:|
| Precision | 0.336066 |
| Recall | 0.788462 |
| F1 | 0.471264 |

The selected threshold was then applied once to the untouched final test set.

### Final Test Performance at Threshold 0.18

| Metric | Result |
|---|---:|
| ROC AUC | 0.796639 |
| PR AUC | 0.422107 |
| Accuracy | 0.748815 |
| Precision | 0.348148 |
| Recall | 0.723077 |
| F1 | 0.470000 |

Confusion matrix:

````text
[[269  88]
 [ 18  47]]
````

Compared with the default `0.50` threshold, the selected threshold substantially increases recall while also increasing false positives.

The complete threshold analysis is stored in:

````text
threshold_analysis.csv
````

The final test result using the validation-selected threshold is stored in:

````text
hybrid_attention_threshold_selected.csv
````

Related figures:

````text
figures/validation_f1_threshold.png
figures/validation_precision_recall_threshold.png
figures/test_confusion_matrix.png
````

---

## 6. Baseline Models

Two conventional machine-learning baselines were evaluated using the same final test partition:

- Logistic Regression
- Random Forest

Their final test results at threshold `0.50` were:

| Model | ROC AUC | PR AUC | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.789528 | 0.479546 | 0.755924 | 0.346774 | 0.661538 | 0.455026 |
| Random Forest | 0.809761 | 0.489432 | 0.872038 | 0.666667 | 0.338462 | 0.448980 |

The baseline results are stored in:

````text
baseline_results.csv
````

---

## 7. Model Comparison

The current comparison includes both the default and validation-selected operating points for the hybrid model.

| Model | Operating Point | ROC AUC | PR AUC | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Threshold 0.50 | 0.789528 | 0.479546 | 0.755924 | 0.346774 | 0.661538 | 0.455026 |
| Random Forest | Threshold 0.50 | 0.809761 | 0.489432 | 0.872038 | 0.666667 | 0.338462 | 0.448980 |
| Hybrid Attention | Threshold 0.50 | 0.796639 | 0.422107 | 0.855450 | 0.750000 | 0.092308 | 0.164384 |
| Hybrid Attention | Validation-selected threshold 0.18 | 0.796639 | 0.422107 | 0.748815 | 0.348148 | 0.723077 | 0.470000 |

ROC AUC and PR AUC are threshold-independent metrics. Precision, recall, F1, and accuracy depend on the selected classification threshold.

The comparison is therefore reported as an evaluation of different operating points rather than as a claim that one model universally outperforms another.

The comparison table is stored in:

````text
model_comparison.csv
````

The visualization is:

````text
figures/model_comparison.png
````

---

## 8. Attention Analysis

The model's learned attention weights were examined on the final test partition.

The average attention matrix was:

| From / To | Size | Complexity | Halstead | Operators / Operands |
|---|---:|---:|---:|---:|
| Size | 0.178891 | 0.194651 | 0.420307 | 0.206151 |
| Complexity | 0.175452 | 0.191753 | 0.425351 | 0.207444 |
| Halstead | 0.192779 | 0.196912 | 0.394670 | 0.215639 |
| Operators / Operands | 0.180989 | 0.193543 | 0.422288 | 0.203180 |

The largest individual average attention value is:

````text
Complexity Ã¢â€ â€™ Halstead = 0.425351
````

Halstead also receives relatively high attention from the other feature groups.

These values describe **model attention behavior**, not causal feature importance or proof that a feature group independently causes software defects.

The raw attention matrix is stored in:

````text
attention_weights.csv
````

The visualization is:

````text
figures/attention_heatmap.png
````

---

## 9. Evaluation Figures

The following figures are generated from the current clean experiment:

````text
figures/
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ attention_heatmap.png
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ model_comparison.png
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ precision_recall_curve.png
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ roc_curve.png
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ test_confusion_matrix.png
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ training_history.png
Ã¢â€Å“Ã¢â€â‚¬Ã¢â€â‚¬ validation_f1_threshold.png
Ã¢â€â€Ã¢â€â‚¬Ã¢â€â‚¬ validation_precision_recall_threshold.png
````

These figures provide visual evidence for:

- Training and validation loss
- ROC performance
- Precision-recall behavior
- Threshold selection
- Confusion-matrix behavior
- Baseline comparison
- Feature-group attention patterns

---

## 10. Reproducibility

The experiment can be reproduced from the project root using the project source modules.

Training:

```powershell
python ".\src\train.py"
```

Default-threshold evaluation:

```powershell
python ".\src\evaluate.py"
```

Threshold analysis:

```powershell
python ".\src\threshold_analysis.py"
```

Baseline evaluation:

```powershell
python ".\src\baseline_models.py"
```

Model comparison:

```powershell
python ".\src\compare_models.py"
```

Attention analysis:

```powershell
python ".\src\attention_analysis.py"
```

Evaluation figures:

```powershell
python ".\src\plot_evaluation.py"
```

Automated pipeline tests:

```powershell
python -m pytest ".\tests\test_pipeline.py" -v
```

The current test suite verifies dataset structure, partitioning, preprocessing, token construction, model shapes, checkpoint loading, result artifacts, and training-history structure.

---

## 11. Interpretation Notes

The dataset contains substantially more non-defective than defective instances:

````text
Non-defective: 1,783
Defective:       326
````

Therefore, accuracy alone does not adequately describe defective-instance detection.

The default threshold of `0.50` produces high precision but low recall. Selecting the threshold using validation F1 changes the operating point toward higher recall, with a corresponding increase in false positives.

The baseline comparison also demonstrates why multiple metrics are necessary. Random Forest has higher ROC AUC and PR AUC than the hybrid model in this experiment, while the validation-selected hybrid threshold produces a slightly higher test F1 than the two baseline F1 values shown at their default threshold.

These observations are specific to this experiment and dataset split. They should not be interpreted as evidence of universal model superiority.

---

## 12. Current Experimental Limitations

Important limitations of the current experiment include:

1. The study uses a single software-defect dataset.
2. The primary experiment uses one fixed random seed.
3. The semantic feature grouping is a project-defined representation rather than an official KC1 taxonomy.
4. The current token representation pads smaller feature groups to a common width.
5. The attention analysis reports averaged attention weights rather than separate per-head weights.
6. No claim of causal feature importance is made from the attention values.
7. The baseline comparison does not constitute a comprehensive hyperparameter search for every model.
8. External validation on additional NASA/PROMISE datasets has not yet been performed.

These limitations should be considered when interpreting the reported results.

---

## 13. Artifact Summary

| Artifact | Purpose |
|---|---|
| `training_history.csv` | Epoch-level training and validation losses |
| `hybrid_attention_results.csv` | Hybrid model test evaluation at threshold 0.50 |
| `hybrid_attention_threshold_selected.csv` | Hybrid model test evaluation at validation-selected threshold |
| `threshold_analysis.csv` | Validation threshold sweep and test application |
| `baseline_results.csv` | Logistic Regression and Random Forest results |
| `model_comparison.csv` | Consolidated model comparison |
| `attention_weights.csv` | Average feature-group attention matrix |
| `figures/training_history.png` | Training history visualization |
| `figures/roc_curve.png` | ROC curves |
| `figures/precision_recall_curve.png` | Precision-recall curves |
| `figures/validation_f1_threshold.png` | Validation F1 across thresholds |
| `figures/validation_precision_recall_threshold.png` | Validation precision/recall across thresholds |
| `figures/test_confusion_matrix.png` | Final test confusion matrix |
| `figures/model_comparison.png` | Model comparison visualization |
| `figures/attention_heatmap.png` | Feature-group attention visualization |