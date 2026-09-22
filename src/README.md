# Source Code



This directory contains the implementation of the software reliability prediction pipeline.



## Module Overview



| Module | Purpose |

|---|---|

| `data_loader.py` | Loads and validates the NASA/PROMISE KC1 dataset |

| `preprocessing.py` | Defines feature groups, performs leakage-aware data splitting and training-only scaling, and creates feature-group tokens |

| `models.py` | Defines the hybrid attention neural network architecture |

| `train.py` | Trains the hybrid attention model and saves the best checkpoint and training history |

| `evaluate.py` | Evaluates the trained model on the final test set |

| `baseline_models.py` | Trains and evaluates Logistic Regression and Random Forest baseline models |

| `threshold_analysis.py` | Selects a classification threshold using validation data and evaluates the selected threshold on the final test set |

| `attention_analysis.py` | Extracts and analyzes feature-group attention weights |

| `compare_models.py` | Combines model evaluation results for comparison and generates comparison visualizations |

| `plot_evaluation.py` | Generates evaluation plots such as ROC, precision-recall, and confusion-matrix figures |

| `feature_profile.py` | Provides feature-level profiling utilities for the KC1 dataset |



## Pipeline



The main experimental workflow is:



```text

KC1 Dataset

&#x20;   |

&#x20;   v

Data Loading \& Validation

&#x20;   |

&#x20;   v

Train / Validation / Test Split

&#x20;   |

&#x20;   v

Training-only Feature Scaling

&#x20;   |

&#x20;   v

Semantic Feature Groups

&#x20;   |

&#x20;   v

Feature-group Tokens

&#x20;   |

&#x20;   v

Learned Feature Embeddings

&#x20;   |

&#x20;   v

Multi-head Self-Attention

&#x20;   |

&#x20;   v

Classification Head

&#x20;   |

&#x20;   v

Validation-based Threshold Selection

&#x20;   |

&#x20;   v

Final Test Evaluation
