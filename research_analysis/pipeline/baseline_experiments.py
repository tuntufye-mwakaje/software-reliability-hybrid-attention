from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier

from dataset_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from research_preprocessing import (
    fit_training_scaler,
    transform_features,
)
from research_data import load_research_dataset


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "results"
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "baseline_repeated_seed_results.csv"
)

OUTPUT_METADATA_FILE = (
    RESULTS_DIR
    / "baseline_repeated_seed_metadata.json"
)

DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

TRAINING_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

SPLIT_SEED = 42

THRESHOLD = 0.5


# ============================================================
# Baseline configuration
# ============================================================

LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 2000,
    "class_weight": None,
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "class_weight": None,
    "n_jobs": -1,
}

MLP_CONFIG = {
    "hidden_layer_sizes": (64, 32),
    "activation": "relu",
    "solver": "adam",
    "alpha": 0.0001,
    "batch_size": 32,
    "learning_rate_init": 0.001,
    "max_iter": 50,
    "early_stopping": False,
    "random_state": None,
}


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Control Python and NumPy randomness.

    The baseline models that use randomness receive the same
    training seed explicitly.
    """

    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Feature preparation
# ============================================================

def prepare_baseline_dataset(dataset_name):
    """
    Prepare one dataset for the classical baselines.

    Uses the frozen seed-42 train/validation/test partitions
    and the established train-only scaling protocol.

    No new split is created.
    """

    (
        train_df,
        validation_df,
        test_df,
    ) = load_research_dataset(
        dataset_name
    )

    # ---------------------------------------------------------
    # Fit scaler using TRAIN ONLY
    # ---------------------------------------------------------

    scaler = fit_training_scaler(
        train_df
    )

    # ---------------------------------------------------------
    # Transform all partitions using the same train scaler
    # ---------------------------------------------------------

    X_train = transform_features(
        train_df,
        scaler,
    ).to_numpy(
        dtype=np.float32
    )

    X_validation = transform_features(
        validation_df,
        scaler,
    ).to_numpy(
        dtype=np.float32
    )

    X_test = transform_features(
        test_df,
        scaler,
    ).to_numpy(
        dtype=np.float32
    )

    # ---------------------------------------------------------
    # Targets
    # ---------------------------------------------------------

    y_train = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int64
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int64
    )

    y_test = test_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.int64
    )

    # ---------------------------------------------------------
    # Return prepared dataset
    # ---------------------------------------------------------

    return {
        "dataset": dataset_name.upper(),

        "train": {
            "X": X_train,
            "y": y_train,
            "row_ids": train_df[
                "row_id"
            ].to_numpy(
                dtype=np.int64
            ),
        },

        "validation": {
            "X": X_validation,
            "y": y_validation,
            "row_ids": validation_df[
                "row_id"
            ].to_numpy(
                dtype=np.int64
            ),
        },

        "test": {
            "X": X_test,
            "y": y_test,
            "row_ids": test_df[
                "row_id"
            ].to_numpy(
                dtype=np.int64
            ),
        },

        "scaler": scaler,
    }

# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    y_true,
    probabilities,
    threshold: float = THRESHOLD,
):
    """
    Calculate probability-ranking and threshold-based metrics.
    """

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    youden_j = (
        sensitivity
        + specificity
        - 1.0
    )

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "youden_j": float(youden_j),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# Logistic Regression
# ============================================================

def train_logistic_regression(
    X_train,
    y_train,
    seed: int,
):
    """
    Train the Logistic Regression baseline.

    Logistic Regression is deterministic for a fixed dataset and
    configuration, but the seed is retained for consistent
    experiment bookkeeping.
    """

    model = LogisticRegression(
        max_iter=LOGISTIC_REGRESSION_CONFIG[
            "max_iter"
        ],
        class_weight=LOGISTIC_REGRESSION_CONFIG[
            "class_weight"
        ],
        random_state=seed,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# Random Forest
# ============================================================

def train_random_forest(
    X_train,
    y_train,
    seed: int,
):
    """
    Train the Random Forest baseline.
    """

    model = RandomForestClassifier(
        n_estimators=RANDOM_FOREST_CONFIG[
            "n_estimators"
        ],
        max_depth=RANDOM_FOREST_CONFIG[
            "max_depth"
        ],
        min_samples_split=RANDOM_FOREST_CONFIG[
            "min_samples_split"
        ],
        min_samples_leaf=RANDOM_FOREST_CONFIG[
            "min_samples_leaf"
        ],
        class_weight=RANDOM_FOREST_CONFIG[
            "class_weight"
        ],
        n_jobs=RANDOM_FOREST_CONFIG[
            "n_jobs"
        ],
        random_state=seed,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# MLP
# ============================================================

def train_mlp(
    X_train,
    y_train,
    seed: int,
):
    """
    Train a conventional feed-forward neural-network baseline.

    The architecture is deliberately separate from the proposed
    Hybrid Attention architecture:

        21 scaled predictors
             |
             v
          Dense 64
             |
             v
          Dense 32
             |
             v
        Binary output
    """

    model = MLPClassifier(
        hidden_layer_sizes=MLP_CONFIG[
            "hidden_layer_sizes"
        ],
        activation=MLP_CONFIG[
            "activation"
        ],
        solver=MLP_CONFIG[
            "solver"
        ],
        alpha=MLP_CONFIG[
            "alpha"
        ],
        batch_size=MLP_CONFIG[
            "batch_size"
        ],
        learning_rate_init=MLP_CONFIG[
            "learning_rate_init"
        ],
        max_iter=MLP_CONFIG[
            "max_iter"
        ],
        early_stopping=MLP_CONFIG[
            "early_stopping"
        ],
        random_state=seed,
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ============================================================
# Probability prediction
# ============================================================

def predict_probabilities(
    model,
    X,
):
    """
    Return positive-class probabilities.
    """

    probabilities = model.predict_proba(
        X
    )[:, 1]

    return np.asarray(
        probabilities,
        dtype=float,
    )


# ============================================================
# Run one baseline model
# ============================================================

def run_model(
    dataset_name: str,
    model_name: str,
    training_seed: int,
    data,
):
    """
    Train and evaluate one baseline model.

    Validation data is evaluated for reference.

    Test data remains locked until after model fitting.
    """

    X_train = data["train"]["X"]
    y_train = data["train"]["y"]

    X_validation = data["validation"]["X"]
    y_validation = data["validation"]["y"]

    X_test = data["test"]["X"]
    y_test = data["test"]["y"]

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    if model_name == "logistic_regression":

        model = train_logistic_regression(
            X_train,
            y_train,
            training_seed,
        )

    elif model_name == "random_forest":

        model = train_random_forest(
            X_train,
            y_train,
            training_seed,
        )

    elif model_name == "mlp":

        model = train_mlp(
            X_train,
            y_train,
            training_seed,
        )

    else:

        raise ValueError(
            f"Unknown baseline model: {model_name}"
        )

    # ---------------------------------------------------------
    # Validation evaluation
    # ---------------------------------------------------------

    validation_probabilities = (
        predict_probabilities(
            model,
            X_validation,
        )
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_probabilities,
    )

    # ---------------------------------------------------------
    # Locked test evaluation
    # ---------------------------------------------------------

    test_probabilities = (
        predict_probabilities(
            model,
            X_test,
        )
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
    )

    # ---------------------------------------------------------
    # Record result
    # ---------------------------------------------------------

    result = {
        "dataset": dataset_name.upper(),

        "model": model_name,

        "training_seed": int(
            training_seed
        ),

        "split_seed": int(
            SPLIT_SEED
        ),

        "threshold": float(
            THRESHOLD
        ),

        "train_samples": int(
            len(y_train)
        ),

        "validation_samples": int(
            len(y_validation)
        ),

        "test_samples": int(
            len(y_test)
        ),

        "train_positive": int(
            y_train.sum()
        ),

        "validation_positive": int(
            y_validation.sum()
        ),

        "test_positive": int(
            y_test.sum()
        ),
    }

    # ---------------------------------------------------------
    # Validation metrics
    # ---------------------------------------------------------

    for metric_name, value in validation_metrics.items():

        result[
            f"validation_{metric_name}"
        ] = value

    # ---------------------------------------------------------
    # Test metrics
    # ---------------------------------------------------------

    for metric_name, value in test_metrics.items():

        result[
            f"test_{metric_name}"
        ] = value

    # ---------------------------------------------------------
    # MLP training information
    # ---------------------------------------------------------

    if model_name == "mlp":

        result["mlp_iterations"] = int(
            model.n_iter_
        )

        result["mlp_final_loss"] = float(
            model.loss_
        )

    else:

        result["mlp_iterations"] = None
        result["mlp_final_loss"] = None

    return result


# ============================================================
# Partition consistency
# ============================================================

def check_partition_consistency(
    dataset_name: str,
    data,
):
    """
    Verify that the frozen partition sizes remain unchanged.
    """

    expected = {
        "CM1": (309, 84, 105),
        "JM1": (8491, 2190, 2523),
        "KC1": (1370, 307, 432),
        "KC2": (346, 66, 110),
        "PC1": (722, 171, 216),
    }

    actual = (
        len(data["train"]["y"]),
        len(data["validation"]["y"]),
        len(data["test"]["y"]),
    )

    if actual != expected[
        dataset_name.upper()
    ]:

        raise ValueError(
            f"{dataset_name}: frozen partition size "
            f"mismatch. Expected "
            f"{expected[dataset_name.upper()]}, "
            f"found {actual}."
        )


# ============================================================
# Main experiment
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print(
        "BASELINE REPEATED-SEED EXPERIMENTS"
    )
    print("=" * 70)

    print()
    print(
        "Datasets:",
        ", ".join(DATASETS),
    )

    print(
        "Training seeds:",
        ", ".join(
            str(seed)
            for seed in TRAINING_SEEDS
        ),
    )

    print(
        "Frozen split seed:",
        SPLIT_SEED,
    )

    print()
    print(
        "Baseline models:"
    )

    print(
        "  1. Logistic Regression"
    )

    print(
        "  2. Random Forest"
    )

    print(
        "  3. MLP"
    )

    print()
    print(
        "Feature representation:"
    )

    print(
        f"  {len(FEATURE_COLUMNS)} standardized canonical predictors"
    )

    print(
        "Scaling: fit on training partition only"
    )

    print(
        "Threshold:",
        THRESHOLD,
    )

    print()

    all_results = []

    # ---------------------------------------------------------
    # Dataset loop
    # ---------------------------------------------------------

    for dataset_name in DATASETS:

        print()
        print("-" * 70)
        print(
            f"Preparing dataset: {dataset_name}"
        )
        print("-" * 70)

        data = prepare_baseline_dataset(
            dataset_name
        )

        check_partition_consistency(
            dataset_name,
            data,
        )

        print(
            f"Train:      {len(data['train']['y'])}"
        )

        print(
            f"Validation: {len(data['validation']['y'])}"
        )

        print(
            f"Test:       {len(data['test']['y'])}"
        )

        print(
            f"Features:   {data['train']['X'].shape[1]}"
        )

        # -----------------------------------------------------
        # Training-seed loop
        # -----------------------------------------------------

        for training_seed in TRAINING_SEEDS:

            set_seed(
                training_seed
            )

            print()
            print(
                f"{dataset_name} | "
                f"Training seed {training_seed}"
            )

            # -------------------------------------------------
            # Logistic Regression
            # -------------------------------------------------

            print(
                "  Running Logistic Regression..."
            )

            lr_result = run_model(
                dataset_name,
                "logistic_regression",
                training_seed,
                data,
            )

            all_results.append(
                lr_result
            )

            print(
                f"    Test ROC-AUC: "
                f"{lr_result['test_roc_auc']:.4f} | "
                f"PR-AUC: "
                f"{lr_result['test_pr_auc']:.4f}"
            )

            # -------------------------------------------------
            # Random Forest
            # -------------------------------------------------

            print(
                "  Running Random Forest..."
            )

            rf_result = run_model(
                dataset_name,
                "random_forest",
                training_seed,
                data,
            )

            all_results.append(
                rf_result
            )

            print(
                f"    Test ROC-AUC: "
                f"{rf_result['test_roc_auc']:.4f} | "
                f"PR-AUC: "
                f"{rf_result['test_pr_auc']:.4f}"
            )

            # -------------------------------------------------
            # MLP
            # -------------------------------------------------

            print(
                "  Running MLP..."
            )

            mlp_result = run_model(
                dataset_name,
                "mlp",
                training_seed,
                data,
            )

            all_results.append(
                mlp_result
            )

            print(
                f"    Test ROC-AUC: "
                f"{mlp_result['test_roc_auc']:.4f} | "
                f"PR-AUC: "
                f"{mlp_result['test_pr_auc']:.4f}"
            )

    # ---------------------------------------------------------
    # Save results
    # ---------------------------------------------------------

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------

    metadata = {
        "experiment": (
            "repeated_seed_baseline_comparison"
        ),

        "datasets": DATASETS,

        "training_seeds": TRAINING_SEEDS,

        "split_seed": SPLIT_SEED,

        "number_of_datasets": len(
            DATASETS
        ),

        "training_seeds_per_dataset": len(
            TRAINING_SEEDS
        ),

        "models": [
            "logistic_regression",
            "random_forest",
            "mlp",
        ],

        "total_runs": int(
            len(results_df)
        ),

        "runs_per_dataset": int(
            len(TRAINING_SEEDS) * 3
        ),

        "feature_count": int(
            len(FEATURE_COLUMNS)
        ),

        "feature_representation": (
            "21 canonical standardized predictors"
        ),

        "feature_columns": list(
            FEATURE_COLUMNS
        ),

        "scaling": (
            "StandardScaler fitted on training "
            "partition only"
        ),

        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),

        "threshold": THRESHOLD,

        "test_usage": (
            "final evaluation only"
        ),

        "hyperparameter_tuning": False,

        "model_selection": (
            "fixed baseline configurations; "
            "no test-set tuning"
        ),

        "logistic_regression": (
            LOGISTIC_REGRESSION_CONFIG
        ),

        "random_forest": (
            {
                key: value
                for key, value
                in RANDOM_FOREST_CONFIG.items()
            }
        ),

        "mlp": (
            {
                key: (
                    list(value)
                    if isinstance(value, tuple)
                    else value
                )
                for key, value
                in MLP_CONFIG.items()
                if key != "random_state"
            }
        ),

        "purpose": (
            "Controlled comparison of conventional "
            "classification baselines against the "
            "proposed Hybrid Attention model."
        ),
    }

    with open(
        OUTPUT_METADATA_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "BASELINE EXPERIMENTS COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        f"Total experiment rows: "
        f"{len(results_df)}"
    )

    print()
    print(
        "Expected:"
    )

    print(
        f"  {len(DATASETS)} datasets × "
        f"{len(TRAINING_SEEDS)} seeds × "
        "3 models = "
        f"{len(DATASETS) * len(TRAINING_SEEDS) * 3}"
    )

    print()
    print(
        "Test ROC-AUC by dataset and model:"
    )

    summary = (
        results_df
        .groupby(
            ["dataset", "model"]
        )[
            "test_roc_auc"
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Test PR-AUC by dataset and model:"
    )

    summary_pr = (
        results_df
        .groupby(
            ["dataset", "model"]
        )[
            "test_pr_auc"
        ]
        .agg(
            [
                "mean",
                "std",
            ]
        )
        .reset_index()
    )

    print(
        summary_pr.to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print()
    print(
        "Metadata saved to:"
    )

    print(
        OUTPUT_METADATA_FILE
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
