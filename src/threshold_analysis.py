from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data_loader import load_kc1
from preprocessing import (
    split_data,
    fit_scale_features,
    create_feature_tokens,
)
from sklearn.model_selection import train_test_split
from models import HybridAttentionModel


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hybrid_attention_best.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"

THRESHOLD_CURVE_PATH = (
    RESULTS_DIR
    / "threshold_analysis.csv"
)

SELECTED_RESULT_PATH = (
    RESULTS_DIR
    / "hybrid_attention_threshold_selected.csv"
)

SEED = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

THRESHOLDS = np.arange(
    0.10,
    0.91,
    0.01,
)


def prepare_model_data(df):
    """
    Reproduce the exact train/validation/test
    partitioning and preprocessing used by train.py.

    The final test set remains untouched during
    threshold selection.
    """

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = split_data(
        df,
        test_size=TEST_SIZE,
        random_state=SEED,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=y_train_full,
    )

    (
        X_train_scaled,
        X_validation_scaled,
        scaler,
    ) = fit_scale_features(
        X_train,
        X_validation,
    )

    X_test_scaled_array = scaler.transform(
        X_test
    )

    X_test_scaled = (
        X_test.astype(float).copy()
    )

    X_test_scaled.loc[:, :] = (
        X_test_scaled_array
    )

    X_train_tokens = create_feature_tokens(
        X_train_scaled
    )

    X_validation_tokens = (
        create_feature_tokens(
            X_validation_scaled
        )
    )

    X_test_tokens = create_feature_tokens(
        X_test_scaled
    )

    return (
        X_train_tokens,
        X_validation_tokens,
        X_test_tokens,
        y_train.to_numpy(
            dtype=np.float32
        ),
        y_validation.to_numpy(
            dtype=np.float32
        ),
        y_test.to_numpy(
            dtype=np.float32
        ),
        scaler,
    )


def load_model():
    """Load the trained Hybrid Attention model."""

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    config = checkpoint["model_config"]

    model = HybridAttentionModel(
        input_dim=config["input_dim"],
        embedding_dim=config["embedding_dim"],
        num_heads=config["num_heads"],
        dropout=config["dropout"],
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model, checkpoint


def get_predictions(model, X):
    """Return predicted probabilities."""

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(X_tensor)

        probabilities = (
            torch.sigmoid(logits)
            .numpy()
        )

    return probabilities


def calculate_metrics(
    y_true,
    probabilities,
    threshold,
):
    """Calculate classification metrics."""

    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "true_negatives": matrix[0, 0],
        "false_positives": matrix[0, 1],
        "false_negatives": matrix[1, 0],
        "true_positives": matrix[1, 1],
    }


def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading KC1 dataset...")

    df = load_kc1()

    print(
        "Preparing train/validation/test data..."
    )

    (
        X_train_tokens,
        X_validation_tokens,
        X_test_tokens,
        y_train,
        y_validation,
        y_test,
        scaler,
    ) = prepare_model_data(df)

    print(
        "Training data:",
        X_train_tokens.shape,
    )

    print(
        "Validation data:",
        X_validation_tokens.shape,
    )

    print(
        "Final test data:",
        X_test_tokens.shape,
    )

    print("\nTarget distributions:")

    print(
        "Training:",
        {
            0: int(
                (y_train == 0).sum()
            ),
            1: int(
                (y_train == 1).sum()
            ),
        },
    )

    print(
        "Validation:",
        {
            0: int(
                (y_validation == 0).sum()
            ),
            1: int(
                (y_validation == 1).sum()
            ),
        },
    )

    print(
        "Test:",
        {
            0: int(
                (y_test == 0).sum()
            ),
            1: int(
                (y_test == 1).sum()
            ),
        },
    )

    print("\nLoading trained model...")

    model, checkpoint = load_model()

    print(
        "Model configuration:",
        checkpoint["model_config"],
    )

    print(
        "Best validation loss:",
        checkpoint["best_validation_loss"],
    )

    validation_probabilities = (
        get_predictions(
            model,
            X_validation_tokens,
        )
    )

    test_probabilities = (
        get_predictions(
            model,
            X_test_tokens,
        )
    )

    print(
        "\nSelecting threshold using "
        "validation F1..."
    )

    threshold_results = []

    for threshold in THRESHOLDS:

        metrics = calculate_metrics(
            y_validation,
            validation_probabilities,
            float(threshold),
        )

        threshold_results.append(
            metrics
        )

    threshold_df = pd.DataFrame(
        threshold_results
    )

    threshold_df.to_csv(
        THRESHOLD_CURVE_PATH,
        index=False,
    )

    best_index = threshold_df[
        "f1"
    ].idxmax()

    best_row = threshold_df.loc[
        best_index
    ]

    selected_threshold = float(
        best_row["threshold"]
    )

    print(
        f"Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation F1: "
        f"{best_row['f1']:.6f}"
    )

    print(
        f"Validation precision: "
        f"{best_row['precision']:.6f}"
    )

    print(
        f"Validation recall: "
        f"{best_row['recall']:.6f}"
    )

    print(
        "\nApplying selected threshold "
        "to untouched final test set..."
    )

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    selected_result = {
        "model": "Hybrid Attention",
        "operating_point": (
            "Validation-selected "
            f"threshold {selected_threshold:.2f}"
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": test_metrics[
            "accuracy"
        ],
        "precision": test_metrics[
            "precision"
        ],
        "recall": test_metrics[
            "recall"
        ],
        "f1": test_metrics[
            "f1"
        ],
        "true_negatives": test_metrics[
            "true_negatives"
        ],
        "false_positives": test_metrics[
            "false_positives"
        ],
        "false_negatives": test_metrics[
            "false_negatives"
        ],
        "true_positives": test_metrics[
            "true_positives"
        ],
        "test_samples": len(y_test),
        "defective_test_samples": int(
            y_test.sum()
        ),
        "threshold": selected_threshold,
        "validation_f1": float(
            best_row["f1"]
        ),
        "validation_precision": float(
            best_row["precision"]
        ),
        "validation_recall": float(
            best_row["recall"]
        ),
    }

    selected_df = pd.DataFrame(
        [selected_result]
    )

    selected_df.to_csv(
        SELECTED_RESULT_PATH,
        index=False,
    )

    print("\nFinal test results")
    print("=" * 80)

    print(
        f"ROC AUC     : {roc_auc:.6f}"
    )

    print(
        f"PR AUC      : {pr_auc:.6f}"
    )

    print(
        f"Accuracy    : "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision   : "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Recall      : "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"F1          : "
        f"{test_metrics['f1']:.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        np.array(
            [
                [
                    test_metrics[
                        "true_negatives"
                    ],
                    test_metrics[
                        "false_positives"
                    ],
                ],
                [
                    test_metrics[
                        "false_negatives"
                    ],
                    test_metrics[
                        "true_positives"
                    ],
                ],
            ]
        )
    )

    print(
        "\nThreshold curve saved to:"
    )

    print(THRESHOLD_CURVE_PATH)

    print(
        "\nSelected-threshold result "
        "saved to:"
    )

    print(SELECTED_RESULT_PATH)


if __name__ == "__main__":
    main()