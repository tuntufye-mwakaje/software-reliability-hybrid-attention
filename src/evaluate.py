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
from models import HybridAttentionModel
from preprocessing import (
    create_feature_tokens,
    fit_scale_features,
    split_data,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hybrid_attention_best.pt"
)

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_PATH = (
    RESULTS_DIR
    / "hybrid_attention_results.csv"
)

SEED = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

THRESHOLD = 0.50


def prepare_model_data(df):
    """
    Reproduce the exact train/validation/test
    partitioning and preprocessing used by train.py.

    The scaler is fitted only on the training
    partition. Validation and final test data
    are transformed using that fitted scaler.
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

    X_test_scaled = X_test.astype(float).copy()

    X_test_scaled.loc[:, :] = (
        X_test_scaled_array
    )

    X_train_tokens = create_feature_tokens(
        X_train_scaled
    )

    X_validation_tokens = create_feature_tokens(
        X_validation_scaled
    )

    X_test_tokens = create_feature_tokens(
        X_test_scaled
    )

    return (
        X_train_tokens,
        X_validation_tokens,
        X_test_tokens,
        y_train.to_numpy(dtype=np.float32),
        y_validation.to_numpy(dtype=np.float32),
        y_test.to_numpy(dtype=np.float32),
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


def evaluate_model(
    model,
    X_test,
    y_test,
    threshold=THRESHOLD,
):
    """Evaluate the model on the untouched final test partition."""

    probabilities = get_predictions(
        model,
        X_test,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return {
        "model": "Hybrid Attention",
        "operating_point": (
            f"Threshold {threshold:.2f}"
        ),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": matrix[0, 0],
        "false_positives": matrix[0, 1],
        "false_negatives": matrix[1, 0],
        "true_positives": matrix[1, 1],
        "test_samples": len(y_test),
        "defective_test_samples": int(
            y_test.sum()
        ),
        "threshold": threshold,
    }


def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading KC1 dataset...")

    df = load_kc1()

    print(
        "\nPreparing exact "
        "train/validation/test data..."
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
        "Model parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    print(
        "Model configuration:",
        checkpoint["model_config"],
    )

    print(
        "Best validation loss:",
        checkpoint["best_validation_loss"],
    )

    print(
        f"\nEvaluating at threshold "
        f"{THRESHOLD:.2f}..."
    )

    results = evaluate_model(
        model,
        X_test_tokens,
        y_test,
        threshold=THRESHOLD,
    )

    results_df = pd.DataFrame(
        [results]
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_PATH,
        index=False,
    )

    print("\nHybrid Attention evaluation")
    print("=" * 80)

    print(
        f"ROC AUC     : "
        f"{results['roc_auc']:.6f}"
    )

    print(
        f"PR AUC      : "
        f"{results['pr_auc']:.6f}"
    )

    print(
        f"Accuracy    : "
        f"{results['accuracy']:.6f}"
    )

    print(
        f"Precision   : "
        f"{results['precision']:.6f}"
    )

    print(
        f"Recall      : "
        f"{results['recall']:.6f}"
    )

    print(
        f"F1          : "
        f"{results['f1']:.6f}"
    )

    print("\nConfusion Matrix:")

    print(
        np.array(
            [
                [
                    results["true_negatives"],
                    results["false_positives"],
                ],
                [
                    results["false_negatives"],
                    results["true_positives"],
                ],
            ]
        )
    )

    print(
        "\nEvaluation results saved to:"
    )

    print(RESULTS_PATH)


if __name__ == "__main__":
    main()