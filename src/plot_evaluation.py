from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix,
)

from data_loader import load_kc1
from evaluate import load_model
from train import prepare_model_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

THRESHOLD_RESULTS_PATH = (
    RESULTS_DIR
    / "threshold_analysis.csv"
)

SELECTED_THRESHOLD = 0.18


def predict_probabilities(
    model,
    features,
):
    """Generate model probabilities."""

    tensor = torch.tensor(
        features,
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():
        logits = model(tensor)

        probabilities = (
            torch.sigmoid(logits)
            .cpu()
            .numpy()
        )

    return probabilities


def save_roc_curve(
    y_test,
    probabilities,
):
    """Save the ROC curve for the untouched final test set."""

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
        ax=axis,
    )

    axis.set_title(
        "Hybrid Attention Model — ROC Curve"
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "roc_curve.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_precision_recall_curve(
    y_test,
    probabilities,
):
    """Save the Precision-Recall curve for the final test set."""

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    PrecisionRecallDisplay.from_predictions(
        y_test,
        probabilities,
        ax=axis,
    )

    axis.set_title(
        "Hybrid Attention Model — Precision-Recall Curve"
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "precision_recall_curve.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_threshold_f1_curve(
    threshold_results,
):
    """Save validation F1 across classification thresholds."""

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        threshold_results["threshold"],
        threshold_results["f1"],
        marker="o",
        markersize=3,
        label="Validation F1",
    )

    axis.axvline(
        SELECTED_THRESHOLD,
        linestyle="--",
        label=(
            f"Selected threshold = "
            f"{SELECTED_THRESHOLD:.2f}"
        ),
    )

    axis.set_xlabel(
        "Classification Threshold"
    )

    axis.set_ylabel(
        "Validation F1"
    )

    axis.set_title(
        "Validation F1 Across Classification Thresholds"
    )

    axis.legend()

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "validation_f1_threshold.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_precision_recall_threshold_curve(
    threshold_results,
):
    """Save validation precision and recall across thresholds."""

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        threshold_results["threshold"],
        threshold_results["precision"],
        marker="o",
        markersize=3,
        label="Precision",
    )

    axis.plot(
        threshold_results["threshold"],
        threshold_results["recall"],
        marker="o",
        markersize=3,
        label="Recall",
    )

    axis.axvline(
        SELECTED_THRESHOLD,
        linestyle="--",
        label=(
            f"Selected threshold = "
            f"{SELECTED_THRESHOLD:.2f}"
        ),
    )

    axis.set_xlabel(
        "Classification Threshold"
    )

    axis.set_ylabel(
        "Metric"
    )

    axis.set_title(
        "Validation Precision and Recall Across Thresholds"
    )

    axis.legend()

    axis.grid(
        True,
        alpha=0.3,
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "validation_precision_recall_threshold.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def save_confusion_matrix(
    y_test,
    probabilities,
):
    """Save the final test confusion matrix at the selected threshold."""

    predictions = (
        probabilities >= SELECTED_THRESHOLD
    ).astype(int)

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Non-defective",
            "Defective",
        ],
    )

    display.plot(
        ax=axis,
        values_format="d",
    )

    axis.set_title(
        "Final Test Confusion Matrix "
        f"(Threshold = {SELECTED_THRESHOLD:.2f})"
    )

    figure.tight_layout()

    output_path = (
        FIGURES_DIR
        / "test_confusion_matrix.png"
    )

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    return output_path


def main():
    """Generate evaluation figures from the current clean experiment."""

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading KC1 dataset...")

    df = load_kc1()

    print(
        "\nPreparing exact "
        "train/validation/test data..."
    )

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = prepare_model_data(df)

    print(
        "Training data:",
        X_train.shape,
    )

    print(
        "Validation data:",
        X_validation.shape,
    )

    print(
        "Final test data:",
        X_test.shape,
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

    print(
        "\nGenerating final test probabilities..."
    )

    test_probabilities = predict_probabilities(
        model,
        X_test,
    )

    print(
        "Loading current threshold analysis..."
    )

    threshold_results = pd.read_csv(
        THRESHOLD_RESULTS_PATH
    )

    print(
        "\nGenerating synchronized figures..."
    )

    generated_files = []

    generated_files.append(
        save_roc_curve(
            y_test,
            test_probabilities,
        )
    )

    generated_files.append(
        save_precision_recall_curve(
            y_test,
            test_probabilities,
        )
    )

    generated_files.append(
        save_threshold_f1_curve(
            threshold_results
        )
    )

    generated_files.append(
        save_precision_recall_threshold_curve(
            threshold_results
        )
    )

    generated_files.append(
        save_confusion_matrix(
            y_test,
            test_probabilities,
        )
    )

    print(
        "\nGenerated figures"
    )

    print(
        "=" * 70
    )

    for file_path in generated_files:
        print(file_path)

    print(
        "\nEvaluation visualization completed."
    )


if __name__ == "__main__":
    main()