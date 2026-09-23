from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from research_preprocessing import prepare_research_dataset
from research_models import HybridAttentionModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "research_analysis" / "results"
MODELS_DIR = PROJECT_ROOT / "research_analysis" / "models"

DATASETS = ["CM1", "JM1", "KC1", "KC2", "PC1"]

# Threshold grid used for operating-point analysis.
# 0.001 resolution gives sufficiently fine coverage while remaining
# deterministic and easy to reproduce.
THRESHOLDS = np.round(np.arange(0.001, 1.000, 0.001), 3)

DEVICE = torch.device("cpu")


def load_checkpoint(dataset_name):
    """
    Load the frozen seed-42 checkpoint for one dataset.

    The model architecture must exactly match the architecture used
    when the checkpoint was trained.

    Architecture:
        input_dim=8
        embedding_dim=32
        num_heads=4
        dropout=0.1

    The checkpoint is loaded on CPU for deterministic evaluation.
    """

    dataset_name = dataset_name.upper()

    checkpoint_path = (
        MODELS_DIR / f"{dataset_name.lower()}_seed42_best.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False,
    )

    # -------------------------------------------------------------
    # Reconstruct the exact model architecture used during training.
    #
    # IMPORTANT:
    # HybridAttentionModel uses the parameter name
    # "embedding_dim", not "embed_dim".
    # -------------------------------------------------------------

    model = HybridAttentionModel(
        input_dim=8,
        embedding_dim=32,
        num_heads=4,
        dropout=0.1,
    )

    # -------------------------------------------------------------
    # Validate checkpoint structure before loading weights.
    # -------------------------------------------------------------

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"{dataset_name}: checkpoint does not contain "
            "'model_state_dict'"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(DEVICE)
    model.eval()

    return model

def predict_probabilities(model, tokens):
    x = torch.tensor(
        tokens,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(x)

        if isinstance(logits, tuple):
            logits = logits[0]

        probabilities = torch.sigmoid(logits)

    return probabilities.cpu().numpy()


def classification_metrics(y_true, probabilities, threshold):
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, predictions)
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "youden_j": float(
            (
                recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                )
                +
                (
                    tn / (tn + fp)
                    if (tn + fp) > 0
                    else 0.0
                )
                - 1.0
            )
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def select_threshold(metrics_df, criterion):

    if criterion == "max_f1":
        score_column = "f1"

    elif criterion == "max_balanced_accuracy":
        score_column = "balanced_accuracy"

    elif criterion == "max_youden_j":
        score_column = "youden_j"

    else:
        raise ValueError(
            f"Unknown threshold criterion: {criterion}"
        )

    maximum = metrics_df[score_column].max()

    # Deterministic tie-breaking:
    # when multiple thresholds have exactly the same criterion
    # score, choose the highest threshold. This avoids unnecessarily
    # lowering the operating threshold in a tied solution.
    candidates = metrics_df[
        np.isclose(
            metrics_df[score_column],
            maximum,
            rtol=1e-12,
            atol=1e-12,
        )
    ]

    selected = candidates.sort_values(
        "threshold",
        ascending=False,
    ).iloc[0]

    return float(selected["threshold"])


def evaluate_dataset(dataset_name):

    print(f"\n{'-' * 72}")
    print(f"DATASET: {dataset_name}")
    print(f"{'-' * 72}")

    # -------------------------------------------------------------
    # Reproduce the exact preprocessing used during checkpoint
    # training.
    #
    # prepare_research_dataset():
    #   1. reconstructs the frozen partitions
    #   2. fits StandardScaler on TRAIN ONLY
    #   3. transforms train/validation/test using that scaler
    #   4. creates the four semantic feature-group tokens
    #   5. returns tokens with shape (samples, 4, 8)
    # -------------------------------------------------------------

    prepared = prepare_research_dataset(
        dataset_name
    )

    train = prepared["train"]
    validation = prepared["validation"]
    test = prepared["test"]

    # -------------------------------------------------------------
    # Load the frozen seed-42 checkpoint
    # -------------------------------------------------------------

    model = load_checkpoint(
        dataset_name
    )

    # -------------------------------------------------------------
    # Generate predictions.
    #
    # Threshold selection is performed ONLY on validation data.
    # Test data remains completely held out until after threshold
    # selection.
    # -------------------------------------------------------------

    validation_probabilities = predict_probabilities(
        model,
        validation["tokens"],
    )

    test_probabilities = predict_probabilities(
        model,
        test["tokens"],
    )

    # -------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------

    validation_target = np.asarray(
        validation["target"],
        dtype=int,
    )

    test_target = np.asarray(
        test["target"],
        dtype=int,
    )

    # -------------------------------------------------------------
    # Evaluate every candidate threshold on VALIDATION only
    # -------------------------------------------------------------

    validation_metrics = []

    for threshold in THRESHOLDS:

        metrics = classification_metrics(
            validation_target,
            validation_probabilities,
            threshold,
        )

        validation_metrics.append(
            metrics
        )

    validation_metrics_df = pd.DataFrame(
        validation_metrics
    )

    # -------------------------------------------------------------
    # Select thresholds using validation performance only
    # -------------------------------------------------------------

    selected_thresholds = {
        "max_f1": select_threshold(
            validation_metrics_df,
            "max_f1",
        ),
        "max_balanced_accuracy": select_threshold(
            validation_metrics_df,
            "max_balanced_accuracy",
        ),
        "max_youden_j": select_threshold(
            validation_metrics_df,
            "max_youden_j",
        ),
    }

    # -------------------------------------------------------------
    # Validation performance at selected thresholds
    # -------------------------------------------------------------

    validation_selected_rows = []

    for criterion, threshold in selected_thresholds.items():

        metrics = classification_metrics(
            validation_target,
            validation_probabilities,
            threshold,
        )

        row = {
            "dataset": dataset_name,
            "criterion": criterion,
            **metrics,
        }

        validation_selected_rows.append(
            row
        )

    validation_selected_df = pd.DataFrame(
        validation_selected_rows
    )

    # -------------------------------------------------------------
    # Final evaluation on HELD-OUT TEST data
    #
    # The thresholds below were determined entirely from the
    # validation partition.
    # -------------------------------------------------------------

    test_rows = []

    for criterion, threshold in selected_thresholds.items():

        metrics = classification_metrics(
            test_target,
            test_probabilities,
            threshold,
        )

        row = {
            "dataset": dataset_name,
            "criterion": criterion,
            **metrics,
        }

        test_rows.append(
            row
        )

    test_metrics_df = pd.DataFrame(
        test_rows
    )

    # -------------------------------------------------------------
    # Display selected thresholds
    # -------------------------------------------------------------

    print("\nSelected thresholds from validation:")

    threshold_display = pd.DataFrame(
        [
            {
                "criterion": criterion,
                "threshold": threshold,
            }
            for criterion, threshold
            in selected_thresholds.items()
        ]
    )

    print(
        threshold_display.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Display validation performance
    # -------------------------------------------------------------

    print(
        "\nValidation performance at selected thresholds:"
    )

    print(
        validation_selected_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------
    # Display final test performance
    # -------------------------------------------------------------

    print("\nFinal test performance:")

    print(
        test_metrics_df.to_string(
            index=False
        )
    )

    return (
        validation_selected_df,
        test_metrics_df,
    )

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_validation = []
    all_test = []
    threshold_summary = []

    print("=" * 72)
    print("VALIDATION-ONLY THRESHOLD ANALYSIS")
    print("=" * 72)

    for dataset_name in DATASETS:
        validation_df, test_df = evaluate_dataset(dataset_name)

        all_validation.append(validation_df)
        all_test.append(test_df)

        # Match each validation-selected threshold with the
        # corresponding locked test result.
        for _, validation_row in validation_df.iterrows():

            criterion = validation_row["criterion"]

            matching_test_rows = test_df[
                test_df["criterion"] == criterion
            ]

            if matching_test_rows.empty:
                raise RuntimeError(
                    f"{dataset_name}: no test result found for "
                    f"criterion '{criterion}'."
                )

            test_row = matching_test_rows.iloc[0]

            threshold_summary.append({
                "dataset": validation_row["dataset"],
                "criterion": criterion,
                "threshold": validation_row["threshold"],

                # Validation performance at the selected threshold.
                "validation_accuracy": validation_row["accuracy"],
                "validation_balanced_accuracy": (
                    validation_row["balanced_accuracy"]
                ),
                "validation_precision": validation_row["precision"],
                "validation_recall": validation_row["recall"],
                "validation_f1": validation_row["f1"],
                "validation_youden_j": validation_row["youden_j"],

                # Locked test performance using the threshold selected
                # from validation data only.
                "test_accuracy": test_row["accuracy"],
                "test_balanced_accuracy": (
                    test_row["balanced_accuracy"]
                ),
                "test_precision": test_row["precision"],
                "test_recall": test_row["recall"],
                "test_f1": test_row["f1"],
                "test_youden_j": test_row["youden_j"],

                "test_true_negatives": (
                    test_row["true_negatives"]
                ),
                "test_false_positives": (
                    test_row["false_positives"]
                ),
                "test_false_negatives": (
                    test_row["false_negatives"]
                ),
                "test_true_positives": (
                    test_row["true_positives"]
                ),
            })

    # Combine results from all datasets.
    all_validation_df = pd.concat(
        all_validation,
        ignore_index=True,
    )

    all_test_df = pd.concat(
        all_test,
        ignore_index=True,
    )

    threshold_summary_df = pd.DataFrame(
        threshold_summary
    )

    # Output files.
    validation_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_validation_thresholds.csv"
    )

    test_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_threshold_test_results.csv"
    )

    summary_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_threshold_summary.csv"
    )

    metadata_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_threshold_metadata.json"
    )

    all_validation_df.to_csv(
        validation_file,
        index=False,
    )

    all_test_df.to_csv(
        test_file,
        index=False,
    )

    threshold_summary_df.to_csv(
        summary_file,
        index=False,
    )

    metadata = {
        "experiment": "hybrid_attention_seed42_threshold_analysis",
        "seed": 42,
        "datasets": DATASETS,
        "threshold_grid": {
            "start": 0.001,
            "stop": 0.999,
            "step": 0.001,
        },
        "threshold_selection_data": "validation_only",
        "test_data_used_for_threshold_selection": False,
        "test_evaluation": (
            "locked_after_validation_threshold_selection"
        ),
        "selection_criteria": [
            "max_f1",
            "max_balanced_accuracy",
            "max_youden_j",
        ],
        "reference_condition": {
            "threshold": 0.5,
            "description": "fixed_0.5_reference",
        },
        "model_checkpoint": "seed42_best_checkpoint",
        "split_protocol": "frozen_grouped_feature_split_seed42",
        "scaling": "fit_train_only",
    }

    with open(
        metadata_file,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\n" + "=" * 72)
    print("THRESHOLD ANALYSIS COMPLETED")
    print("=" * 72)

    print("\nValidation results:")
    print(
        all_validation_df.to_string(
            index=False
        )
    )

    print("\nFinal locked test results:")
    print(
        all_test_df.to_string(
            index=False
        )
    )

    print("\nThreshold summary:")
    display_columns = [
        "dataset",
        "criterion",
        "threshold",
        "test_recall",
        "test_f1",
        "test_balanced_accuracy",
    ]

    print(
        threshold_summary_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print(f"  {validation_file}")
    print(f"  {test_file}")
    print(f"  {summary_file}")
    print(f"  {metadata_file}")

if __name__ == "__main__":
    main()
