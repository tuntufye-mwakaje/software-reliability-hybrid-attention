from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

from data_loader import load_kc1
from preprocessing import (
    split_data,
    fit_scale_features,
)


SEED = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

BASELINE_RESULTS_PATH = (
    RESULTS_DIR
    / "baseline_results.csv"
)


def prepare_baseline_data(df):
    """
    Create the same train/validation/test partitions used
    by the hybrid attention model.

    Scaling is fitted only on the training partition.
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
    ) = __import__(
        "sklearn.model_selection",
        fromlist=["train_test_split"],
    ).train_test_split(
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

    return (
        X_train_scaled,
        X_validation_scaled,
        X_test_scaled,
        y_train.to_numpy(dtype=np.int64),
        y_validation.to_numpy(dtype=np.int64),
        y_test.to_numpy(dtype=np.int64),
    )


def evaluate_classifier(
    model_name,
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    """Train and evaluate a baseline classifier."""

    model.fit(
        X_train,
        y_train,
    )

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "model": model_name,
        "threshold": 0.50,
        "roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities,
        ),
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
    }

    matrix = confusion_matrix(
        y_test,
        predictions,
    )

    return metrics, matrix


if __name__ == "__main__":

    print("Loading KC1 dataset...")
    df = load_kc1()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = prepare_baseline_data(df)

    print("\nDataset partitions")
    print("=" * 70)

    print(
        "Training:",
        X_train.shape,
    )

    print(
        "Validation:",
        X_validation.shape,
    )

    print(
        "Final test:",
        X_test.shape,
    )

    # ------------------------------------------------------------
    # Logistic Regression
    # ------------------------------------------------------------

    logistic_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=SEED,
    )

    logistic_metrics, logistic_matrix = (
        evaluate_classifier(
            "Logistic Regression",
            logistic_model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    # ------------------------------------------------------------
    # Random Forest
    # ------------------------------------------------------------

    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )

    random_forest_metrics, random_forest_matrix = (
        evaluate_classifier(
            "Random Forest",
            random_forest_model,
            X_train,
            y_train,
            X_test,
            y_test,
        )
    )

    results = [
        logistic_metrics,
        random_forest_metrics,
    ]

    results_df = pd.DataFrame(
        results
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        BASELINE_RESULTS_PATH,
        index=False,
    )

    print("\nBaseline results")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "roc_auc": "{:.6f}".format,
                "pr_auc": "{:.6f}".format,
                "accuracy": "{:.6f}".format,
                "precision": "{:.6f}".format,
                "recall": "{:.6f}".format,
                "f1": "{:.6f}".format,
            },
        )
    )

    print("\nLogistic Regression confusion matrix")
    print("=" * 70)
    print(logistic_matrix)

    print("\nRandom Forest confusion matrix")
    print("=" * 70)
    print(random_forest_matrix)

    print(
        "\nBaseline results saved to:"
    )

    print(BASELINE_RESULTS_PATH)