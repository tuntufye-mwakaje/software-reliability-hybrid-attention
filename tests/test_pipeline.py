from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

from src.data_loader import load_kc1
from src.models import HybridAttentionModel
from src.preprocessing import (
    FEATURE_COLUMNS,
    FEATURE_GROUPS,
    create_feature_tokens,
    fit_scale_features,
    split_data,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "hybrid_attention_best.pt"
RESULTS_DIR = PROJECT_ROOT / "results"


def test_dataset_schema_and_shape():
    """Verify the expected KC1 dataset structure."""

    df = load_kc1()

    assert df.shape == (
        2109,
        22,
    )

    dataset_features = list(df.columns[:-1])

    # The raw KC1 dataset keeps its source column order.
    # The model's FEATURE_COLUMNS are intentionally ordered
    # according to the semantic feature groups used by the
    # attention representation.
    assert set(dataset_features) == set(FEATURE_COLUMNS)

    assert df.columns[-1] == "defects"

    assert df["defects"].value_counts().to_dict() == {
        False: 1783,
        True: 326,
    }


def test_feature_groups_cover_all_features():
    """Verify that every model feature belongs to exactly one group."""

    grouped_features = [
        feature
        for features in FEATURE_GROUPS.values()
        for feature in features
    ]

    assert len(grouped_features) == len(FEATURE_COLUMNS)

    assert len(set(grouped_features)) == len(FEATURE_COLUMNS)

    assert set(grouped_features) == set(FEATURE_COLUMNS)


def test_three_way_split_sizes_and_stratification():
    """Verify deterministic stratified train/validation/test partitions."""

    df = load_kc1()

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = split_data(
        df,
        test_size=0.20,
        random_state=42,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
        stratify=y_train_full,
    )

    assert len(X_train) == 1349
    assert len(X_validation) == 338
    assert len(X_test) == 422

    assert len(y_train) == 1349
    assert len(y_validation) == 338
    assert len(y_test) == 422

    assert y_train.sum() == 209
    assert y_validation.sum() == 52
    assert y_test.sum() == 65


def test_scaler_is_fit_only_on_training_data():
    """Verify scaling produces centered training features."""

    df = load_kc1()

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = split_data(
        df,
        test_size=0.20,
        random_state=42,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
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

    training_means = X_train_scaled.mean().abs()

    assert np.all(
        training_means.to_numpy() < 1e-6
    )

    assert np.all(
        np.isfinite(
            X_train_scaled.to_numpy()
        )
    )

    assert np.all(
        np.isfinite(
            X_validation_scaled.to_numpy()
        )
    )


def test_feature_token_shape():
    """Verify the semantic feature-token representation."""

    df = load_kc1()

    (
        X_train_full,
        X_test,
        y_train_full,
        y_test,
    ) = split_data(
        df,
        test_size=0.20,
        random_state=42,
    )

    (
        X_train,
        X_validation,
        y_train,
        y_validation,
    ) = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.20,
        random_state=42,
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

    X_train_tokens = create_feature_tokens(
        X_train_scaled
    )

    assert X_train_tokens.shape == (
        1349,
        4,
        8,
    )

    assert X_train_tokens.dtype == np.float32


def test_model_forward_shapes():
    """Verify model output and attention dimensions."""

    model = HybridAttentionModel()

    dummy_input = torch.randn(
        8,
        4,
        8,
    )

    logits, attention_weights = model(
        dummy_input,
        return_attention=True,
    )

    assert logits.shape == (
        8,
    )

    assert attention_weights.shape == (
        8,
        4,
        4,
    )


def test_checkpoint_exists_and_loads():
    """Verify the trained checkpoint can be reconstructed."""

    assert MODEL_PATH.exists()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    assert "model_state_dict" in checkpoint
    assert "model_config" in checkpoint
    assert "best_validation_loss" in checkpoint
    assert "seed" in checkpoint

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

    dummy_input = torch.randn(
        2,
        4,
        8,
    )

    with torch.no_grad():
        logits = model(dummy_input)

    assert logits.shape == (
        2,
    )


def test_result_artifacts_exist():
    """Verify the main reproducibility and analysis artifacts exist."""

    expected_files = [
        RESULTS_DIR / "baseline_results.csv",
        RESULTS_DIR / "hybrid_attention_results.csv",
        RESULTS_DIR / "hybrid_attention_threshold_selected.csv",
        RESULTS_DIR / "model_comparison.csv",
        RESULTS_DIR / "threshold_analysis.csv",
        RESULTS_DIR / "training_history.csv",
        RESULTS_DIR / "attention_weights.csv",
        RESULTS_DIR / "figures" / "training_history.png",
        RESULTS_DIR / "figures" / "attention_heatmap.png",
        RESULTS_DIR / "figures" / "model_comparison.png",
    ]

    for file_path in expected_files:
        assert file_path.exists(), (
            f"Missing expected artifact: {file_path}"
        )


def test_training_history_structure():
    """Verify the training-history artifact has the expected schema."""

    history_path = RESULTS_DIR / "training_history.csv"

    history = pd.read_csv(
        history_path
    )

    expected_columns = [
        "epoch",
        "train_loss",
        "validation_loss",
        "learning_rate",
    ]

    assert list(history.columns) == expected_columns

    assert len(history) >= 1

    assert np.all(
        np.isfinite(
            history[
                [
                    "train_loss",
                    "validation_loss",
                    "learning_rate",
                ]
            ].to_numpy()
        )
    )