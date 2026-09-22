from pathlib import Path
import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from data_loader import load_kc1
from models import HybridAttentionModel
from preprocessing import (
    create_feature_tokens,
    fit_scale_features,
    split_data,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "hybrid_attention_best.pt"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)

TRAINING_HISTORY_PATH = (
    RESULTS_DIR
    / "training_history.csv"
)

TRAINING_HISTORY_FIGURE_PATH = (
    RESULTS_DIR
    / "figures"
    / "training_history.png"
)


# ============================================================
# Reproducibility
# ============================================================

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Experiment configuration
# ============================================================

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

BATCH_SIZE = 32
EPOCHS = 50

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 10

GRADIENT_CLIP_MAX_NORM = 1.0


# ============================================================
# Data preparation
# ============================================================

def prepare_three_way_split(df):
    """
    Create the exact train/validation/test split used by
    the evaluation and threshold-analysis pipelines.

    Final test:
        20% of the complete dataset.

    Training/validation:
        Remaining 80%, split into 80% training and
        20% validation.

    All splits are stratified by the defect target.
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

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


def prepare_model_data(df):
    """
    Prepare model inputs while preventing preprocessing leakage.

    The scaler is fitted only on the training partition.
    Validation and test data are transformed using that
    training-fitted scaler.
    """

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = prepare_three_way_split(df)

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
    )


# ============================================================
# Training and validation
# ============================================================

def train_one_epoch(
    model,
    data_loader,
    criterion,
    optimizer,
):
    """
    Train the model for one epoch.
    """

    model.train()

    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in data_loader:

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = criterion(
            logits,
            y_batch,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=GRADIENT_CLIP_MAX_NORM,
        )

        optimizer.step()

        batch_size = X_batch.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(
    model,
    data_loader,
    criterion,
):
    """
    Calculate average loss on a validation dataset.
    """

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():

        for X_batch, y_batch in data_loader:

            logits = model(X_batch)

            loss = criterion(
                logits,
                y_batch,
            )

            batch_size = X_batch.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


# ============================================================
# Training-history visualization
# ============================================================

def save_training_history(
    training_history,
):
    """
    Save training/validation loss history to CSV and
    generate a publication/repository-ready training curve.
    """

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR = (
        RESULTS_DIR
        / "figures"
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_df = pd.DataFrame(
        training_history
    )

    history_df.to_csv(
        TRAINING_HISTORY_PATH,
        index=False,
    )

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        history_df["epoch"],
        history_df["train_loss"],
        label="Training Loss",
    )

    plt.plot(
        history_df["epoch"],
        history_df["validation_loss"],
        label="Validation Loss",
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Hybrid Attention Training History"
    )

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        TRAINING_HISTORY_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return history_df


# ============================================================
# Main training pipeline
# ============================================================

def main():

    print(
        "Loading KC1 dataset..."
    )

    df = load_kc1()

    print(
        "\nPreparing train/validation/test data..."
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

    print(
        "\nTraining target distributions:"
    )

    print(
        "Training:",
        {
            int(label): int(
                np.sum(y_train == label)
            )
            for label in [0, 1]
        },
    )

    print(
        "Validation:",
        {
            int(label): int(
                np.sum(y_validation == label)
            )
            for label in [0, 1]
        },
    )

    print(
        "Final test:",
        {
            int(label): int(
                np.sum(y_test == label)
            )
            for label in [0, 1]
        },
    )

    # --------------------------------------------------------
    # Convert arrays to PyTorch tensors
    # --------------------------------------------------------

    X_train_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
    )

    y_train_tensor = torch.tensor(
        y_train,
        dtype=torch.float32,
    )

    X_validation_tensor = torch.tensor(
        X_validation,
        dtype=torch.float32,
    )

    y_validation_tensor = torch.tensor(
        y_validation,
        dtype=torch.float32,
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_dataset = TensorDataset(
        X_train_tensor,
        y_train_tensor,
    )

    validation_dataset = TensorDataset(
        X_validation_tensor,
        y_validation_tensor,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = HybridAttentionModel(
        input_dim=8,
        embedding_dim=32,
        num_heads=4,
        dropout=0.1,
    )

    print(
        "\nModel parameters:",
        sum(
            parameter.numel()
            for parameter in model.parameters()
        ),
    )

    # --------------------------------------------------------
    # Loss, optimizer and scheduler
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )

    # --------------------------------------------------------
    # Training state
    # --------------------------------------------------------

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None

    patience_counter = 0

    training_history = []

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        validation_loss = evaluate_loss(
            model,
            validation_loader,
            criterion,
        )

        scheduler.step(
            validation_loss
        )

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        training_history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "learning_rate": current_learning_rate,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Validation Loss: {validation_loss:.6f} | "
            f"LR: {current_learning_rate:.6f}"
        )

        # ----------------------------------------------------
        # Best-model tracking
        # ----------------------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if (
            patience_counter
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered "
                f"at epoch {epoch}."
            )

            break

    # --------------------------------------------------------
    # Restore best model
    # --------------------------------------------------------

    if best_model_state is None:
        raise RuntimeError(
            "No best model state was captured."
        )

    model.load_state_dict(
        best_model_state
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": 8,
            "embedding_dim": 32,
            "num_heads": 4,
            "dropout": 0.1,
        },
        "best_validation_loss": best_validation_loss,
        "seed": SEED,
    }

    torch.save(
        checkpoint,
        MODEL_PATH,
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_df = save_training_history(
        training_history
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\nBest validation loss:",
        best_validation_loss,
    )

    print(
        "\nTraining history:"
    )

    print(
        history_df.to_string(
            index=False
        )
    )

    print(
        "\nModel checkpoint saved to:"
    )

    print(
        MODEL_PATH
    )

    print(
        "\nTraining history saved to:"
    )

    print(
        TRAINING_HISTORY_PATH
    )

    print(
        "\nTraining history figure saved to:"
    )

    print(
        TRAINING_HISTORY_FIGURE_PATH
    )


if __name__ == "__main__":
    main()