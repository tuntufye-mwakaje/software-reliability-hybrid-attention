from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

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

ATTENTION_RESULTS_PATH = (
    RESULTS_DIR
    / "attention_weights.csv"
)

ATTENTION_FIGURE_PATH = (
    RESULTS_DIR
    / "figures"
    / "attention_heatmap.png"
)

SEED = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

FEATURE_GROUP_NAMES = [
    "Size",
    "Complexity",
    "Halstead",
    "Operators / Operands",
]


def prepare_test_data(df):
    """
    Reproduce the exact train/validation/test preprocessing
    used by the training pipeline.

    The scaler is fitted only on the training partition.
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

    X_test_tokens = create_feature_tokens(
        X_test_scaled
    )

    return (
        X_test_tokens,
        y_test.to_numpy(dtype=np.float32),
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


def extract_attention(model, X_test_tokens):
    """
    Extract the model's aggregate attention weights.

    Current model configuration returns:
        (samples, 4, 4)

    because PyTorch's MultiheadAttention averages
    attention weights across heads by default.
    """

    X_tensor = torch.tensor(
        X_test_tokens,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits, attention_weights = model(
            X_tensor,
            return_attention=True,
        )

        probabilities = (
            torch.sigmoid(logits)
            .numpy()
        )

    return (
        probabilities,
        attention_weights.numpy(),
    )


def calculate_average_attention(
    attention_weights,
):
    """
    Calculate the mean attention matrix across
    all final-test samples.
    """

    return attention_weights.mean(
        axis=0
    )


def save_attention_matrix(
    attention_matrix,
):
    """
    Save the average attention matrix in
    machine-readable long format.
    """

    rows = []

    for source_index, source_name in enumerate(
        FEATURE_GROUP_NAMES
    ):
        for target_index, target_name in enumerate(
            FEATURE_GROUP_NAMES
        ):
            rows.append(
                {
                    "source_group": source_name,
                    "target_group": target_name,
                    "attention_weight": float(
                        attention_matrix[
                            source_index,
                            target_index,
                        ]
                    ),
                }
            )

    attention_df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    attention_df.to_csv(
        ATTENTION_RESULTS_PATH,
        index=False,
    )

    return attention_df


def create_attention_heatmap(
    attention_matrix,
):
    """
    Create a heatmap of average attention
    between semantic feature groups.
    """

    FIGURE_DIR = (
        RESULTS_DIR
        / "figures"
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(9, 7)
    )

    image = ax.imshow(
        attention_matrix,
        aspect="auto",
    )

    ax.set_xticks(
        range(len(FEATURE_GROUP_NAMES))
    )

    ax.set_yticks(
        range(len(FEATURE_GROUP_NAMES))
    )

    ax.set_xticklabels(
        FEATURE_GROUP_NAMES,
        rotation=30,
        ha="right",
    )

    ax.set_yticklabels(
        FEATURE_GROUP_NAMES
    )

    ax.set_xlabel(
        "Attended-to Feature Group"
    )

    ax.set_ylabel(
        "Query Feature Group"
    )

    ax.set_title(
        "Average Self-Attention Across KC1 Feature Groups"
    )

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )

    colorbar.set_label(
        "Average Attention Weight"
    )

    for row in range(
        attention_matrix.shape[0]
    ):
        for column in range(
            attention_matrix.shape[1]
        ):
            ax.text(
                column,
                row,
                f"{attention_matrix[row, column]:.3f}",
                ha="center",
                va="center",
            )

    fig.tight_layout()

    fig.savefig(
        ATTENTION_FIGURE_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading KC1 dataset...")

    df = load_kc1()

    print(
        "\nPreparing final test data..."
    )

    X_test_tokens, y_test = (
        prepare_test_data(df)
    )

    print(
        "Test token shape:",
        X_test_tokens.shape,
    )

    print(
        "Test target shape:",
        y_test.shape,
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
        "\nExtracting attention weights..."
    )

    (
        probabilities,
        attention_weights,
    ) = extract_attention(
        model,
        X_test_tokens,
    )

    print(
        "Probability shape:",
        probabilities.shape,
    )

    print(
        "Attention tensor shape:",
        attention_weights.shape,
    )

    average_attention = (
        calculate_average_attention(
            attention_weights
        )
    )

    print(
        "\nAverage attention matrix:"
    )

    print(
        pd.DataFrame(
            average_attention,
            index=FEATURE_GROUP_NAMES,
            columns=FEATURE_GROUP_NAMES,
        ).to_string()
    )

    attention_df = save_attention_matrix(
        average_attention
    )

    create_attention_heatmap(
        average_attention
    )

    print(
        "\nAttention results saved to:"
    )

    print(
        ATTENTION_RESULTS_PATH
    )

    print(
        "\nAttention heatmap saved to:"
    )

    print(
        ATTENTION_FIGURE_PATH
    )


if __name__ == "__main__":
    main()