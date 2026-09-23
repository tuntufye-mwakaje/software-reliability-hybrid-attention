from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from research_preprocessing import prepare_research_dataset
from research_models import HybridAttentionModel, count_parameters


# ============================================================
# Ablation experiment configuration
# ============================================================

# IMPORTANT:
# Dataset partitions are already frozen using seed 42.
# This experiment NEVER regenerates the partitions.
SPLIT_SEED = 42

# Independent training seeds on the same frozen partitions.
TRAINING_SEEDS = [42, 43, 44, 45, 46]

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.5
GRADIENT_CLIP_NORM = 1.0

EMBEDDING_DIM = 32
FULL_NUM_HEADS = 4
SINGLE_NUM_HEADS = 1
DROPOUT = 0.1

DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

# ------------------------------------------------------------
# Ablation variants
# ------------------------------------------------------------
#
# A0:
# Full Hybrid Attention
#
# A1:
# No Self-Attention
#
# A2:
# No Semantic Tokenization
#
# A3:
# Single-Head Attention
#
VARIANTS = [
    "A0_full_hybrid_attention",
    "A1_no_self_attention",
    "A2_no_semantic_tokenization",
    "A3_single_head_attention",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "results"
)

MODELS_DIR = (
    PROJECT_ROOT
    / "research_analysis"
    / "models"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """
    Set all training randomness controlled by the training seed.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    """
    Use CUDA when available; otherwise CPU.
    """

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


# ============================================================
# Data representation
# ============================================================

def tokens_to_flat_features(
    tokens: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the original 21 standardized predictors from
    the four semantic tokens.

    Existing preprocessing creates:

        size               = 5 features
        complexity         = 4 features
        halstead           = 8 features
        operators_operands = 4 features

    Each group is padded to width 8.

    Therefore:

        5 + 4 + 8 + 4 = 21

    This reconstruction removes only the padding positions.

    No raw data are reloaded.

    No scaler is refitted.

    No partition is regenerated.
    """

    if tokens.ndim != 3:

        raise ValueError(
            "Expected token array with three dimensions, "
            f"got {tokens.shape}."
        )

    if tokens.shape[1:] != (4, 8):

        raise ValueError(
            "Expected token shape "
            "(samples, 4, 8), "
            f"got {tokens.shape}."
        )

    flat_features = np.concatenate(
        [
            tokens[:, 0, :5],
            tokens[:, 1, :4],
            tokens[:, 2, :8],
            tokens[:, 3, :4],
        ],
        axis=1,
    )

    if flat_features.shape[1] != 21:

        raise RuntimeError(
            "A2 reconstruction failed. "
            f"Expected 21 features, "
            f"got {flat_features.shape[1]}."
        )

    return flat_features.astype(
        np.float32
    )


def get_variant_inputs(
    prepared_partition: dict,
    variant: str,
) -> np.ndarray:
    """
    Return the input representation required by a variant.
    """

    tokens = prepared_partition["tokens"]

    if variant in {
        "A0_full_hybrid_attention",
        "A1_no_self_attention",
        "A3_single_head_attention",
    }:

        return tokens.astype(
            np.float32
        )

    if variant == "A2_no_semantic_tokenization":

        return tokens_to_flat_features(
            tokens
        )

    raise ValueError(
        f"Unknown ablation variant: {variant}"
    )


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    inputs: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """
    Create a deterministic DataLoader matching the main
    training protocol.
    """

    x = torch.tensor(
        inputs,
        dtype=torch.float32,
    )

    y = torch.tensor(
        targets.astype(np.float32),
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        x,
        y,
    )

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
    )


# ============================================================
# Ablation model definitions
# ============================================================

class NoSelfAttentionModel(
    nn.Module
):
    """
    A1: No Self-Attention.

    Retained:

        semantic feature tokens
        8 -> 32 feature-group embedding
        mean pooling
        classifier

    Removed:

        multi-head self-attention
        attention residual block
        attention feed-forward block
    """

    def __init__(
        self,
        input_dim: int = 8,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = nn.Sequential(

            nn.Linear(
                input_dim,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU(),
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim,
                32,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.embedding(x)

        x = x.mean(
            dim=1
        )

        return self.classifier(
            x
        ).squeeze(-1)


class NoSemanticTokenizationModel(
    nn.Module
):
    """
    A2: No Semantic Tokenization.

    Input:

        21 standardized software metrics

    Representation:

        21 -> 32 learned projection
        LayerNorm
        GELU

    The classifier remains matched to the 32-dimensional
    representation used by the other variants.
    """

    def __init__(
        self,
        input_dim: int = 21,
        embedding_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(

            nn.Linear(
                input_dim,
                embedding_dim,
            ),

            nn.LayerNorm(
                embedding_dim
            ),

            nn.GELU(),
        )

        self.classifier = nn.Sequential(

            nn.Linear(
                embedding_dim,
                32,
            ),

            nn.GELU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                32,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.input_projection(
            x
        )

        return self.classifier(
            x
        ).squeeze(-1)


def build_model(
    variant: str,
) -> nn.Module:
    """
    Construct exactly one ablation variant.
    """

    if variant == (
        "A0_full_hybrid_attention"
    ):

        return HybridAttentionModel(
            input_dim=8,
            embedding_dim=EMBEDDING_DIM,
            num_heads=FULL_NUM_HEADS,
            dropout=DROPOUT,
        )

    if variant == (
        "A1_no_self_attention"
    ):

        return NoSelfAttentionModel(
            input_dim=8,
            embedding_dim=EMBEDDING_DIM,
            dropout=DROPOUT,
        )

    if variant == (
        "A2_no_semantic_tokenization"
    ):

        return NoSemanticTokenizationModel(
            input_dim=21,
            embedding_dim=EMBEDDING_DIM,
            dropout=DROPOUT,
        )

    if variant == (
        "A3_single_head_attention"
    ):

        return HybridAttentionModel(
            input_dim=8,
            embedding_dim=EMBEDDING_DIM,
            num_heads=SINGLE_NUM_HEADS,
            dropout=DROPOUT,
        )

    raise ValueError(
        f"Unknown ablation variant: {variant}"
    )


# ============================================================
# Classification metrics
# ============================================================

def calculate_binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Calculate threshold-dependent binary metrics.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    targets = targets.astype(int)

    tp = int(
        np.sum(
            (targets == 1)
            & (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (targets == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (targets == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (targets == 1)
            & (predictions == 0)
        )
    )

    accuracy = (
        (tp + tn) / len(targets)
        if len(targets) > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def calculate_auc_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    """
    Calculate threshold-independent ROC-AUC and PR-AUC.
    """

    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    unique_classes = np.unique(
        targets
    )

    if len(unique_classes) < 2:

        return {
            "roc_auc": np.nan,
            "pr_auc": np.nan,
        }

    return {
        "roc_auc": float(
            roc_auc_score(
                targets,
                probabilities,
            )
        ),

        "pr_auc": float(
            average_precision_score(
                targets,
                probabilities,
            )
        ),
    }


# ============================================================
# Model evaluation
# ============================================================

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    """
    Evaluate a model without gradient computation.
    """

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_targets = []

    all_probabilities = []

    with torch.no_grad():

        for inputs, targets in loader:

            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )

            logits = model(
                inputs
            )

            loss = criterion(
                logits,
                targets,
            )

            batch_size = inputs.size(
                0
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_targets.append(
                targets.cpu().numpy()
            )

            all_probabilities.append(
                probabilities.cpu().numpy()
            )

    targets = np.concatenate(
        all_targets
    )

    probabilities = np.concatenate(
        all_probabilities
    )

    metrics = calculate_auc_metrics(
        targets,
        probabilities,
    )

    metrics.update(
        calculate_binary_metrics(
            targets,
            probabilities,
            threshold=0.5,
        )
    )

    metrics["loss"] = (
        total_loss
        / total_samples
    )

    return metrics


# ============================================================
# Model training
# ============================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
) -> tuple[nn.Module, dict]:

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=LR_REDUCTION_FACTOR,
            patience=LR_REDUCTION_PATIENCE,
        )
    )

    best_validation_loss = float(
        "inf"
    )

    best_state = None

    best_epoch = 0

    epochs_without_improvement = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        sample_count = 0

        for inputs, targets in train_loader:

            inputs = inputs.to(
                device
            )

            targets = targets.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                inputs
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRADIENT_CLIP_NORM,
            )

            optimizer.step()

            batch_size = inputs.size(
                0
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            sample_count += (
                batch_size
            )

        train_loss = (
            running_loss
            / sample_count
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_loss = (
            validation_metrics[
                "loss"
            ]
        )

        scheduler.step(
            validation_loss
        )

        current_lr = (
            optimizer.param_groups[0][
                "lr"
            ]
        )

        history.append(
            {
                "epoch": epoch,

                "train_loss": train_loss,

                "validation_loss":
                    validation_loss,

                "validation_roc_auc":
                    validation_metrics[
                        "roc_auc"
                    ],

                "validation_pr_auc":
                    validation_metrics[
                        "pr_auc"
                    ],

                "learning_rate":
                    current_lr,
            }
        )

        print(
            f"    Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_loss:.6f} | "
            f"val_roc_auc="
            f"{validation_metrics['roc_auc']:.6f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_state = {
                key:
                    value.detach()
                    .cpu()
                    .clone()

                for key, value
                in model.state_dict().items()
            }

            best_epoch = epoch

            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                f"    Early stopping "
                f"at epoch {epoch}."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No valid model checkpoint "
            "was produced."
        )

    model.load_state_dict(
        best_state
    )

    return model, {
        "history": history,

        "best_epoch": best_epoch,

        "best_validation_loss":
            best_validation_loss,
    }


# ============================================================
# Single ablation run
# ============================================================

def run_experiment(
    dataset_name: str,
    variant: str,
    training_seed: int,
    device: torch.device,
) -> dict:

    print()
    print("=" * 80)

    print(
        f"Dataset: {dataset_name} | "
        f"Variant: {variant} | "
        f"Training seed: {training_seed}"
    )

    print("=" * 80)

    set_seed(
        training_seed
    )

    prepared = (
        prepare_research_dataset(
            dataset_name
        )
    )

    train = prepared["train"]

    validation = prepared[
        "validation"
    ]

    test = prepared["test"]

    # --------------------------------------------------------
    # Input representation
    # --------------------------------------------------------

    train_inputs = get_variant_inputs(
        train,
        variant,
    )

    validation_inputs = (
        get_variant_inputs(
            validation,
            variant,
        )
    )

    test_inputs = get_variant_inputs(
        test,
        variant,
    )

    # --------------------------------------------------------
    # Structural input validation
    # --------------------------------------------------------

    if variant == (
        "A2_no_semantic_tokenization"
    ):

        expected_dimension = 21

    else:

        expected_dimension = 8

    if train_inputs.shape[-1] != (
        expected_dimension
    ):

        raise RuntimeError(
            f"{variant}: expected input "
            f"dimension {expected_dimension}, "
            f"got {train_inputs.shape[-1]}."
        )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = create_dataloader(
        train_inputs,
        train["target"],
        BATCH_SIZE,
        shuffle=True,
        seed=training_seed,
    )

    validation_loader = create_dataloader(
        validation_inputs,
        validation["target"],
        BATCH_SIZE,
        shuffle=False,
        seed=training_seed,
    )

    test_loader = create_dataloader(
        test_inputs,
        test["target"],
        BATCH_SIZE,
        shuffle=False,
        seed=training_seed,
    )

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------

    model = build_model(
        variant
    ).to(device)

    parameter_count = count_parameters(
        model
    )

    print(
        f"Training seed: {training_seed}"
    )

    print(
        f"Frozen split seed: {SPLIT_SEED}"
    )

    print(
        f"Variant: {variant}"
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print(
        f"Train samples: "
        f"{len(train['target'])}"
    )

    print(
        f"Validation samples: "
        f"{len(validation['target'])}"
    )

    print(
        f"Test samples: "
        f"{len(test['target'])}"
    )

    print(
        f"Input shape: "
        f"{train_inputs.shape[1:]}"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model, training_info = train_model(
        model,
        train_loader,
        validation_loader,
        device,
    )

    # --------------------------------------------------------
    # Final test evaluation
    # --------------------------------------------------------

    criterion = nn.BCEWithLogitsLoss()

    test_metrics = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
    )

    print()
    print(
        "Final test results"
    )
    print(
        "-" * 40
    )

    print(
        f"ROC-AUC:   "
        f"{test_metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:    "
        f"{test_metrics['pr_auc']:.6f}"
    )

    print(
        f"Accuracy:  "
        f"{test_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.6f}"
    )

    print(
        f"Recall:    "
        f"{test_metrics['recall']:.6f}"
    )

    print(
        f"F1:        "
        f"{test_metrics['f1']:.6f}"
    )

    print()
    print(
        "Confusion matrix"
    )
    print(
        "-" * 40
    )

    print(
        f"TN={test_metrics['true_negatives']}  "
        f"FP={test_metrics['false_positives']}"
    )

    print(
        f"FN={test_metrics['false_negatives']}  "
        f"TP={test_metrics['true_positives']}"
    )

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    safe_variant = variant.lower()

    checkpoint_path = (
        MODELS_DIR
        / (
            f"ablation_{safe_variant}_"
            f"{dataset_name.lower()}_"
            f"seed{training_seed}_best.pt"
        )
    )

    torch.save(
        {
            "dataset": dataset_name,

            "variant": variant,

            "training_seed":
                training_seed,

            "split_seed":
                SPLIT_SEED,

            "model_state_dict":
                model.state_dict(),

            "best_epoch":
                training_info[
                    "best_epoch"
                ],

            "best_validation_loss":
                training_info[
                    "best_validation_loss"
                ],

            "train_samples":
                len(train["target"]),

            "validation_samples":
                len(
                    validation["target"]
                ),

            "test_samples":
                len(test["target"]),

            "parameter_count":
                parameter_count,
        },
        checkpoint_path,
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR
        / (
            f"ablation_{safe_variant}_"
            f"{dataset_name.lower()}_"
            f"seed{training_seed}_"
            "training_history.csv"
        )
    )

    pd.DataFrame(
        training_info["history"]
    ).to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    if variant == (
        "A0_full_hybrid_attention"
    ):

        num_heads = (
            FULL_NUM_HEADS
        )

    elif variant == (
        "A3_single_head_attention"
    ):

        num_heads = (
            SINGLE_NUM_HEADS
        )

    else:

        num_heads = 0

    return {

        "variant":
            variant,

        "dataset":
            dataset_name,

        "training_seed":
            training_seed,

        "split_seed":
            SPLIT_SEED,

        "best_epoch":
            training_info[
                "best_epoch"
            ],

        "best_validation_loss":
            training_info[
                "best_validation_loss"
            ],

        "test_loss":
            test_metrics[
                "loss"
            ],

        "roc_auc":
            test_metrics[
                "roc_auc"
            ],

        "pr_auc":
            test_metrics[
                "pr_auc"
            ],

        "accuracy":
            test_metrics[
                "accuracy"
            ],

        "precision":
            test_metrics[
                "precision"
            ],

        "recall":
            test_metrics[
                "recall"
            ],

        "f1":
            test_metrics[
                "f1"
            ],

        "true_negatives":
            test_metrics[
                "true_negatives"
            ],

        "false_positives":
            test_metrics[
                "false_positives"
            ],

        "false_negatives":
            test_metrics[
                "false_negatives"
            ],

        "true_positives":
            test_metrics[
                "true_positives"
            ],

        "train_samples":
            len(train["target"]),

        "validation_samples":
            len(
                validation["target"]
            ),

        "test_samples":
            len(test["target"]),

        "parameter_count":
            parameter_count,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "embedding_dim":
            EMBEDDING_DIM,

        "num_heads":
            num_heads,

        "dropout":
            DROPOUT,

        "threshold":
            0.5,

        "input_dimension":
            int(
                train_inputs.shape[-1]
            ),

        "checkpoint":
            str(checkpoint_path),
    }


# ============================================================
# Structural validation
# ============================================================

def validate_results(
    results_df: pd.DataFrame,
) -> None:
    """
    Validate the completed 100-run ablation experiment.
    """

    expected_runs = (
        len(VARIANTS)
        * len(DATASETS)
        * len(TRAINING_SEEDS)
    )

    if len(results_df) != (
        expected_runs
    ):

        raise RuntimeError(
            f"Expected {expected_runs} "
            f"runs, got {len(results_df)}."
        )

    if set(
        results_df["variant"]
    ) != set(VARIANTS):

        raise RuntimeError(
            "Ablation variant set does "
            "not match the configured design."
        )

    if set(
        results_df["dataset"]
    ) != set(DATASETS):

        raise RuntimeError(
            "Dataset set does not match "
            "the configured design."
        )

    if set(
        results_df["training_seed"]
    ) != set(TRAINING_SEEDS):

        raise RuntimeError(
            "Training seed set does not "
            "match the configured design."
        )

    if not (
        results_df["split_seed"]
        == SPLIT_SEED
    ).all():

        raise RuntimeError(
            "Ablation results contain "
            "a split seed other than 42."
        )

    duplicate_columns = [
        "variant",
        "dataset",
        "training_seed",
    ]

    if results_df.duplicated(
        subset=duplicate_columns
    ).any():

        raise RuntimeError(
            "Duplicate ablation runs detected."
        )

    required_metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]

    for column in required_metrics:

        if results_df[
            column
        ].isna().any():

            raise RuntimeError(
                f"Missing values detected "
                f"in metric column: {column}"
            )

    expected_partition_sizes = {

        "CM1": (
            309,
            84,
            105,
        ),

        "JM1": (
            8491,
            2190,
            2523,
        ),

        "KC1": (
            1370,
            307,
            432,
        ),

        "KC2": (
            346,
            66,
            110,
        ),

        "PC1": (
            722,
            171,
            216,
        ),
    }

    for (
        dataset_name,
        expected,
    ) in expected_partition_sizes.items():

        dataset_rows = results_df[
            results_df["dataset"]
            == dataset_name
        ]

        observed = (

            int(
                dataset_rows[
                    "train_samples"
                ].iloc[0]
            ),

            int(
                dataset_rows[
                    "validation_samples"
                ].iloc[0]
            ),

            int(
                dataset_rows[
                    "test_samples"
                ].iloc[0]
            ),
        )

        if observed != expected:

            raise RuntimeError(
                f"Frozen partition size "
                f"mismatch for {dataset_name}: "
                f"expected {expected}, "
                f"got {observed}."
            )

        for (
            column,
            expected_value,
        ) in zip(

            [
                "train_samples",
                "validation_samples",
                "test_samples",
            ],

            expected,
        ):

            if not (
                dataset_rows[column]
                == expected_value
            ).all():

                raise RuntimeError(
                    "Partition inconsistency "
                    f"across variants/seeds "
                    f"for {dataset_name}: "
                    f"{column}"
                )

    print()
    print("=" * 80)
    print(
        "ABLATION STRUCTURAL "
        "VALIDATION PASSED"
    )
    print("=" * 80)

    print(
        f"Total runs: "
        f"{len(results_df)}"
    )

    print()
    print(
        "Runs by variant:"
    )

    print(
        results_df[
            "variant"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Runs by dataset:"
    )

    print(
        results_df[
            "dataset"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()
    print(
        "Training seeds:"
    )

    print(
        sorted(
            results_df[
                "training_seed"
            ]
            .unique()
            .tolist()
        )
    )

    print()
    print(
        "Frozen partition sizes:"
    )

    for dataset_name in DATASETS:

        row = results_df[
            results_df["dataset"]
            == dataset_name
        ].iloc[0]

        print(
            f"{dataset_name}: "
            f"train="
            f"{int(row['train_samples'])}, "
            f"validation="
            f"{int(row['validation_samples'])}, "
            f"test="
            f"{int(row['test_samples'])}"
        )


# ============================================================
# Main experiment
# ============================================================

def main() -> None:

    print()
    print("=" * 80)

    print(
        "FOUR-VARIANT ABLATION EXPERIMENT"
    )

    print("=" * 80)

    print(
        f"Frozen split seed: "
        f"{SPLIT_SEED}"
    )

    print(
        "Training seeds: "
        + ", ".join(
            str(seed)
            for seed in TRAINING_SEEDS
        )
    )

    print(
        f"Datasets: "
        f"{', '.join(DATASETS)}"
    )

    print(
        "Variants: "
        f"{', '.join(VARIANTS)}"
    )

    total_runs = (
        len(VARIANTS)
        * len(DATASETS)
        * len(TRAINING_SEEDS)
    )

    print(
        f"Total training runs: "
        f"{total_runs}"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    results = []

    # --------------------------------------------------------
    # Variant -> seed -> dataset
    # --------------------------------------------------------

    for variant in VARIANTS:

        print()
        print(
            "#" * 80
        )

        print(
            f"STARTING ABLATION VARIANT: "
            f"{variant}"
        )

        print(
            "#" * 80
        )

        for training_seed in TRAINING_SEEDS:

            print()
            print(
                f"STARTING TRAINING SEED "
                f"{training_seed}"
            )

            for dataset_name in DATASETS:

                result = run_experiment(
                    dataset_name,
                    variant,
                    training_seed,
                    device,
                )

                results.append(
                    result
                )

    results_df = pd.DataFrame(
        results
    )

    results_df = (
        results_df
        .sort_values(
            by=[
                "variant",
                "dataset",
                "training_seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Structural validation
    # --------------------------------------------------------

    validate_results(
        results_df
    )

    # --------------------------------------------------------
    # Save complete results
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "ablation_repeated_seed_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "experiment":
            "four_variant_repeated_seed_ablation",

        "variants":
            VARIANTS,

        "variant_definitions": {

            "A0_full_hybrid_attention":
                (
                    "Full model: semantic tokens "
                    "+ 32-dimensional feature-group "
                    "embedding + 4-head self-attention "
                    "+ residual/LayerNorm "
                    "+ feed-forward block "
                    "+ mean pooling "
                    "+ classifier."
                ),

            "A1_no_self_attention":
                (
                    "Semantic tokens "
                    "+ 32-dimensional feature-group "
                    "embedding "
                    "+ mean pooling "
                    "+ classifier; "
                    "self-attention block removed."
                ),

            "A2_no_semantic_tokenization":
                (
                    "Original 21 standardized "
                    "predictors treated as one "
                    "vector with a dedicated "
                    "21-to-32 projection, "
                    "LayerNorm, GELU, and matched "
                    "classifier; semantic tokenization "
                    "and attention removed."
                ),

            "A3_single_head_attention":
                (
                    "Same full architecture as A0, "
                    "but self-attention uses one "
                    "head instead of four."
                ),
        },

        "split_seed":
            SPLIT_SEED,

        "training_seeds":
            TRAINING_SEEDS,

        "datasets":
            DATASETS,

        "total_runs":
            len(results_df),

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "lr_reduction_patience":
            LR_REDUCTION_PATIENCE,

        "lr_reduction_factor":
            LR_REDUCTION_FACTOR,

        "gradient_clip_norm":
            GRADIENT_CLIP_NORM,

        "embedding_dim":
            EMBEDDING_DIM,

        "full_num_heads":
            FULL_NUM_HEADS,

        "single_num_heads":
            SINGLE_NUM_HEADS,

        "dropout":
            DROPOUT,

        "device":
            str(device),

        "threshold":
            0.5,

        "split_protocol":
            "frozen_grouped_feature_split_seed42",

        "scaling_protocol":
            "fit_train_only",

        "training_randomness":
            "controlled_by_training_seed",

        "model_initialization":
            "seed_controlled",

        "training_dataloader":
            "shuffle_true_seed_controlled",

        "validation_dataloader":
            "shuffle_false",

        "test_dataloader":
            "shuffle_false",

        "model_selection":
            "minimum_validation_loss",

        "test_usage":
            "final_evaluation_only",

        "a2_input_reconstruction":
            (
                "Original 21 standardized "
                "predictors reconstructed from "
                "non-padded positions of the four "
                "preprocessing tokens; no new "
                "scaling or partitioning."
            ),

        "parameter_counts":
            (
                results_df[
                    [
                        "variant",
                        "parameter_count",
                    ]
                ]
                .drop_duplicates()
                .sort_values(
                    "variant"
                )
                .to_dict(
                    orient="records"
                )
            ),
    }

    metadata_path = (
        RESULTS_DIR
        / "ablation_repeated_seed_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 80)

    print(
        "ABLATION EXPERIMENT COMPLETE"
    )

    print("=" * 80)

    print(
        f"Completed runs: "
        f"{len(results_df)}"
    )

    print()

    print(
        results_df[
            [
                "variant",
                "dataset",
                "training_seed",
                "best_epoch",
                "roc_auc",
                "pr_auc",
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        results_path
    )

    print()
    print(
        "Metadata saved to:"
    )

    print(
        metadata_path
    )


if __name__ == "__main__":
    main()