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
# Research experiment configuration
# ============================================================

# IMPORTANT:
# The dataset partitions are already frozen using seed 42.
# This value refers to the frozen partition protocol, NOT
# the randomness used during model training.
SPLIT_SEED = 42

# Training seeds used for repeated-seed experiments.
#
# The same frozen train/validation/test partitions are used
# for every training seed.
TRAINING_SEEDS = [
    42,
    43,
    44,
    45,
    46,
]

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 10
LR_REDUCTION_PATIENCE = 5
LR_REDUCTION_FACTOR = 0.5
GRADIENT_CLIP_NORM = 1.0

EMBEDDING_DIM = 32
NUM_HEADS = 4
DROPOUT = 0.1

DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_DIR = PROJECT_ROOT / "research_analysis" / "results"
MODELS_DIR = PROJECT_ROOT / "research_analysis" / "models"

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
    Set random seeds for reproducible model training.
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
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# DataLoader
# ============================================================

def create_dataloader(
    tokens: np.ndarray,
    targets: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    """
    Convert research preprocessing outputs into a PyTorch
    DataLoader.

    The training DataLoader uses a seed-controlled generator.
    Validation and test DataLoaders do not shuffle data.
    """

    x = torch.tensor(
        tokens,
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

    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
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
    Calculate threshold-dependent binary classification metrics.
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
        2 * precision * recall / (precision + recall)
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

    unique_classes = np.unique(targets)

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

            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            loss = criterion(
                logits,
                targets,
            )

            batch_size = inputs.size(0)

            total_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

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
        total_loss / total_samples
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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_REDUCTION_FACTOR,
        patience=LR_REDUCTION_PATIENCE,
    )

    best_validation_loss = float("inf")

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

            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits = model(inputs)

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

            batch_size = inputs.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            sample_count += batch_size

        train_loss = (
            running_loss / sample_count
        )

        validation_metrics = evaluate_model(
            model,
            validation_loader,
            criterion,
            device,
        )

        validation_loss = validation_metrics[
            "loss"
        ]

        scheduler.step(
            validation_loss
        )

        current_lr = optimizer.param_groups[0][
            "lr"
        ]

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "validation_roc_auc": validation_metrics[
                    "roc_auc"
                ],
                "validation_pr_auc": validation_metrics[
                    "pr_auc"
                ],
                "learning_rate": current_lr,
            }
        )

        print(
            f"    Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={validation_loss:.6f} | "
            f"val_roc_auc={validation_metrics['roc_auc']:.6f}"
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
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
                f"    Early stopping at epoch {epoch}."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    return model, {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
    }


# ============================================================
# Single dataset experiment
# ============================================================

def run_dataset(
    dataset_name: str,
    training_seed: int,
    device: torch.device,
) -> dict:

    print()
    print("=" * 70)
    print(
        f"Dataset: {dataset_name} | "
        f"Training seed: {training_seed}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Set randomness for this training run.
    # --------------------------------------------------------

    set_seed(training_seed)

    # --------------------------------------------------------
    # Load the already-frozen research partitions.
    #
    # IMPORTANT:
    # prepare_research_dataset() uses the frozen seed-42
    # partition protocol. The training seed does NOT regenerate
    # the dataset split.
    # --------------------------------------------------------

    prepared = prepare_research_dataset(
        dataset_name
    )

    train = prepared["train"]

    validation = prepared["validation"]

    test = prepared["test"]

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    train_loader = create_dataloader(
        train["tokens"],
        train["target"],
        BATCH_SIZE,
        shuffle=True,
        seed=training_seed,
    )

    validation_loader = create_dataloader(
        validation["tokens"],
        validation["target"],
        BATCH_SIZE,
        shuffle=False,
        seed=training_seed,
    )

    test_loader = create_dataloader(
        test["tokens"],
        test["target"],
        BATCH_SIZE,
        shuffle=False,
        seed=training_seed,
    )

    # --------------------------------------------------------
    # Model initialization
    # --------------------------------------------------------

    model = HybridAttentionModel(
        input_dim=8,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT,
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
        f"Trainable parameters: {parameter_count:,}"
    )

    print(
        f"Train samples: {len(train['target'])}"
    )

    print(
        f"Validation samples: {len(validation['target'])}"
    )

    print(
        f"Test samples: {len(test['target'])}"
    )

    # --------------------------------------------------------
    # Train model
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
    print("Final test results")
    print("-" * 40)

    print(
        f"ROC-AUC:   {test_metrics['roc_auc']:.6f}"
    )

    print(
        f"PR-AUC:    {test_metrics['pr_auc']:.6f}"
    )

    print(
        f"Accuracy:  {test_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: {test_metrics['precision']:.6f}"
    )

    print(
        f"Recall:    {test_metrics['recall']:.6f}"
    )

    print(
        f"F1:        {test_metrics['f1']:.6f}"
    )

    print()
    print("Confusion matrix")
    print("-" * 40)

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

    checkpoint_path = (
        MODELS_DIR
        / (
            f"{dataset_name.lower()}"
            f"_seed{training_seed}_best.pt"
        )
    )

    torch.save(
        {
            "dataset": dataset_name,

            "training_seed": training_seed,

            "split_seed": SPLIT_SEED,

            "model_state_dict": model.state_dict(),

            "model_config": {
                "input_dim": 8,
                "embedding_dim": EMBEDDING_DIM,
                "num_heads": NUM_HEADS,
                "dropout": DROPOUT,
            },

            "best_epoch": training_info[
                "best_epoch"
            ],

            "best_validation_loss": training_info[
                "best_validation_loss"
            ],

            "train_samples": len(
                train["target"]
            ),

            "validation_samples": len(
                validation["target"]
            ),

            "test_samples": len(
                test["target"]
            ),
        },
        checkpoint_path,
    )

    # --------------------------------------------------------
    # Save training history
    # --------------------------------------------------------

    history_path = (
        RESULTS_DIR
        / (
            f"{dataset_name.lower()}"
            f"_seed{training_seed}_training_history.csv"
        )
    )

    pd.DataFrame(
        training_info["history"]
    ).to_csv(
        history_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final result record
    # --------------------------------------------------------

    result = {
        "dataset": dataset_name,

        "training_seed": training_seed,

        "split_seed": SPLIT_SEED,

        "best_epoch": training_info[
            "best_epoch"
        ],

        "best_validation_loss": training_info[
            "best_validation_loss"
        ],

        "test_loss": test_metrics[
            "loss"
        ],

        "roc_auc": test_metrics[
            "roc_auc"
        ],

        "pr_auc": test_metrics[
            "pr_auc"
        ],

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

        "train_samples": len(
            train["target"]
        ),

        "validation_samples": len(
            validation["target"]
        ),

        "test_samples": len(
            test["target"]
        ),

        "parameter_count": parameter_count,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "embedding_dim": EMBEDDING_DIM,

        "num_heads": NUM_HEADS,

        "dropout": DROPOUT,

        "threshold": 0.5,

        "checkpoint": str(
            checkpoint_path
        ),
    }

    return result


# ============================================================
# Frozen partition consistency check
# ============================================================

def check_partition_consistency(
    results: list[dict],
) -> None:
    """
    Verify that repeated-seed experiments preserve the same
    frozen train/validation/test partition sizes.

    This is a structural verification step. The actual
    partitioning remains controlled by the existing
    preprocessing pipeline.
    """

    results_df = pd.DataFrame(
        results
    )

    if results_df.empty:
        raise RuntimeError(
            "No experiment results were produced."
        )

    expected_columns = [
        "dataset",
        "train_samples",
        "validation_samples",
        "test_samples",
    ]

    missing_columns = [
        column
        for column in expected_columns
        if column not in results_df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            "Missing partition columns: "
            + ", ".join(missing_columns)
        )

    for dataset_name in DATASETS:

        dataset_results = results_df[
            results_df["dataset"] == dataset_name
        ]

        if dataset_results.empty:
            raise RuntimeError(
                f"No results found for dataset {dataset_name}."
            )

        for column in [
            "train_samples",
            "validation_samples",
            "test_samples",
        ]:

            unique_values = (
                dataset_results[column]
                .dropna()
                .unique()
            )

            if len(unique_values) != 1:

                raise RuntimeError(
                    "Frozen partition consistency check failed "
                    f"for {dataset_name}, column {column}: "
                    f"{unique_values}"
                )

    print()
    print("=" * 70)
    print("FROZEN PARTITION CONSISTENCY CHECK")
    print("=" * 70)

    for dataset_name in DATASETS:

        dataset_results = results_df[
            results_df["dataset"] == dataset_name
        ].iloc[0]

        print(
            f"{dataset_name}: "
            f"train={int(dataset_results['train_samples'])}, "
            f"validation={int(dataset_results['validation_samples'])}, "
            f"test={int(dataset_results['test_samples'])}"
        )

    print()
    print(
        "Partition sizes are consistent across all "
        "training seeds."
    )


# ============================================================
# Main repeated-seed experiment
# ============================================================

def main() -> None:

    print()
    print("=" * 70)
    print(
        "REPEATED-SEED MULTI-DATASET "
        "HYBRID ATTENTION EXPERIMENT"
    )
    print("=" * 70)

    print(
        f"Frozen split seed: {SPLIT_SEED}"
    )

    print(
        "Training seeds: "
        + ", ".join(
            str(seed)
            for seed in TRAINING_SEEDS
        )
    )

    print(
        f"Datasets: {', '.join(DATASETS)}"
    )

    print(
        f"Total training runs: "
        f"{len(TRAINING_SEEDS) * len(DATASETS)}"
    )

    device = get_device()

    print(
        f"Device: {device}"
    )

    results = []

    # --------------------------------------------------------
    # Run every training seed on every dataset.
    # --------------------------------------------------------

    for training_seed in TRAINING_SEEDS:

        print()
        print("#" * 70)
        print(
            f"STARTING TRAINING SEED {training_seed}"
        )
        print("#" * 70)

        for dataset_name in DATASETS:

            result = run_dataset(
                dataset_name,
                training_seed,
                device,
            )

            results.append(
                result
            )

    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # Sort results consistently.
    results_df = results_df.sort_values(
        by=[
            "dataset",
            "training_seed",
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Verify frozen partition consistency.
    # --------------------------------------------------------

    check_partition_consistency(
        results
    )

    # --------------------------------------------------------
    # Save complete repeated-seed results.
    # --------------------------------------------------------

    results_path = (
        RESULTS_DIR
        / "hybrid_attention_repeated_seed_results.csv"
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save individual seed result files.
    # --------------------------------------------------------

    for training_seed in TRAINING_SEEDS:

        seed_results = results_df[
            results_df["training_seed"]
            == training_seed
        ].copy()

        seed_results_path = (
            RESULTS_DIR
            / (
                "hybrid_attention_"
                f"seed{training_seed}_results.csv"
            )
        )

        seed_results.to_csv(
            seed_results_path,
            index=False,
        )

    # --------------------------------------------------------
    # Experiment metadata
    # --------------------------------------------------------

    metadata = {
        "experiment": (
            "repeated_seed_multi_dataset_hybrid_attention"
        ),

        "split_seed": SPLIT_SEED,

        "training_seeds": TRAINING_SEEDS,

        "datasets": DATASETS,

        "total_runs": (
            len(TRAINING_SEEDS)
            * len(DATASETS)
        ),

        "batch_size": BATCH_SIZE,

        "epochs": EPOCHS,

        "learning_rate": LEARNING_RATE,

        "weight_decay": WEIGHT_DECAY,

        "early_stopping_patience": (
            EARLY_STOPPING_PATIENCE
        ),

        "lr_reduction_patience": (
            LR_REDUCTION_PATIENCE
        ),

        "lr_reduction_factor": (
            LR_REDUCTION_FACTOR
        ),

        "gradient_clip_norm": (
            GRADIENT_CLIP_NORM
        ),

        "embedding_dim": EMBEDDING_DIM,

        "num_heads": NUM_HEADS,

        "dropout": DROPOUT,

        "device": str(device),

        "parameter_count": count_parameters(
            HybridAttentionModel()
        ),

        "threshold": 0.5,

        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),

        "scaling_protocol": (
            "fit_train_only"
        ),

        "training_randomness": (
            "controlled_by_training_seed"
        ),

        "model_initialization": (
            "seed_controlled"
        ),

        "training_dataloader": (
            "shuffle_true_seed_controlled"
        ),

        "validation_dataloader": (
            "shuffle_false"
        ),

        "test_dataloader": (
            "shuffle_false"
        ),

        "model_selection": (
            "minimum_validation_loss"
        ),

        "test_usage": (
            "final_evaluation_only"
        ),

        "repeated_seed_design": (
            "same frozen partitions with independent "
            "training seeds"
        ),
    }

    metadata_path = (
        RESULTS_DIR
        / "hybrid_attention_repeated_seed_metadata.json"
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
    # Display final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("REPEATED-SEED EXPERIMENT COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Completed runs: {len(results_df)}"
    )

    print()
    print(
        results_df[
            [
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
    print("Results saved to:")
    print(results_path)

    print()
    print("Metadata saved to:")
    print(metadata_path)


if __name__ == "__main__":
    main()