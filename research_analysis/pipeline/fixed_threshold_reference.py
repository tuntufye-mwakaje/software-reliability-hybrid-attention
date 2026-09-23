"""
Fixed 0.5 Reference Evaluation

Purpose:
    Evaluate the frozen seed-42 Hybrid Attention checkpoints at the
    conventional fixed classification threshold of 0.5.

Research protocol:
    - Frozen seed-42 train/validation/test partitions
    - Training-only feature scaling
    - Existing seed-42 best checkpoints
    - No threshold optimization
    - Threshold fixed at 0.5
    - Validation and test evaluated independently
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research_preprocessing import prepare_research_dataset
from research_models import HybridAttentionModel
from threshold_analysis import classification_metrics


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

SEED = 42
FIXED_THRESHOLD = 0.5

DATASETS = [
    "CM1",
    "JM1",
    "KC1",
    "KC2",
    "PC1",
]

PIPELINE_DIR = Path(__file__).resolve().parent
RESEARCH_ANALYSIS_DIR = PIPELINE_DIR.parent

MODELS_DIR = RESEARCH_ANALYSIS_DIR / "models"
RESULTS_DIR = RESEARCH_ANALYSIS_DIR / "results"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------

def load_checkpoint(dataset_name):
    dataset_name = dataset_name.upper()

    checkpoint_path = (
        MODELS_DIR
        / f"{dataset_name.lower()}_seed42_best.pt"
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

    model = HybridAttentionModel(
        input_dim=8,
        embedding_dim=32,
        num_heads=4,
        dropout=0.1,
    )

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


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

def predict_probabilities(model, tokens):
    x = torch.tensor(
        tokens,
        dtype=torch.float32,
        device=DEVICE,
    )

    with torch.no_grad():
        logits = model(x)

    probabilities = (
        torch.sigmoid(logits)
        .cpu()
        .numpy()
    )

    return probabilities


# ---------------------------------------------------------------------
# Dataset evaluation
# ---------------------------------------------------------------------

def evaluate_dataset(dataset_name):
    print("\n" + "-" * 72)
    print(f"DATASET: {dataset_name}")
    print("-" * 72)

    prepared = prepare_research_dataset(
        dataset_name
    )

    validation = prepared["validation"]
    test = prepared["test"]

    model = load_checkpoint(
        dataset_name
    )

    validation_probabilities = predict_probabilities(
        model,
        validation["tokens"],
    )

    test_probabilities = predict_probabilities(
        model,
        test["tokens"],
    )

    validation_target = np.asarray(
        validation["target"],
        dtype=int,
    )

    test_target = np.asarray(
        test["target"],
        dtype=int,
    )

    validation_metrics = classification_metrics(
        validation_target,
        validation_probabilities,
        FIXED_THRESHOLD,
    )

    test_metrics = classification_metrics(
        test_target,
        test_probabilities,
        FIXED_THRESHOLD,
    )

    validation_row = {
        "dataset": dataset_name,
        "threshold": FIXED_THRESHOLD,
        **validation_metrics,
    }

    test_row = {
        "dataset": dataset_name,
        "threshold": FIXED_THRESHOLD,
        **test_metrics,
    }

    print("\nValidation performance:")
    print(
        pd.DataFrame([validation_row])
        .to_string(index=False)
    )

    print("\nFinal test performance:")
    print(
        pd.DataFrame([test_row])
        .to_string(index=False)
    )

    return validation_row, test_row


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validation_results = []
    test_results = []

    print("=" * 72)
    print("FIXED 0.5 REFERENCE EVALUATION")
    print("=" * 72)

    print(f"\nSeed: {SEED}")
    print(f"Fixed threshold: {FIXED_THRESHOLD}")
    print(f"Device: {DEVICE}")

    for dataset_name in DATASETS:
        validation_row, test_row = evaluate_dataset(
            dataset_name
        )

        validation_results.append(
            validation_row
        )

        test_results.append(
            test_row
        )

    validation_df = pd.DataFrame(
        validation_results
    )

    test_df = pd.DataFrame(
        test_results
    )

    # -------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------

    validation_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_fixed_0.5_validation.csv"
    )

    test_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_fixed_0.5_test.csv"
    )

    metadata_file = (
        RESULTS_DIR
        / "hybrid_attention_seed42_fixed_0.5_metadata.json"
    )

    validation_df.to_csv(
        validation_file,
        index=False,
    )

    test_df.to_csv(
        test_file,
        index=False,
    )

    metadata = {
        "experiment": (
            "hybrid_attention_seed42_fixed_0.5_reference"
        ),
        "seed": SEED,
        "datasets": DATASETS,
        "threshold": FIXED_THRESHOLD,
        "threshold_selection": "not_applicable_fixed_reference",
        "validation_data_used_for_threshold_selection": False,
        "test_data_used_for_threshold_selection": False,
        "test_evaluation": (
            "locked_evaluation_at_fixed_threshold_0.5"
        ),
        "model_checkpoint": (
            "seed42_best_checkpoint"
        ),
        "split_protocol": (
            "frozen_grouped_feature_split_seed42"
        ),
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

    # -------------------------------------------------------------
    # Final summary
    # -------------------------------------------------------------

    print("\n" + "=" * 72)
    print("FIXED 0.5 REFERENCE EVALUATION COMPLETED")
    print("=" * 72)

    print("\nValidation results:")
    print(
        validation_df.to_string(
            index=False
        )
    )

    print("\nFinal test results:")
    print(
        test_df.to_string(
            index=False
        )
    )

    print("\nSaved files:")
    print(f"  {validation_file}")
    print(f"  {test_file}")
    print(f"  {metadata_file}")


if __name__ == "__main__":
    main()