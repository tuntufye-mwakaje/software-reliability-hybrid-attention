from pathlib import Path
import json
import math
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "research_analysis" / "results"
MODELS_DIR = PROJECT_ROOT / "research_analysis" / "models"

EXPECTED = {
    "CM1":  {"train": 309,  "validation": 84,   "test": 105,  "positive": 10,  "negative": 95},
    "JM1":  {"train": 8491, "validation": 2190, "test": 2523, "positive": 420, "negative": 2103},
    "KC1":  {"train": 1370, "validation": 307,  "test": 432,  "positive": 67,  "negative": 365},
    "KC2":  {"train": 346,  "validation": 66,   "test": 110,  "positive": 21,  "negative": 89},
    "PC1":  {"train": 722,  "validation": 171,  "test": 216, "positive": 18, "negative": 198},
}

DATASETS = list(EXPECTED.keys())
TOL = 1e-6

RESULTS_FILE = RESULTS_DIR / "hybrid_attention_seed42_results.csv"
METADATA_FILE = RESULTS_DIR / "hybrid_attention_seed42_metadata.json"


def fail(message):
    raise AssertionError(message)


def check_close(name, actual, expected, tolerance=TOL):
    if not math.isclose(float(actual), float(expected),
                        rel_tol=tolerance, abs_tol=tolerance):
        fail(f"{name}: expected {expected}, got {actual}")


def main():

    print("=" * 72)
    print("RESEARCH EXPERIMENT ARTIFACT AUDIT")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 1. Results CSV
    # ---------------------------------------------------------------
    print("\n[1] Checking results CSV...")

    if not RESULTS_FILE.exists():
        fail(f"Missing results file: {RESULTS_FILE}")

    df = pd.read_csv(RESULTS_FILE)

    if set(df["dataset"]) != set(DATASETS):
        fail(
            f"Dataset mismatch. Expected {DATASETS}, "
            f"found {df['dataset'].tolist()}"
        )

    if len(df) != len(DATASETS):
        fail(f"Expected {len(DATASETS)} rows, found {len(df)}")

    if df["dataset"].duplicated().any():
        fail("Duplicate dataset result rows detected")

    required_columns = [
        "dataset",
        "training_seed",
        "split_seed",
        "best_epoch",
        "best_validation_loss",
        "test_loss",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "train_samples",
        "validation_samples",
        "test_samples",
        "parameter_count",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "embedding_dim",
        "num_heads",
        "dropout",
        "threshold",
    ]

    missing = [c for c in required_columns if c not in df.columns]

    if missing:
        fail(f"Missing required result columns: {missing}")

    print(f"  PASS: {len(df)} expected dataset results found")

    # ---------------------------------------------------------------
    # 2. Per-dataset validation
    # ---------------------------------------------------------------
    print("\n[2] Checking per-dataset counts and metrics...")

    for _, row in df.iterrows():

        dataset = row["dataset"]
        expected = EXPECTED[dataset]

        # Fixed experiment configuration
        # Training seed and frozen grouped split seed are both 42
        # for this fixed-seed reference experiment.
        if int(row["training_seed"]) != 42:
            fail(f"{dataset}: training_seed is not 42")

        if int(row["split_seed"]) != 42:
            fail(f"{dataset}: split_seed is not 42")

        if float(row["threshold"]) != 0.5:
            fail(f"{dataset}: threshold is not 0.5")

        if int(row["parameter_count"]) != 9985:
            fail(
                f"{dataset}: expected 9985 parameters, "
                f"got {row['parameter_count']}"
            )

        if int(row["batch_size"]) != 32:
            fail(f"{dataset}: batch_size is not 32")

        check_close(
            f"{dataset} learning_rate",
            row["learning_rate"],
            0.001
        )

        check_close(
            f"{dataset} weight_decay",
            row["weight_decay"],
            0.0001
        )

        if int(row["embedding_dim"]) != 32:
            fail(f"{dataset}: embedding_dim is not 32")

        if int(row["num_heads"]) != 4:
            fail(f"{dataset}: num_heads is not 4")

        check_close(
            f"{dataset} dropout",
            row["dropout"],
            0.1
        )

        # Sample counts
        for key, column in [
            ("train", "train_samples"),
            ("validation", "validation_samples"),
            ("test", "test_samples"),
        ]:
            if int(row[column]) != expected[key]:
                fail(
                    f"{dataset}: {column} expected {expected[key]}, "
                    f"got {row[column]}"
                )

        # Test class counts derived from confusion matrix
        tn = int(row["true_negatives"])
        fp = int(row["false_positives"])
        fn = int(row["false_negatives"])
        tp = int(row["true_positives"])

        if tn + fp + fn + tp != expected["test"]:
            fail(
                f"{dataset}: confusion matrix total "
                f"{tn + fp + fn + tp} != test size {expected['test']}"
            )

        if tp + fn != expected["positive"]:
            fail(
                f"{dataset}: TP+FN = {tp + fn}, "
                f"expected {expected['positive']}"
            )

        if tn + fp != expected["negative"]:
            fail(
                f"{dataset}: TN+FP = {tn + fp}, "
                f"expected {expected['negative']}"
            )

        # Recompute metrics from confusion matrix
        total = tn + fp + fn + tp

        accuracy = (tn + tp) / total

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

        check_close(f"{dataset} accuracy", row["accuracy"], accuracy)
        check_close(f"{dataset} precision", row["precision"], precision)
        check_close(f"{dataset} recall", row["recall"], recall)
        check_close(f"{dataset} F1", row["f1"], f1)

        # ROC-AUC and PR-AUC
        for metric in ["roc_auc", "pr_auc"]:

            value = float(row[metric])

            if not math.isfinite(value):
                fail(f"{dataset}: {metric} is not finite")

            if not 0.0 <= value <= 1.0:
                fail(
                    f"{dataset}: {metric} outside [0,1]: {value}"
                )

        # Losses
        for metric in ["best_validation_loss", "test_loss"]:

            value = float(row[metric])

            if not math.isfinite(value) or value < 0:
                fail(
                    f"{dataset}: invalid {metric}: {value}"
                )

        # Epoch
        best_epoch = int(row["best_epoch"])

        if not 1 <= best_epoch <= 50:
            fail(
                f"{dataset}: invalid best_epoch {best_epoch}"
            )

        print(
            f"  PASS {dataset}: "
            f"train={int(row['train_samples'])}, "
            f"val={int(row['validation_samples'])}, "
            f"test={int(row['test_samples'])}, "
            f"ROC-AUC={float(row['roc_auc']):.6f}, "
            f"PR-AUC={float(row['pr_auc']):.6f}, "
            f"F1={float(row['f1']):.6f}"
        )

    # ---------------------------------------------------------------
    # 3. Metadata
    # ---------------------------------------------------------------
    print("\n[3] Checking experiment metadata...")

    if not METADATA_FILE.exists():
        fail(f"Missing metadata file: {METADATA_FILE}")

    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    metadata_text = json.dumps(metadata)

    required_metadata_values = [
        "seed42",
        "frozen_grouped_feature_split_seed42",
        "fit_train_only",
        "minimum_validation_loss",
        "final_evaluation_only",
    ]

    for required_value in required_metadata_values:

        if required_value not in metadata_text:
            fail(
                "Metadata does not contain expected value: "
                f"{required_value}"
            )

    print("  PASS: fixed seed and frozen evaluation protocol recorded")

    # ---------------------------------------------------------------
    # 4. Model checkpoints
    # ---------------------------------------------------------------
    print("\n[4] Checking model checkpoints...")

    for dataset in DATASETS:

        checkpoint = (
            MODELS_DIR / f"{dataset.lower()}_seed42_best.pt"
        )

        if not checkpoint.exists():
            fail(
                f"{dataset}: missing checkpoint "
                f"{checkpoint.name}"
            )

        try:
            state = torch.load(
                checkpoint,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            fail(
                f"{dataset}: checkpoint could not be loaded: {exc}"
            )

        if not isinstance(state, dict):
            fail(
                f"{dataset}: checkpoint is not a dictionary"
            )

        if "model_state_dict" not in state:
            fail(
                f"{dataset}: checkpoint missing model_state_dict"
            )

        print(
            f"  PASS {dataset}: {checkpoint.name}"
        )

    # ---------------------------------------------------------------
    # 5. Training histories
    # ---------------------------------------------------------------
    print("\n[5] Checking training histories...")

    for dataset in DATASETS:

        history_file = (
            RESULTS_DIR
            / f"{dataset.lower()}_seed42_training_history.csv"
        )

        if not history_file.exists():
            fail(
                f"{dataset}: missing training history"
            )

        history = pd.read_csv(history_file)

        if "epoch" not in history.columns:
            fail(
                f"{dataset}: history missing epoch column"
            )

        epochs = (
            history["epoch"]
            .astype(int)
            .tolist()
        )

        if not epochs:
            fail(
                f"{dataset}: empty training history"
            )

        expected_epochs = list(
            range(1, max(epochs) + 1)
        )

        if epochs != expected_epochs:
            fail(
                f"{dataset}: epochs are not contiguous "
                f"from 1: {epochs}"
            )

        numeric_columns = [
            c for c in history.columns
            if c != "epoch"
        ]

        for column in numeric_columns:

            values = pd.to_numeric(
                history[column],
                errors="coerce"
            )

            if values.isna().any():
                fail(
                    f"{dataset}: NaN/non-numeric values "
                    f"in history column {column}"
                )

            if not values.map(math.isfinite).all():
                fail(
                    f"{dataset}: non-finite values "
                    f"in history column {column}"
                )

        print(
            f"  PASS {dataset}: "
            f"{len(history)} recorded epochs "
            f"(1-{max(epochs)})"
        )

    # ---------------------------------------------------------------
    # Final result
    # ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("ALL RESEARCH RESULT ARTIFACT AUDITS PASSED")
    print("=" * 72)

    print("\nValidated:")
    print("  - Exactly five expected dataset results")
    print("  - No duplicate dataset rows")
    print("  - Frozen train/validation/test counts")
    print("  - Frozen test class counts")
    print("  - Confusion-matrix totals")
    print("  - Accuracy, precision, recall and F1")
    print("  - ROC-AUC and PR-AUC")
    print("  - Seed = 42")
    print("  - Threshold = 0.5")
    print("  - Parameter count = 9,985")
    print("  - Model hyperparameters")
    print("  - Experiment metadata")
    print("  - Five model checkpoints")
    print("  - Five training histories")
    print("\nNo training was performed by this audit.")


if __name__ == "__main__":
    main()
