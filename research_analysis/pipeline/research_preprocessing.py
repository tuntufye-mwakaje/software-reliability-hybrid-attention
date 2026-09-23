import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

from dataset_schema import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)

from research_data import load_research_dataset


# ---------------------------------------------------------------------
# Semantic feature groups
# ---------------------------------------------------------------------

FEATURE_GROUPS = {
    "size": [
        "loc",
        "lOCode",
        "lOComment",
        "lOBlank",
        "locCodeAndComment",
    ],
    "complexity": [
        "v(g)",
        "ev(g)",
        "iv(g)",
        "branchCount",
    ],
    "halstead": [
        "n",
        "v",
        "l",
        "d",
        "i",
        "e",
        "b",
        "t",
    ],
    "operators_operands": [
        "uniq_Op",
        "uniq_Opnd",
        "total_Op",
        "total_Opnd",
    ],
}


# ---------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------

def validate_feature_groups():
    """
    Verify that the semantic feature groups account for exactly
    the canonical 21 predictor variables.
    """

    grouped_features = [
        feature
        for features in FEATURE_GROUPS.values()
        for feature in features
    ]

    if len(grouped_features) != len(FEATURE_COLUMNS):
        raise ValueError(
            "Feature-group configuration does not contain "
            "exactly the canonical number of predictors."
        )

    if len(set(grouped_features)) != len(grouped_features):
        raise ValueError(
            "Duplicate feature detected across feature groups."
        )

    if set(grouped_features) != set(FEATURE_COLUMNS):
        missing = set(FEATURE_COLUMNS) - set(grouped_features)
        extra = set(grouped_features) - set(FEATURE_COLUMNS)

        raise ValueError(
            f"Feature-group mismatch. "
            f"Missing={sorted(missing)}, "
            f"Extra={sorted(extra)}"
        )

    return True


# ---------------------------------------------------------------------
# Training-only scaling
# ---------------------------------------------------------------------

def fit_training_scaler(train_df):
    """
    Fit StandardScaler using training observations only.
    """

    validate_feature_groups()

    X_train = train_df[FEATURE_COLUMNS].copy()

    scaler = StandardScaler()

    scaler.fit(X_train)

    return scaler


def transform_features(df, scaler):
    """
    Transform predictors using an already-fitted training scaler.

    No fitting occurs in this function.
    """

    X = df[FEATURE_COLUMNS].copy()

    X_scaled = scaler.transform(X)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=FEATURE_COLUMNS,
        index=df.index,
    )

    return X_scaled


# ---------------------------------------------------------------------
# Feature-group token construction
# ---------------------------------------------------------------------

def create_feature_tokens(X_scaled):
    """
    Convert standardized predictors into four semantic feature-group
    tokens.

    Output shape:

        (samples, 4, 8)

    The largest semantic group contains 8 features. Smaller groups
    are zero-padded to width 8.
    """

    validate_feature_groups()

    group_arrays = []

    max_group_size = max(
        len(features)
        for features in FEATURE_GROUPS.values()
    )

    for group_name, features in FEATURE_GROUPS.items():

        values = X_scaled[
            features
        ].to_numpy(
            dtype=np.float32
        )

        padding_width = (
            max_group_size
            - values.shape[1]
        )

        if padding_width > 0:

            values = np.pad(
                values,
                pad_width=(
                    (0, 0),
                    (0, padding_width),
                ),
                mode="constant",
                constant_values=0.0,
            )

        group_arrays.append(values)

    tokens = np.stack(
        group_arrays,
        axis=1,
    )

    return tokens.astype(np.float32)


# ---------------------------------------------------------------------
# Complete frozen-dataset preprocessing
# ---------------------------------------------------------------------

def prepare_research_dataset(dataset_name):
    """
    Prepare one frozen research dataset.

    Processing order:

        frozen raw dataset
                |
                v
        frozen train/validation/test partitions
                |
                v
        fit scaler on TRAIN ONLY
                |
                v
        transform train/validation/test
                |
                v
        create semantic feature tokens
                |
                v
        return tensors and metadata
    """

    (
        train_df,
        validation_df,
        test_df,
    ) = load_research_dataset(dataset_name)

    # -------------------------------------------------------------
    # Fit ONLY on training data
    # -------------------------------------------------------------

    scaler = fit_training_scaler(
        train_df
    )

    # -------------------------------------------------------------
    # Transform all partitions using the training scaler
    # -------------------------------------------------------------

    X_train_scaled = transform_features(
        train_df,
        scaler,
    )

    X_validation_scaled = transform_features(
        validation_df,
        scaler,
    )

    X_test_scaled = transform_features(
        test_df,
        scaler,
    )

    # -------------------------------------------------------------
    # Convert standardized predictors into attention tokens
    # -------------------------------------------------------------

    X_train_tokens = create_feature_tokens(
        X_train_scaled
    )

    X_validation_tokens = create_feature_tokens(
        X_validation_scaled
    )

    X_test_tokens = create_feature_tokens(
        X_test_scaled
    )

    # -------------------------------------------------------------
    # Targets
    # -------------------------------------------------------------

    y_train = train_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    y_validation = validation_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    y_test = test_df[
        TARGET_COLUMN
    ].to_numpy(
        dtype=np.float32
    )

    return {
        "dataset": dataset_name.upper(),

        "train": {
            "tokens": X_train_tokens,
            "target": y_train,
            "row_ids": train_df["row_id"].to_numpy(
                dtype=np.int64
            ),
        },

        "validation": {
            "tokens": X_validation_tokens,
            "target": y_validation,
            "row_ids": validation_df["row_id"].to_numpy(
                dtype=np.int64
            ),
        },

        "test": {
            "tokens": X_test_tokens,
            "target": y_test,
            "row_ids": test_df["row_id"].to_numpy(
                dtype=np.int64
            ),
        },

        "scaler": scaler,

        "feature_groups": FEATURE_GROUPS,
    }


# ---------------------------------------------------------------------
# Command-line validation
# ---------------------------------------------------------------------

if __name__ == "__main__":

    print("Research preprocessing validation")
    print("=" * 70)

    validate_feature_groups()

    print()
    print("Feature groups:")
    for group_name, features in FEATURE_GROUPS.items():
        print(
            f"  {group_name}: "
            f"{len(features)} features"
        )

    print()
    print(
        f"Total predictors: {len(FEATURE_COLUMNS)}"
    )

    print(
        "Maximum token width:",
        max(
            len(features)
            for features in FEATURE_GROUPS.values()
        ),
    )

    print()
    print("-" * 70)

    for dataset_name in [
        "CM1",
        "JM1",
        "KC1",
        "KC2",
        "PC1",
    ]:

        data = prepare_research_dataset(
            dataset_name
        )

        train_tokens = data["train"]["tokens"]
        validation_tokens = data["validation"]["tokens"]
        test_tokens = data["test"]["tokens"]

        print()
        print(f"Dataset: {dataset_name}")

        print(
            f"  Train tokens:      "
            f"{train_tokens.shape}"
        )

        print(
            f"  Validation tokens: "
            f"{validation_tokens.shape}"
        )

        print(
            f"  Test tokens:       "
            f"{test_tokens.shape}"
        )

        print(
            f"  Train targets:      "
            f"{data['train']['target'].shape}"
        )

        print(
            f"  Validation targets: "
            f"{data['validation']['target'].shape}"
        )

        print(
            f"  Test targets:       "
            f"{data['test']['target'].shape}"
        )

    print()
    print("=" * 70)
    print("Research preprocessing validation completed successfully.")