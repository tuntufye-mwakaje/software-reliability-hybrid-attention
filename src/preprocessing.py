import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET_COLUMN = "defects"


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


FEATURE_COLUMNS = [
    feature
    for features in FEATURE_GROUPS.values()
    for feature in features
]


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split KC1 into training and test partitions.

    The split occurs before any fitted preprocessing operation.
    """

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def fit_scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
):
    """
    Fit StandardScaler on training data only and transform
    both training and test partitions.
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=FEATURE_COLUMNS,
        index=X_train.index,
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=FEATURE_COLUMNS,
        index=X_test.index,
    )

    return X_train_scaled, X_test_scaled, scaler


def create_feature_tokens(
    X_scaled: pd.DataFrame,
):
    """
    Convert standardized KC1 metrics into four feature-group tokens.

    Each token contains the standardized metrics belonging to one
    semantic feature group.

    Output shape:
        (samples, 4, maximum_group_size)

    Groups:
        1. size
        2. complexity
        3. halstead
        4. operators_operands

    Groups are zero-padded to the largest group size so that they
    can be represented as a single tensor.
    """

    group_arrays = []

    max_group_size = max(
        len(features)
        for features in FEATURE_GROUPS.values()
    )

    for group_name, features in FEATURE_GROUPS.items():

        values = X_scaled[features].to_numpy(dtype=np.float32)

        padding_width = max_group_size - values.shape[1]

        if padding_width > 0:
            values = np.pad(
                values,
                pad_width=((0, 0), (0, padding_width)),
                mode="constant",
                constant_values=0.0,
            )

        group_arrays.append(values)

    tokens = np.stack(group_arrays, axis=1)

    return tokens.astype(np.float32)


def prepare_attention_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Prepare KC1 for the hybrid attention model.

    Processing order:

        raw KC1
            |
            v
        train/test split
            |
            v
        fit scaler on training data only
            |
            v
        transform train/test
            |
            v
        create feature-group tokens
    """

    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = split_data(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    (
        X_train_scaled,
        X_test_scaled,
        scaler,
    ) = fit_scale_features(
        X_train,
        X_test,
    )

    X_train_tokens = create_feature_tokens(
        X_train_scaled
    )

    X_test_tokens = create_feature_tokens(
        X_test_scaled
    )

    return (
        X_train_tokens,
        X_test_tokens,
        y_train.to_numpy(dtype=np.float32),
        y_test.to_numpy(dtype=np.float32),
        scaler,
    )


if __name__ == "__main__":
    from data_loader import load_kc1

    df = load_kc1()

    (
        X_train_tokens,
        X_test_tokens,
        y_train,
        y_test,
        scaler,
    ) = prepare_attention_data(df)

    print("Training token shape:", X_train_tokens.shape)
    print("Test token shape:", X_test_tokens.shape)

    print("\nTraining target shape:", y_train.shape)
    print("Test target shape:", y_test.shape)

    print("\nTraining target distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nTest target distribution:")
    print(pd.Series(y_test).value_counts())

    print("\nTraining token summary:")
    print(
        "Mean:",
        round(float(X_train_tokens.mean()), 6),
    )
    print(
        "Standard deviation:",
        round(float(X_train_tokens.std()), 6),
    )

    print("\nToken dimensions:")
    print(
        "samples =",
        X_train_tokens.shape[0],
    )
    print(
        "feature groups =",
        X_train_tokens.shape[1],
    )
    print(
        "token width =",
        X_train_tokens.shape[2],
    )