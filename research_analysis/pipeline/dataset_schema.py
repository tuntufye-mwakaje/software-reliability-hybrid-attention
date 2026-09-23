from pathlib import Path
import pandas as pd


FEATURE_COLUMNS = [
    "loc",
    "v(g)",
    "ev(g)",
    "iv(g)",
    "n",
    "v",
    "l",
    "d",
    "i",
    "e",
    "b",
    "t",
    "lOCode",
    "lOComment",
    "lOBlank",
    "locCodeAndComment",
    "uniq_Op",
    "uniq_Opnd",
    "total_Op",
    "total_Opnd",
    "branchCount",
]

TARGET_COLUMN = "defects"

CANONICAL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


COLUMN_ALIASES = {
    "iv(G)": "iv(g)",
    "N": "n",
    "V": "v",
    "L": "l",
    "D": "d",
    "I": "i",
    "E": "e",
    "B": "b",
    "T": "t",
    "lOCodeAndComment": "locCodeAndComment",
}


def load_dataset(path):
    """
    Load one raw NASA/PROMISE dataset and normalize it
    into the canonical research schema.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)

    # Normalize known column aliases.
    df = df.rename(columns=COLUMN_ALIASES)

    # Normalize KC2 target.
    if "problems" in df.columns:
        df["defects"] = df["problems"].map({
            "no": False,
            "yes": True,
        })

        if df["defects"].isna().any():
            raise ValueError(
                f"Unexpected KC2 target values in {path}"
            )

        df = df.drop(columns=["problems"])

    # Validate columns.
    missing_columns = [
        col for col in CANONICAL_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing canonical columns in {path}: "
            f"{missing_columns}"
        )

    # Keep only the canonical research variables.
    df = df[CANONICAL_COLUMNS].copy()

    # Validate missing values.
    missing_cells = int(df.isna().sum().sum())

    if missing_cells > 0:
        raise ValueError(
            f"{path} contains {missing_cells} missing cells."
        )

    # Validate predictor types.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # Normalize target to boolean.
    if df[TARGET_COLUMN].dtype != bool:
        if set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(bool)
        else:
            raise ValueError(
                f"Target column in {path} could not be "
                f"normalized to boolean."
            )

    # Final schema assertion.
    if list(df.columns) != CANONICAL_COLUMNS:
        raise AssertionError(
            f"Canonical schema mismatch for {path}"
        )

    return df


def canonicalize_dataset(df, source_name="<dataframe>"):
    """
    Normalize an already-loaded DataFrame into the canonical
    research schema.

    This mirrors the normalization performed by load_dataset(),
    but accepts a DataFrame instead of a file path.
    """

    df = df.copy()

    # Normalize known column aliases.
    df = df.rename(columns=COLUMN_ALIASES)

    # Normalize KC2 target.
    if "problems" in df.columns:
        df["defects"] = df["problems"].map({
            "no": False,
            "yes": True,
        })

        if df["defects"].isna().any():
            raise ValueError(
                f"Unexpected KC2 target values in {source_name}"
            )

        df = df.drop(columns=["problems"])

    # Validate columns.
    missing_columns = [
        col for col in CANONICAL_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing canonical columns in {source_name}: "
            f"{missing_columns}"
        )

    # Keep only canonical research variables.
    df = df[CANONICAL_COLUMNS].copy()

    # Validate missing values.
    missing_cells = int(df.isna().sum().sum())

    if missing_cells > 0:
        raise ValueError(
            f"{source_name} contains {missing_cells} missing cells."
        )

    # Convert predictors to numeric.
    for column in FEATURE_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # Normalize target to boolean.
    if df[TARGET_COLUMN].dtype != bool:
        if set(df[TARGET_COLUMN].unique()).issubset({0, 1}):
            df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(bool)
        else:
            raise ValueError(
                f"Target column in {source_name} could not be "
                f"normalized to boolean."
            )

    # Final schema assertion.
    if list(df.columns) != CANONICAL_COLUMNS:
        raise AssertionError(
            f"Canonical schema mismatch for {source_name}"
        )

    return df
