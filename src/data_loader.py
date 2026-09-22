from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PATH = DATA_DIR / "kc1.csv"

OPENML_DATA_ID = 1067

EXPECTED_COLUMNS = [
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
    "defects",
]

EXPECTED_ROWS = 2109
EXPECTED_DEFECT_COUNTS = {
    False: 1783,
    True: 326,
}


def download_kc1(output_path: Path = DATA_PATH) -> pd.DataFrame:
    """Download KC1 from OpenML and save it locally."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_openml(
        data_id=OPENML_DATA_ID,
        as_frame=True,
        parser="auto",
    )

    df = dataset.frame.copy()
    df.to_csv(output_path, index=False)

    return df


def load_kc1(
    data_path: Path = DATA_PATH,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load and validate the KC1 dataset."""
    data_path = Path(data_path)

    if not data_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"KC1 dataset not found at: {data_path}"
            )

        print("KC1 dataset not found. Downloading from OpenML...")
        df = download_kc1(data_path)
    else:
        df = pd.read_csv(data_path)

    validate_kc1(df)

    return df


def validate_kc1(df: pd.DataFrame) -> None:
    """Validate the KC1 schema and expected target distribution."""
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected KC1 columns.\n"
            f"Expected: {EXPECTED_COLUMNS}\n"
            f"Received: {list(df.columns)}"
        )

    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected number of rows: {len(df)}. "
            f"Expected {EXPECTED_ROWS}."
        )

    defect_counts = df["defects"].value_counts().to_dict()

    if defect_counts != EXPECTED_DEFECT_COUNTS:
        raise ValueError(
            "Unexpected defect distribution.\n"
            f"Expected: {EXPECTED_DEFECT_COUNTS}\n"
            f"Received: {defect_counts}"
        )


if __name__ == "__main__":
    dataset = load_kc1()

    print(f"Dataset path: {DATA_PATH}")
    print(f"Shape: {dataset.shape}")
    print("\nTarget distribution:")
    print(dataset["defects"].value_counts())