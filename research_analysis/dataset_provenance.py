import hashlib
import pandas as pd
from pathlib import Path

DATA_DIR = Path("research_analysis/datasets")
OUTPUT = Path("research_analysis/dataset_provenance.csv")


def sha256_file(path):
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


records = []

for file in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(file)

    target_column = (
        "problems"
        if "problems" in df.columns
        else "defects"
    )

    target_counts = (
        df[target_column]
        .value_counts(dropna=False)
        .to_dict()
    )

    records.append({
        "dataset": file.stem.upper(),
        "filename": file.name,
        "sha256": sha256_file(file),
        "file_size_bytes": file.stat().st_size,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_cells": int(df.isna().sum().sum()),
        "duplicate_full_rows": int(df.duplicated().sum()),
        "target_column": target_column,
        "target_distribution": str(target_counts),
        "columns_exact": " | ".join(df.columns),
    })


result = pd.DataFrame(records)

result.to_csv(OUTPUT, index=False)

print()
print("=" * 80)
print("DATASET PROVENANCE AUDIT")
print("=" * 80)
print()

print(
    result[
        [
            "dataset",
            "filename",
            "rows",
            "columns",
            "missing_cells",
            "duplicate_full_rows",
            "target_column",
        ]
    ].to_string(index=False)
)

print()
print("SHA-256 hashes:")
print()

for _, row in result.iterrows():
    print(f"{row['dataset']}:")
    print(f"  {row['sha256']}")
    print()

print(f"Saved: {OUTPUT}")
