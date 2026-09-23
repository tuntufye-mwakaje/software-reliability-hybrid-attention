import pandas as pd
from pathlib import Path

DATA_DIR = Path("research_analysis/datasets")
OUTPUT = Path("research_analysis/conflicting_feature_groups.csv")

records = []

for file in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(file)

    feature_columns = list(df.columns[:-1])
    target_column = df.columns[-1]

    # Count identical feature vectors and identify their distinct labels.
    grouped = (
        df.groupby(feature_columns, dropna=False)[target_column]
        .agg(
            occurrence_count="size",
            distinct_labels=lambda x: "|".join(sorted(set(x.astype(str))))
        )
        .reset_index()
    )

    conflicts = grouped[
        (grouped["occurrence_count"] > 1) &
        (grouped["distinct_labels"].str.contains(r"\|", regex=True))
    ].copy()

    if not conflicts.empty:
        conflicts.insert(0, "dataset", file.stem.upper())

        records.append(
            conflicts[
                ["dataset", "occurrence_count", "distinct_labels"]
                + feature_columns
            ]
        )

if records:
    result = pd.concat(records, ignore_index=True)
else:
    result = pd.DataFrame(
        columns=["dataset", "occurrence_count", "distinct_labels"]
    )

result.to_csv(OUTPUT, index=False)

print("Conflicting feature groups by dataset:")

if not result.empty:
    print(result.groupby("dataset").size().to_string())
else:
    print("None found.")

print()
print("Total conflicting feature groups:", len(result))
print()
print("Saved:", OUTPUT)
