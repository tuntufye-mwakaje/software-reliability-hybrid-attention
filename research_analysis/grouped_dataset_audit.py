import pandas as pd
from pathlib import Path

DATA_DIR = Path("research_analysis/datasets")

for file in sorted(DATA_DIR.glob("*.csv")):
    df = pd.read_csv(file)

    # Normalize column names only for analysis.
    df = df.rename(columns={
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
    })

    # Normalize KC2 target.
    if "problems" in df.columns:
        df["defects"] = df["problems"].map({
            "no": False,
            "yes": True,
        })
        df = df.drop(columns=["problems"])

    feature_columns = [
        "loc", "v(g)", "ev(g)", "iv(g)",
        "n", "v", "l", "d", "i", "e", "b", "t",
        "lOCode", "lOComment", "lOBlank",
        "locCodeAndComment",
        "uniq_Op", "uniq_Opnd", "total_Op",
        "total_Opnd", "branchCount",
    ]

    X = df[feature_columns]
    y = df["defects"]

    # Each unique predictor vector is one group.
    group_ids = pd.util.hash_pandas_object(X, index=False)

    group_table = pd.DataFrame({
        "group_id": group_ids,
        "target": y.astype(bool),
    })

    summary = (
        group_table
        .groupby("group_id")["target"]
        .agg(
            occurrences="size",
            positive_count="sum",
            negative_count=lambda x: (~x).sum(),
        )
    )

    summary["conflicting_label"] = (
        (summary["positive_count"] > 0) &
        (summary["negative_count"] > 0)
    )

    print()
    print("=" * 60)
    print(file.stem.upper())
    print("=" * 60)

    print("Rows:", len(df))
    print("Unique feature groups:", len(summary))
    print("Positive rows:", int(y.sum()))
    print("Negative rows:", int((~y).sum()))
    print("Positive feature groups:", int((summary["positive_count"] > 0).sum()))
    print("Negative feature groups:", int((summary["negative_count"] > 0).sum()))
    print("Conflicting-label groups:", int(summary["conflicting_label"].sum()))
    print("Largest group size:", int(summary["occurrences"].max()))
    print("Singleton groups:", int((summary["occurrences"] == 1).sum()))
