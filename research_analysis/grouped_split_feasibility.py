import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("research_analysis/datasets")

FEATURE_COLUMNS = [
    "loc", "v(g)", "ev(g)", "iv(g)",
    "n", "v", "l", "d", "i", "e", "b", "t",
    "lOCode", "lOComment", "lOBlank",
    "locCodeAndComment",
    "uniq_Op", "uniq_Opnd", "total_Op",
    "total_Opnd", "branchCount",
]


def load_and_normalize(file):
    df = pd.read_csv(file)

    # Normalize metric names.
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

    return df


def create_groups(df):
    X = df[FEATURE_COLUMNS]

    # Identical predictor vectors form one leakage-prevention group.
    group_ids = pd.factorize(
        pd.util.hash_pandas_object(X, index=False)
    )[0]

    group_table = pd.DataFrame({
        "group_id": group_ids,
        "target": df["defects"].astype(bool),
    })

    # One row per unique feature group.
    groups = (
        group_table
        .groupby("group_id")
        .agg(
            occurrences=("target", "size"),
            positive=("target", "sum"),
        )
        .reset_index()
    )

    groups["negative"] = (
        groups["occurrences"] - groups["positive"]
    )

    # A group is assigned a stratification class.
    #
    # For conflicting-label groups, assign the majority label.
    # The complete group will still remain together.
    groups["stratify_label"] = (
        groups["positive"] >= groups["negative"]
    ).astype(int)

    return groups


def summarize_partition(name, df, selected_groups, groups):
    selected = groups[groups["group_id"].isin(selected_groups)]

    rows = int(selected["occurrences"].sum())
    positives = int(selected["positive"].sum())
    negatives = int(selected["negative"].sum())

    print(
        f"{name:<10} "
        f"groups={len(selected):>5} "
        f"rows={rows:>6} "
        f"positive={positives:>5} "
        f"negative={negatives:>6} "
        f"positive_rate={positives / rows:.4f}"
    )


for file in sorted(DATA_DIR.glob("*.csv")):
    df = load_and_normalize(file)
    groups = create_groups(df)

    print()
    print("=" * 70)
    print(file.stem.upper())
    print("=" * 70)

    print("Total rows:", len(df))
    print("Total groups:", len(groups))
    print(
        "Conflicting groups:",
        int(
            (
                (groups["positive"] > 0)
                & (groups["negative"] > 0)
            ).sum()
        )
    )

    # First split:
    # 80% development groups / 20% test groups
    train_val_groups, test_groups = train_test_split(
        groups,
        test_size=0.20,
        random_state=42,
        stratify=groups["stratify_label"],
    )

    # Second split:
    # 80% of development -> train
    # 20% of development -> validation
    train_groups, val_groups = train_test_split(
        train_val_groups,
        test_size=0.20,
        random_state=42,
        stratify=train_val_groups["stratify_label"],
    )

    summarize_partition(
        "TRAIN",
        df,
        train_groups["group_id"],
        groups,
    )

    summarize_partition(
        "VALIDATION",
        df,
        val_groups["group_id"],
        groups,
    )

    summarize_partition(
        "TEST",
        df,
        test_groups["group_id"],
        groups,
    )

    # Verify no group appears in more than one partition.
    train_ids = set(train_groups["group_id"])
    val_ids = set(val_groups["group_id"])
    test_ids = set(test_groups["group_id"])

    overlap = (
        (train_ids & val_ids)
        | (train_ids & test_ids)
        | (val_ids & test_ids)
    )

    print("Cross-partition group overlap:", len(overlap))

    # Verify both classes exist in every partition.
    for name, selected in [
        ("TRAIN", train_groups),
        ("VALIDATION", val_groups),
        ("TEST", test_groups),
    ]:
        positives = int(selected["positive"].sum())
        negatives = int(selected["negative"].sum())

        print(
            f"{name} class check: "
            f"positive={positives > 0}, "
            f"negative={negatives > 0}"
        )
