import pandas as pd

from data_loader import load_kc1


FEATURE_GROUPS = {
    "Size": [
        "loc",
        "lOCode",
        "lOComment",
        "lOBlank",
        "locCodeAndComment",
    ],
    "Complexity": [
        "v(g)",
        "ev(g)",
        "iv(g)",
        "branchCount",
    ],
    "Halstead": [
        "n",
        "v",
        "l",
        "d",
        "i",
        "e",
        "b",
        "t",
    ],
    "Operators_Operands": [
        "uniq_Op",
        "uniq_Opnd",
        "total_Op",
        "total_Opnd",
    ],
}


def validate_feature_groups(df: pd.DataFrame) -> None:
    """Verify that every KC1 feature appears exactly once."""

    grouped_features = [
        feature
        for features in FEATURE_GROUPS.values()
        for feature in features
    ]

    expected_features = [
        column
        for column in df.columns
        if column != "defects"
    ]

    if len(grouped_features) != len(set(grouped_features)):
        raise ValueError("A feature appears in more than one feature group.")

    if set(grouped_features) != set(expected_features):
        missing = sorted(set(expected_features) - set(grouped_features))
        extra = sorted(set(grouped_features) - set(expected_features))

        raise ValueError(
            f"Feature grouping does not match KC1.\n"
            f"Missing features: {missing}\n"
            f"Unexpected features: {extra}"
        )


def print_feature_profile(df: pd.DataFrame) -> None:
    """Print descriptive statistics for every KC1 feature."""

    print("\nKC1 FEATURE PROFILE")
    print("=" * 80)

    profile = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "min": df.min(numeric_only=True),
            "mean": df.mean(numeric_only=True),
            "std": df.std(numeric_only=True),
            "max": df.max(numeric_only=True),
            "unique": df.nunique(),
        }
    )

    print(profile.to_string())

    print("\nFEATURE GROUPS")
    print("=" * 80)

    for group_name, features in FEATURE_GROUPS.items():
        print(f"\n{group_name} ({len(features)} features)")
        for feature in features:
            print(f"  - {feature}")


if __name__ == "__main__":
    df = load_kc1()

    validate_feature_groups(df)
    print_feature_profile(df)

    print("\nValidation:")
    print("- All 21 KC1 predictor features are assigned exactly once.")
    print("- Target column 'defects' is excluded from feature groups.")
    print("- Feature grouping is valid.")