import pandas as pd
from sklearn.model_selection import train_test_split


def create_feature_groups(df, feature_columns):
    """
    Assign one group ID to every unique predictor vector.

    Identical feature vectors receive the same group ID.
    Labels are not used to construct the groups.
    """

    group_ids = pd.factorize(
        pd.util.hash_pandas_object(
            df[feature_columns],
            index=False,
        )
    )[0]

    return pd.Series(
        group_ids,
        index=df.index,
        name="group_id",
    )


def grouped_train_val_test_split(
    df,
    feature_columns,
    target_column="defects",
    test_size=0.20,
    validation_size=0.20,
    random_state=42,
):
    """
    Split observations by unique feature-vector groups.

    Overall target proportions are approximately preserved while
    preventing identical predictor vectors from crossing partitions.

    Returns:
        train_df
        validation_df
        test_df
    """

    if test_size <= 0 or test_size >= 1:
        raise ValueError("test_size must be between 0 and 1.")

    if validation_size <= 0 or validation_size >= 1:
        raise ValueError(
            "validation_size must be between 0 and 1."
        )

    groups = create_feature_groups(
        df,
        feature_columns,
    )

    group_table = pd.DataFrame({
        "group_id": groups,
        "target": df[target_column].astype(bool),
    })

    # One row per unique feature vector.
    group_summary = (
        group_table
        .groupby("group_id")
        .agg(
            occurrences=("target", "size"),
            positive=("target", "sum"),
        )
        .reset_index()
    )

    group_summary["negative"] = (
        group_summary["occurrences"]
        - group_summary["positive"]
    )

    # Used only to approximately stratify the GROUPS.
    #
    # Original observation-level labels remain untouched.
    group_summary["stratify_label"] = (
        group_summary["positive"]
        >= group_summary["negative"]
    ).astype(int)

    # First: development/test split.
    train_val_groups, test_groups = train_test_split(
        group_summary,
        test_size=test_size,
        random_state=random_state,
        stratify=group_summary["stratify_label"],
    )

    # Second: train/validation split.
    #
    # validation_size is expressed relative to the complete dataset,
    # so convert it to the corresponding proportion of the
    # development partition.
    relative_validation_size = (
        validation_size / (1.0 - test_size)
    )

    train_groups, validation_groups = train_test_split(
        train_val_groups,
        test_size=relative_validation_size,
        random_state=random_state,
        stratify=train_val_groups["stratify_label"],
    )

    train_ids = set(train_groups["group_id"])
    validation_ids = set(validation_groups["group_id"])
    test_ids = set(test_groups["group_id"])

    # Hard safety check: no group may cross partitions.
    overlaps = (
        (train_ids & validation_ids)
        | (train_ids & test_ids)
        | (validation_ids & test_ids)
    )

    if overlaps:
        raise AssertionError(
            f"Feature-group leakage detected: {len(overlaps)} groups."
        )

    train_mask = groups.isin(train_ids)
    validation_mask = groups.isin(validation_ids)
    test_mask = groups.isin(test_ids)

    train_df = df.loc[train_mask].copy()
    validation_df = df.loc[validation_mask].copy()
    test_df = df.loc[test_mask].copy()

    # Every original observation must belong to exactly one partition.
    if (
        len(train_df)
        + len(validation_df)
        + len(test_df)
        != len(df)
    ):
        raise AssertionError(
            "Partition sizes do not account for every observation."
        )

    # Verify no feature-vector leakage after creating the actual
    # observation-level partitions.
    train_features = set(
        pd.util.hash_pandas_object(
            train_df[feature_columns],
            index=False,
        )
    )

    validation_features = set(
        pd.util.hash_pandas_object(
            validation_df[feature_columns],
            index=False,
        )
    )

    test_features = set(
        pd.util.hash_pandas_object(
            test_df[feature_columns],
            index=False,
        )
    )

    if train_features & validation_features:
        raise AssertionError(
            "Feature leakage detected between train and validation."
        )

    if train_features & test_features:
        raise AssertionError(
            "Feature leakage detected between train and test."
        )

    if validation_features & test_features:
        raise AssertionError(
            "Feature leakage detected between validation and test."
        )

    return (
        train_df,
        validation_df,
        test_df,
    )
