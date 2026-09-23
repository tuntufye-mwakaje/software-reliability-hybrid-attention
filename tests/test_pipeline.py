import numpy as np
import pandas as pd
import pytest
import torch

from src.dataset_schema import (
    CANONICAL_COLUMNS, FEATURE_COLUMNS, TARGET_COLUMN, canonicalize_dataset,
)
from src.grouped_split import create_feature_groups, grouped_train_val_test_split
from src.research_models import HybridAttentionModel
from src.research_preprocessing import (FEATURE_GROUPS, create_feature_tokens, fit_training_scaler, transform_features)

def make_synthetic_dataset(n_rows=80):
    rng = np.random.default_rng(42)
    data = {feature: rng.normal(size=n_rows) for feature in FEATURE_COLUMNS}
    data[TARGET_COLUMN] = np.array([i % 2 == 0 for i in range(n_rows)])
    return pd.DataFrame(data)

def test_canonical_schema():
    assert len(FEATURE_COLUMNS) == 21
    assert TARGET_COLUMN == "defects"
    assert CANONICAL_COLUMNS == FEATURE_COLUMNS + [TARGET_COLUMN]

def test_feature_groups_cover_all_features_exactly_once():
    grouped = [feature for features in FEATURE_GROUPS.values() for feature in features]
    assert len(grouped) == 21
    assert len(set(grouped)) == 21
    assert set(grouped) == set(FEATURE_COLUMNS)
    assert {name: len(features) for name, features in FEATURE_GROUPS.items()} == {"size": 5, "complexity": 4, "halstead": 8, "operators_operands": 4}

def test_canonicalize_dataset_normalizes_aliases_and_target():
    df = make_synthetic_dataset(10)
    df = df.rename(columns={"iv(g)": "iv(G)", "n": "N"})
    df["defects"] = [0, 1] * 5
    result = canonicalize_dataset(df)
    assert list(result.columns) == CANONICAL_COLUMNS
    assert result["defects"].dtype == bool
    assert result["defects"].tolist() == [False, True] * 5

def test_invalid_target_is_rejected():
    df = make_synthetic_dataset(10)
    df["defects"] = ["invalid"] * 10
    with pytest.raises(ValueError):
        canonicalize_dataset(df)

def test_grouped_split_prevents_feature_vector_leakage():
    df = make_synthetic_dataset(80)
    duplicate = df.iloc[[0]].copy()
    duplicate["defects"] = True
    df = pd.concat([df, duplicate], ignore_index=True)
    train, validation, test = grouped_train_val_test_split(df, FEATURE_COLUMNS, target_column=TARGET_COLUMN, test_size=0.20, validation_size=0.20, random_state=42)
    train_hashes = set(pd.util.hash_pandas_object(train[FEATURE_COLUMNS], index=False))
    validation_hashes = set(pd.util.hash_pandas_object(validation[FEATURE_COLUMNS], index=False))
    test_hashes = set(pd.util.hash_pandas_object(test[FEATURE_COLUMNS], index=False))
    assert not train_hashes & validation_hashes
    assert not train_hashes & test_hashes
    assert not validation_hashes & test_hashes
    assert len(train) + len(validation) + len(test) == len(df)

def test_training_scaler_is_fit_on_training_data_only():
    df = make_synthetic_dataset(40)
    train = df.iloc[:30].copy()
    validation = df.iloc[30:].copy()
    scaler = fit_training_scaler(train)
    train_scaled = transform_features(train, scaler)
    validation_scaled = transform_features(validation, scaler)
    assert np.allclose(train_scaled.mean().to_numpy(), 0.0, atol=1e-6)
    assert np.isfinite(train_scaled.to_numpy()).all()
    assert np.isfinite(validation_scaled.to_numpy()).all()

def test_feature_tokens_have_expected_shape_and_dtype():
    df = make_synthetic_dataset(12)
    scaler = fit_training_scaler(df)
    X_scaled = transform_features(df, scaler)
    tokens = create_feature_tokens(X_scaled)

    assert tokens.shape == (12, 4, 8)
    assert tokens.dtype == np.float32

def test_model_forward_shapes():
    model = HybridAttentionModel()
    dummy_input = torch.randn(8, 4, 8)
    logits, attention = model(dummy_input, return_attention=True)
    assert logits.shape == (8,)
    assert attention.shape == (8, 4, 4)

def test_model_parameter_count():
    model = HybridAttentionModel()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == 9985

def test_model_eval_forward_is_deterministic():
    torch.manual_seed(42)
    model = HybridAttentionModel()
    model.eval()
    dummy_input = torch.randn(8, 4, 8)
    with torch.no_grad():
        first = model(dummy_input)
        second = model(dummy_input)
    assert torch.equal(first, second)

def test_create_feature_groups_assigns_identical_rows_to_same_group():
    df = make_synthetic_dataset(20)
    df.loc[1, FEATURE_COLUMNS] = df.loc[0, FEATURE_COLUMNS]
    groups = create_feature_groups(df, FEATURE_COLUMNS)
    assert groups.iloc[0] == groups.iloc[1]
    assert groups.nunique() == 19
