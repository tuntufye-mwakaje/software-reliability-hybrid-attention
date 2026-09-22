# Models

This directory contains trained model artifacts produced by the software reliability prediction pipeline.

## Hybrid Attention Model

The primary model is a hybrid attention-based neural network designed for software defect prediction using the NASA/PROMISE KC1 dataset.

The model represents software metrics as four semantic feature-group tokens:

| Feature Group | Metrics |
|---|---|
| Size | `loc`, `lOCode`, `lOComment`, `lOBlank`, `locCodeAndComment` |
| Complexity | `v(g)`, `ev(g)`, `iv(g)`, `branchCount` |
| Halstead | `n`, `v`, `l`, `d`, `i`, `e`, `b`, `t` |
| Operators / Operands | `uniq_Op`, `uniq_Opnd`, `total_Op`, `total_Opnd` |

Each token is projected from an 8-dimensional representation into a 32-dimensional learned embedding.

The embedded feature groups are then processed using multi-head self-attention with:

- Embedding dimension: `32`
- Attention heads: `4`
- Dropout: `0.1`
- Feed-forward expansion: `2x`
- Layer normalization
- GELU activation

The resulting token representations are mean-pooled and passed through a small classification head that produces a single defect-prediction logit.

## Model Configuration

The trained experiment uses:

```text
Input dimension:       8
Number of tokens:      4
Embedding dimension:   32
Attention heads:       4
Dropout:               0.1
Trainable parameters:  9,985