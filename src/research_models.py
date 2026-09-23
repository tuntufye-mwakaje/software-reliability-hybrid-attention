from __future__ import annotations

import torch
import torch.nn as nn


class FeatureGroupEmbedding(nn.Module):
    """
    Project each semantic feature-group token into a learned embedding space.

    Input:
        (batch, num_tokens, input_dim)

    Output:
        (batch, num_tokens, embedding_dim)
    """

    def __init__(
        self,
        input_dim: int = 8,
        embedding_dim: int = 32,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class MultiHeadFeatureAttention(nn.Module):
    """
    Multi-head self-attention across semantic software-metric groups.
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
        )

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        attention_output, attention_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
        )

        x = self.norm1(
            x + self.dropout(attention_output)
        )

        feed_forward_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(feed_forward_output)
        )

        return x, attention_weights


class HybridAttentionModel(nn.Module):
    """
    Hybrid attention-based software defect prediction model.

    Architecture intentionally matches the historical model in
    src/models.py so that the new multi-dataset experiments remain
    comparable with the original KC1 experiment.

    Input:
        (batch, 4, 8)

    Output:
        logits of shape (batch,)

    The four tokens correspond to:

        1. size
        2. complexity
        3. halstead
        4. operators_operands
    """

    def __init__(
        self,
        input_dim: int = 8,
        embedding_dim: int = 32,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding = FeatureGroupEmbedding(
            input_dim=input_dim,
            embedding_dim=embedding_dim,
        )

        self.feature_attention = MultiHeadFeatureAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ):
        x = self.embedding(x)

        x, attention_weights = self.feature_attention(x)

        x = x.mean(dim=1)

        logits = self.classifier(x).squeeze(-1)

        if return_attention:
            return logits, attention_weights

        return logits


def count_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    model = HybridAttentionModel()

    dummy_input = torch.randn(8, 4, 8)

    logits, attention_weights = model(
        dummy_input,
        return_attention=True,
    )

    print("Research model test")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Logits shape:", logits.shape)
    print("Attention shape:", attention_weights.shape)
    print("Trainable parameters:", count_parameters(model))
