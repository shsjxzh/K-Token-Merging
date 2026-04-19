from __future__ import annotations

import torch
import torch.nn as nn


DEFAULT_HIDDEN_DIM_FACTOR = {
    2: 2.0,
    3: 2.0,
    4: 1.5,
}


class AverageInitializedEncoder(nn.Module):
    """Merge K token embeddings into one latent embedding."""

    def __init__(
        self,
        embedding_dim: int,
        merge_factor: int,
        hidden_dim_factor: dict[int, float] | None = None,
    ) -> None:
        super().__init__()
        hidden_dim_factor = hidden_dim_factor or DEFAULT_HIDDEN_DIM_FACTOR
        if merge_factor not in hidden_dim_factor:
            raise ValueError(f"Unsupported merge factor: {merge_factor}")

        self.embedding_dim = embedding_dim
        self.merge_factor = merge_factor
        expand_factor = hidden_dim_factor[merge_factor]
        hidden_dim = int(expand_factor * merge_factor) * embedding_dim

        self.net = nn.Sequential(
            nn.Linear(merge_factor * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, merge_factor * embedding_dim),
            nn.ReLU(),
            nn.Linear(merge_factor * embedding_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, self.embedding_dim, self.merge_factor)
        x_mean = x_reshaped.mean(dim=-1)
        return x_mean + self.net(x)

