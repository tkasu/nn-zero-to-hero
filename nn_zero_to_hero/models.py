from typing import Optional

import torch
import torch.nn as nn


class WordTokenModel(nn.Module):
    def __init__(
        self,
        token_count: int,
        block_size: int,
        embedding_layer_size: int,
        hidden_layer_size: int,
        generator: Optional[torch.Generator],
    ):
        super().__init__()

        if not generator:
            generator = torch.Generator()

        self.token_count = token_count
        self.block_size = block_size
        self.embedding_layer_size = embedding_layer_size
        self.hidden_layer_size = hidden_layer_size

        self.C = nn.Parameter(
            torch.randn((token_count, embedding_layer_size), generator=generator)
        )
        self.W1 = nn.Parameter(
            torch.randn(
                (embedding_layer_size * block_size, hidden_layer_size),
                generator=generator,
            )
        )
        self.b1 = nn.Parameter(torch.randn(hidden_layer_size, generator=generator))
        self.W2 = nn.Parameter(
            torch.randn((hidden_layer_size, token_count), generator=generator)
        )
        self.b2 = nn.Parameter(torch.randn(token_count, generator=generator))

    def forward(self, X: torch.Tensor):
        emb = self.C[X]  # (BATCH_SIZE, block_size, embedding_layer_size)
        h = torch.tanh(
            emb.view(-1, self.embedding_layer_size * self.block_size) @ self.W1
            + self.b1
        )  # (BATCH_SIZE, hidden_layer_size)
        logits = h @ self.W2 + self.b2  # (BATCH_SIZE, token_count)
        return logits
