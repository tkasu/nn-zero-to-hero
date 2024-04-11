from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from schedulefree import SGDScheduleFree


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


class WordTokenModelL(L.LightningModule):
    def __init__(
        self,
        token_count: int,
        block_size: int,
        embedding_layer_size: int,
        hidden_layer_size: int,
    ):
        super().__init__()
        model = nn.Sequential(
            nn.Embedding(
                embedding_dim=embedding_layer_size, num_embeddings=token_count
            ),
            nn.Flatten(),
            nn.Linear(embedding_layer_size * block_size, hidden_layer_size),
            nn.Tanh(),
            nn.Linear(hidden_layer_size, token_count),
        )
        self.model = torch.compile(model, mode="reduce-overhead")
        self.loss_func = F.cross_entropy

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_func(logits, Y)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_func(logits, Y)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_func(logits, Y)
        self.log("test_loss", loss)
        return loss

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.model(batch)

    def configure_optimizers(self):
        optimizer = SGDScheduleFree(self.model.parameters(), lr=1.0)
        return optimizer
