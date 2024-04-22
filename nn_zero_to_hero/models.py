import itertools
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import matplotlib.pyplot as plt
from schedulefree import SGDScheduleFree, AdamWScheduleFree


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
        hidden_layer_count: int = 1,
        lr: float = 0.1,
        optimize: bool = True,
    ):
        super().__init__()
        assert hidden_layer_count > 0, "hidden_layer_count should be greater than 0"
        self.lr = lr
        self.prev_epoch = None
        self.update_to_data_ratios = defaultdict(list)

        extra_hidden_layers = list(
            itertools.chain(
                *[
                    (nn.Linear(hidden_layer_size, hidden_layer_size), nn.Tanh())
                    for _ in range(1, hidden_layer_count)
                ]
            )
        )

        model = nn.Sequential(
            nn.Embedding(
                embedding_dim=embedding_layer_size, num_embeddings=token_count
            ),
            nn.Flatten(),
            nn.Linear(embedding_layer_size * block_size, hidden_layer_size),
            nn.Tanh(),
            *extra_hidden_layers,
            nn.Linear(hidden_layer_size, token_count),
        )
        self.optimized_model = optimize
        if self.optimized_model:
            model = torch.compile(model, mode="reduce-overhead")
        self.model = model
        self.loss_func = F.cross_entropy

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        X, Y = batch
        logits = self.forward(X)
        loss = self.loss_func(logits, Y)
        self.log("train_loss", loss)
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
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        # optimizer = SGDScheduleFree(self.model.parameters(), lr=self.lr)
        return optimizer

    def _save_param_dist_plot(self):
        legends = []
        fig = plt.figure()
        for name, params in self.model.named_parameters():
            if "weigh" in name:
                legends.append(f"weights - {name}")
                plt.hist(params.view(-1).cpu().tolist(), 50, histtype="step")
        plt.legend(legends, loc="upper left", fontsize="x-small")
        plt.title("Weights distribution")
        self.logger.experiment.add_figure(
            "weights_dist", fig, global_step=self.global_step
        )

    def _save_grad_dist_plot(self):
        legends = []
        fig = plt.figure()
        for name, params in self.model.named_parameters():
            if "weigh" in name:
                legends.append(f"grads - {name}")
                plt.hist(params.grad.view(-1).cpu().tolist(), 50, histtype="step")
        plt.legend(legends, loc="upper left", fontsize="x-small")
        plt.title("Grad distribution")
        self.logger.experiment.add_figure(
            "grad_dist", fig, global_step=self.global_step
        )

    def _save_update_data_ratio_plot(self):
        legends = []
        fig = plt.figure()
        for name, ratios in self.update_to_data_ratios.items():
            legends.append(f"ratio - {name}")
            plt.plot(ratios)
        plt.legend(legends, loc="lower right", fontsize="x-small")
        self.logger.experiment.add_figure(
            "update_to_data_ratio_by_step", fig, global_step=self.global_step
        )

    def _update_ud_ratio(self):
        for name, params in self.model.named_parameters():
            if "weigh" in name:
                self.update_to_data_ratios[name].append(
                    ((params.grad * self.lr).std() / params.data.std()).log10().item()
                )

    def on_before_optimizer_step(self, _optimizer):
        self._update_ud_ratio()

        if self.current_epoch != self.prev_epoch:
            self.prev_epoch = self.current_epoch
            if self.current_epoch % 10 == 0:
                self._save_grad_dist_plot()

    def on_train_epoch_end(self) -> None:
        if self.current_epoch % 10 == 0:
            self._save_param_dist_plot()

    def on_train_end(self) -> None:
        self._save_update_data_ratio_plot()
