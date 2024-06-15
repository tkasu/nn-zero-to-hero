import random

import click
import torch
import torch.nn.functional as F
import lightning as L
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from nn_zero_to_hero.datasets import WordTokensDataset
from nn_zero_to_hero.models import build_word_token_wave_model
from nn_zero_to_hero.tokens import sample_from_model, tokens_to_int_mapping

torch.set_float32_matmul_precision("high")  # Use TensorFloat32
torch.backends.cuda.matmul.allow_tf32 = True

BATCH_SIZE = 448
BLOCK_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
EPOCHS = 100
LOSS_FUNC = F.cross_entropy


@click.command()
def train():
    words = open("data/names.txt", "r").read().splitlines()
    chars = sorted(list(set("".join(words))))
    STOI, ITOS = tokens_to_int_mapping(chars)

    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    train_dataset = WordTokensDataset(words[:n1], BLOCK_SIZE, STOI)
    validation_dataset = WordTokensDataset(words[n1:n2], BLOCK_SIZE, STOI)
    test_dataset = WordTokensDataset(words[n2:], BLOCK_SIZE, STOI)

    train_dataloader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=4
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), num_workers=4
    )
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset))

    model = build_word_token_wave_model(
        token_count=len(STOI),
        embedding_layer_size=14,
        hidden_layer_size=224,
        lr=0.2,
    )
    print(model)

    l_logger = TensorBoardLogger("db_logs", name="makemore_part5")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss", verbose=True, min_delta=0.00001)],
        logger=l_logger,
    )
    trainer.fit(model, train_dataloader, validation_dataloader)

    trainer.test(model, dataloaders=test_dataloader)

    print("Samples from the model:")
    g = torch.Generator().manual_seed(2147483647 + 10)
    for _ in range(10):
        s = sample_from_model(
            model.cpu(),
            block_size=BLOCK_SIZE,
            itos=ITOS,
            device=torch.device("cpu"),
            generator=g,
        )
        print(s)


if __name__ == "__main__":
    train()
