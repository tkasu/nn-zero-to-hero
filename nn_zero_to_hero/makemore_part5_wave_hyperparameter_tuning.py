from datetime import datetime
import os
import random

import optuna
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

BLOCK_SIZE = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
EPOCHS = 50
LOSS_FUNC = F.cross_entropy
NUM_TRIALS = 50

words = open("data/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
STOI, ITOS = tokens_to_int_mapping(chars)

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_dataset = WordTokensDataset(words[:n1], BLOCK_SIZE, STOI)
validation_dataset = WordTokensDataset(words[n1:n2], BLOCK_SIZE, STOI)
# test_dataset = WordTokensDataset(words[n2:], BLOCK_SIZE, STOI)


def objective(trial: optuna.trial) -> float:

    batch_size = trial.suggest_int("batch_size", 128, 1024, step=32)
    n_embedding_layer_size = trial.suggest_int("embedding_layer_size", 10, 24, step=2)
    n_hidden_layer_size = trial.suggest_int("hidden_layer_size", 64, 256, step=32)
    lr = trial.suggest_float("lr", 0.05, 1, step=0.05)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, pin_memory=True, num_workers=4
    )
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=len(validation_dataset), num_workers=4
    )

    model = build_word_token_wave_model(
        token_count=len(STOI),
        embedding_layer_size=n_embedding_layer_size,
        hidden_layer_size=n_hidden_layer_size,
        lr=lr,
        logging=False,
        optimize=False,
    )

    l_logger = TensorBoardLogger("db_logs", name="makemore_part5_hyperparameter_tuning")

    trainer = L.Trainer(
        max_epochs=EPOCHS,
        accelerator="gpu",
        callbacks=[EarlyStopping(monitor="val_loss", verbose=True, min_delta=0.0001)],
        logger=l_logger,
    )
    trainer.fit(model, train_dataloader, validation_dataloader)

    loss = trainer.callback_metrics["val_loss"].item()
    return loss


db_file_path = os.path.abspath("optuna_db/db.sqlite3")
current_dt_iso = datetime.now().isoformat()

study = optuna.create_study(
    direction="minimize",
    storage=f"sqlite:///{db_file_path}",
    study_name=f"makemore-part5-{current_dt_iso}",
)
study.optimize(objective, n_trials=NUM_TRIALS)

trial = study.best_trial

print(f"Loss: {trial.value}")
print(f"Best hyperparameters: {trial.params}")
# Loss: 1.9978783130645752
# Best hyperparameters: {'batch_size': 448, 'embedding_layer_size': 14, 'hidden_layer_size': 224, 'lr': 0.2}
