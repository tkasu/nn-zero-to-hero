import os
import random
import torch
import torch.nn.functional as F
import optuna
from datetime import datetime

from nn_zero_to_hero.datasets import WordTokensDataset
from nn_zero_to_hero.loss import calculate_loss
from nn_zero_to_hero.models import WordTokenModel
from nn_zero_to_hero.optimizers import StepBasedLrGDOptimizer
from nn_zero_to_hero.tokens import tokens_to_int_mapping
from nn_zero_to_hero.trainers import train_model_simple

BLOCK_SIZE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 20
LOSS_FUNC = F.cross_entropy
NUM_TRIALS = 50

words = open("data/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
STOI, _ = tokens_to_int_mapping(chars)

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_dataset = WordTokensDataset(words[:n1], BLOCK_SIZE, STOI)
validation_dataset = WordTokensDataset(words[n1:n2], BLOCK_SIZE, STOI)
test_dataset = WordTokensDataset(words[n2:], BLOCK_SIZE, STOI)


def objective(trial: optuna.Trial) -> float:

    n_embedding_layer_size = trial.suggest_int("embedding_layer_size", 2, 15)
    n_hidden_layer_size = trial.suggest_int("hidden_layer_size", 100, 500, step=50)
    n_lr_start = trial.suggest_float("lr", 1e-2, 1e-1, step=1e-2)
    batch_size = trial.suggest_int("batch_size", 32, 512, step=32)

    batches_by_epoch = len(train_dataset) // batch_size

    model = WordTokenModel(
        token_count=len(STOI),
        block_size=BLOCK_SIZE,
        embedding_layer_size=n_embedding_layer_size,
        hidden_layer_size=n_hidden_layer_size,
        generator=torch.Generator().manual_seed(2147483647),
    ).to(DEVICE)

    optimizer = StepBasedLrGDOptimizer(
        model.parameters(),
        max_step_to_lr=[
            (batches_by_epoch * EPOCHS * 0.5, n_lr_start),
            (batches_by_epoch * EPOCHS * 0.75, n_lr_start * 0.1),
            (None, n_lr_start * 0.01),
        ],
    )

    train_model_simple(
        model=model,
        dataset=train_dataset,
        optimizer=optimizer,
        epochs=EPOCHS,
        batch_size=batch_size,
        trial=trial,
        device=DEVICE,
    )

    validation_loss = calculate_loss(
        model, validation_dataset, LOSS_FUNC, device=DEVICE
    )
    return validation_loss


db_file_path = os.path.abspath("optuna_db/db.sqlite3")
current_dt_iso = datetime.now().isoformat()

study = optuna.create_study(
    direction="minimize",
    storage=f"sqlite:///{db_file_path}",
    study_name=f"makemore-part2-{current_dt_iso}",
)
study.optimize(objective, n_trials=NUM_TRIALS)

trial = study.best_trial

print(f"Loss: {trial.value}")
print(f"Best hyperparameters: {trial.params}")
