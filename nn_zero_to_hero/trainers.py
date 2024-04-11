from typing import List, Tuple, Optional

import optuna
import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm


def train_model_simple(
    model: torch.nn.Module,
    *,
    dataset: Dataset,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    device: Optional[torch.device] = None,
    trial: Optional[optuna.Trial] = None,
) -> Tuple[List[int], List[float]]:
    model.train()

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True
    )

    lossi = []
    stepi = []

    for epoch in tqdm(range(epochs), leave=False):
        epoch_loss_sum = 0.0
        for X_batch, Y_batch in dataloader:

            if device:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)

            logits = model.forward(X_batch)
            loss = F.cross_entropy(logits, Y_batch)
            epoch_loss_sum += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # track stats
        # TODO: Add validation loss
        stepi.append(epoch)
        average_loss = epoch_loss_sum / (epoch + 1)
        lossi.append(average_loss)
        if trial:
            trial.report(average_loss, step=epoch)

    stats_df = pd.DataFrame({"step": stepi, "loss": lossi})
    return stats_df
