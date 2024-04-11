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
    pin_memory: bool = False,
    prefetch_factor: int = 10,
) -> Tuple[List[int], List[float]]:
    model.train()

    dataloader_batch_size = batch_size * prefetch_factor
    dataloader = DataLoader(
        dataset, batch_size=dataloader_batch_size, shuffle=True, pin_memory=pin_memory
    )

    lossi = []
    stepi = []

    for epoch in tqdm(range(epochs), leave=False):
        epoch_loss_sum = 0.0
        epoch_batch_count = 0

        # Load data in bigger chunks to speed up training
        for X_batch_dl, Y_batch_dl in dataloader:
            if device:
                X_batch_dl = X_batch_dl.to(device, non_blocking=pin_memory)
                Y_batch_dl = Y_batch_dl.to(device, non_blocking=pin_memory)

            X_batches = torch.split(X_batch_dl, batch_size)
            Y_batches = torch.split(Y_batch_dl, batch_size)

            for X_batch, Y_batch in zip(X_batches, Y_batches):
                epoch_batch_count += 1

                logits = model.forward(X_batch)
                loss = F.cross_entropy(logits, Y_batch)
                epoch_loss_sum += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # track stats
        # TODO: Add validation loss
        stepi.append(epoch)
        average_loss = epoch_loss_sum / epoch_batch_count
        lossi.append(average_loss)
        if trial:
            trial.report(average_loss, step=epoch)
        print(f"Epoch {epoch} - Average training loss: {average_loss:.4f}")

    stats_df = pd.DataFrame({"step": stepi, "loss": lossi})
    return stats_df
