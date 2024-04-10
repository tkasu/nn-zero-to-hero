import torch

from nn_zero_to_hero.datasets import WordTokensDataset
from nn_zero_to_hero.models import WordTokenModel


def calculate_loss(
    model: WordTokenModel, dataset: WordTokensDataset, loss_func, device: torch.device
):
    model.eval()
    with torch.no_grad():
        X, Y = dataset.X, dataset.Y
        X = X.to(device)
        Y = Y.to(device)
        logits = model.forward(X)
        loss = loss_func(logits, Y)
    return loss.item()
