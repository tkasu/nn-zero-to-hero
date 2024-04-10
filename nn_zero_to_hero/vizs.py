from typing import Dict

import torch
import matplotlib.pyplot as plt


def plot_embeddings(
    embeddings: torch.Tensor, itos: Dict[int, str], dim_1: int = 0, dim_2: int = 1
):
    # visualize dimensions 0 and 1 of the embedding matrix C for all characters
    C_cpu = embeddings.cpu()

    plt.figure(figsize=(8, 8))
    plt.scatter(C_cpu[:, dim_1].data, C_cpu[:, dim_2].data, s=200)
    for i in range(C_cpu.shape[0]):
        plt.text(
            C_cpu[i, dim_1].item(),
            C_cpu[i, dim_2].item(),
            itos[i],
            ha="center",
            va="center",
            color="white",
        )
    plt.grid("minor")
    return plt
