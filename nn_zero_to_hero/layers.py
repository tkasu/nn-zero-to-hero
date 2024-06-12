import torch
import torch.nn as nn


class FlattenConsecutive(nn.Module):

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, X: torch.Tensor):
        B, T, C = X.shape
        X = X.reshape(B, T // self.n, C * self.n)
        if X.shape[1] == 1:
            X = X.squeeze(1)
        self.out = X
        return self.out


class Permute(nn.Module):

    def __init__(self, dims: tuple):
        super().__init__()
        self.dims = dims

    def forward(self, X: torch.Tensor):
        self.out = X.permute(*self.dims)
        return self.out


class ShapePrinter(nn.Module):

    def __init__(self, id: str = None):
        self.id = id
        super().__init__()

    def forward(self, X: torch.Tensor):
        print(f"ID: {self.id} shape: {X.shape}")
        return X
