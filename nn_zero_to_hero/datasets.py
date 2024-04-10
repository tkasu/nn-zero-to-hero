from typing import Dict, List

import torch
from torch.utils.data import Dataset

from nn_zero_to_hero.tokens import tokens_to_int_mapping


class WordTokensDataset(Dataset):
    def __init__(self, words: List[str], block_size: int, stoi: Dict[str, int]):
        X, Y = [], []

        for w in words:

            context = [0] * block_size
            for ch in w + ".":
                ix = stoi[ch]
                X.append(context)
                Y.append(ix)
                context = context[1:] + [ix]  # crop and append

        X = torch.tensor(X)
        Y = torch.tensor(Y)

        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self.X.shape[0]
