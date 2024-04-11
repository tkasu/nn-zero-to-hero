import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_zero_to_hero.datasets import WordTokensDataset
from nn_zero_to_hero.loss import calculate_loss
from nn_zero_to_hero.tokens import sample_from_model, tokens_to_int_mapping
from nn_zero_to_hero.trainers import train_model_simple

torch.set_float32_matmul_precision("high")  # Use TensorFloat32
torch.backends.cuda.matmul.allow_tf32 = True

BATCH_SIZE = 128
BLOCK_SIZE = 3
LR = 0.1
MOMENTUM = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")
EPOCHS = 50
LOSS_FUNC = F.cross_entropy

words = open("data/names.txt", "r").read().splitlines()
chars = sorted(list(set("".join(words))))
STOI, ITOS = tokens_to_int_mapping(chars)

random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))

train_dataset = WordTokensDataset(words[:n1], BLOCK_SIZE, STOI)
validation_dataset = WordTokensDataset(words[n1:n2], BLOCK_SIZE, STOI)
test_dataset = WordTokensDataset(words[n2:], BLOCK_SIZE, STOI)


def get_model(
    token_count: int,
    block_size: int,
    embedding_layer_size: int,
    hidden_layer_size: int,
) -> nn.Module:
    return nn.Sequential(
        nn.Embedding(embedding_dim=embedding_layer_size, num_embeddings=token_count),
        nn.Flatten(),
        nn.Linear(embedding_layer_size * block_size, hidden_layer_size),
        nn.Tanh(),
        nn.Linear(hidden_layer_size, token_count),
    )


model = torch.compile(
    get_model(
        token_count=len(STOI),
        block_size=BLOCK_SIZE,
        embedding_layer_size=5,
        hidden_layer_size=100,
    ).to(DEVICE)
)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

stats_df = train_model_simple(
    model=model,
    dataset=train_dataset,
    optimizer=optimizer,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    device=DEVICE,
    prefetch_factor=10,
)

print("Training losses")
print(stats_df["loss"])

training_loss = calculate_loss(model, train_dataset, LOSS_FUNC, DEVICE)
validation_loss = calculate_loss(model, validation_dataset, LOSS_FUNC, DEVICE)
print(f"{training_loss = :4f}, {validation_loss = :4f}")

print("Samples from the model:")
g = torch.Generator(DEVICE).manual_seed(2147483647 + 10)
for _ in range(10):
    s = sample_from_model(
        model,
        block_size=BLOCK_SIZE,
        device=DEVICE,
        itos=ITOS,
        generator=g,
    )
    print(s)
