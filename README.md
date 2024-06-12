# Neural Networks from Zero to Hero exercises

My exercises for Andrej Karpathy's lecture series Neural Networks: Zero to Hero:
https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ

## Requirements

* Python 3.11
* poetry

## Installation

`poetry install`

## Formatting

`poetry run black .`

## Clean notebook output before commit

`poetry run nbstripout nn_zero_to_hero/notebooks/*.ipynb`

## Sections

### Makemore part 2: MLP

Inital template based on this video from Andrej: https://www.youtube.com/watch?v=TCH_1BHY58I

#### Refactored notebook

See `nn_zero_to_hero/notebooks/makemore_part2_mlp.ipynb`

#### Hyperparameter tuning

1. Alter the configuration in nn_zero_to_hero/makemore_part2_mlp_hyperparameter_tuning
2. `poetry run python -m nn_zero_to_hero.makemore_part2_mlp_hyperparameter_tuning`

#### Native Pytorch with nn-module

`poetry run python -m nn_zero_to_hero.makemore_part2_mlp_native`

#### Pytorch Lightning version

1. `poetry run python -m nn_zero_to_hero.makemore_part2_mlp_lightning`
2. Inspect with Tensorboard `tensorboard --logdir db_logs`

### Makemore part 3: MLP with Activation and Gradient analysis, with or without batch norm

1. `poetry run python -m nn_zero_to_hero.makemore_part3_mlp_deep_param_logging --use-batch-norm`
2. Inspect with Tensorboard `tensorboard --logdir db_logs`
    1. See `Images` for parameter distribution logging

### Makemore part 5: Wavenet like MLP

1. `poetry run python -m nn_zero_to_hero.nn_zero_to_hero.makemore_part5_wave`
2. Inspect with Tensorboard `tensorboard --logdir db_logs`
