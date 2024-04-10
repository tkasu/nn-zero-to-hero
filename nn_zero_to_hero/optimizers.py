from typing import List, Tuple, Optional

import torch


class StepBasedLrGDOptimizer(torch.optim.Optimizer):
    def __init__(self, params, max_step_to_lr: List[Tuple[Optional[int], float]]):
        defaults = dict(max_epoch_to_lr=max_step_to_lr)
        super(StepBasedLrGDOptimizer, self).__init__(params, defaults)
        self.state = {"step": 0}

    def step(self):
        step = self.state["step"]

        for group in self.param_groups:
            lr = next(
                lr
                for max_epoch, lr in group["max_epoch_to_lr"]
                if max_epoch is None or step < max_epoch
            )

            for p in group["params"]:
                if p.grad is None:
                    raise ValueError("Invalid None gradient")
                p.data.add_(-lr * p.grad.data)

        self.state["step"] += 1
