"""
`continuiti.trainer.scheduler`

Learning rate scheduler for Trainer in continuiti.
"""

import torch
from continuiti.trainer.callbacks import Callback


class LinearLRScheduler(Callback):
    """
    Callback for a linear learning rate scheduler.

    ```
    lr(epoch) = lr0 * (1 - epoch / max_epochs)
    ```

    where `lr0` is the initial learning rate of the optimizer.

    Args:
        optimizer: Optimizer. The learning rate of the first parameter group will be updated.
        max_epochs: Maximum number of epochs.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, max_epochs: int):
        self.optimizer = optimizer
        self.max_epochs = max_epochs

        lr0 = self.optimizer.param_groups[0]["lr"]
        self.schedule = lambda epoch: lr0 * (1 - epoch / max_epochs)

    def __call__(self, logs):
        self.optimizer.param_groups[0]["lr"] = self.schedule(logs.epoch)
