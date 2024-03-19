"""
`continuity.trainer.callbacks`

Callbacks for Trainer in Continuity.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import matplotlib.pyplot as plt
import torch
from .logs import Logs

try:
    import mlflow
except ImportError:
    pass


class Callback(ABC):
    """
    Callback base class for `fit` method of `Trainer`.
    """

    @abstractmethod
    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        raise NotImplementedError

    @abstractmethod
    def on_train_begin(self):
        """Called at the beginning of training."""

    @abstractmethod
    def on_train_end(self):
        """Called at the end of training."""


class PrintTrainingLoss(Callback):
    """
    Callback to print training/validation loss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        print(
            f"\rEpoch {logs.epoch}:  ",
            f"loss/train = {logs.loss_train:.4e}  ",
            f"loss/val = {logs.loss_val:.4e}  " if logs.loss_val is not None else "",
            f"({logs.seconds_per_epoch:.3f} s/epoch)",
            end="",
        )

    def on_train_begin(self):
        """Called at the beginning of training."""

    def on_train_end(self):
        """Called at the end of training."""
        print("")


class LearningCurve(Callback):
    """
    Callback to plot learning curve.

    Args:
        keys: List of keys to plot. Default is ["loss_train"].
    """

    def __init__(self, keys: Optional[List[str]] = None):
        if keys is None:
            keys = ["loss_train", "loss_val"]

        self.keys = keys
        self.on_train_begin()
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        for key in self.keys:
            try:
                val = logs.__getattribute__(key)
                self.losses[key].append(val)
            except AttributeError:
                pass

    def on_train_begin(self):
        """Called at the beginning of training."""
        self.losses = {key: [] for key in self.keys}

    def on_train_end(self):
        """Called at the end of training."""
        for key in self.keys:
            vals = self.losses[key]
            epochs = list(range(1, len(vals) + 1))
            plt.plot(epochs, vals)

        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend(self.keys)
        plt.show()


class OptunaCallback(Callback):
    """
    Callback to report intermediate values to Optuna.

    Args:
        trial: Optuna trial.
    """

    def __init__(self, trial):
        self.trial = trial
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        self.trial.report(logs.loss_train, step=logs.epoch)

    def on_train_begin(self):
        """Called at the beginning of training."""

    def on_train_end(self):
        """Called at the end of training."""


class MLFlowLogger(Callback):
    """
    Callback to log metrics to MLFlow.
    """

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        mlflow.log_metric("loss/train", logs.loss_train, step=logs.epoch)
        mlflow.log_metric("loss/val", logs.loss_val, step=logs.epoch)
        mlflow.log_metric("seconds_per_epoch", logs.seconds_per_epoch, step=logs.epoch)

    def on_train_begin(self):
        """Called at the beginning of training."""
        mlflow.start_run()

    def on_train_end(self):
        """Called at the end of training."""
        mlflow.end_run()


class LossHistory(Callback):
    """
    Callback that stores the loss history.
    """

    def __call__(self, logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        self.train_history.append(logs.loss_train)
        if logs.loss_val is not None:
            self.val_history.append(logs.loss_val)

    def on_train_begin(self):
        """Called at the beginning of training."""
        self.train_history = []
        self.val_history = []

    def on_train_end(self):
        """Called at the end of training."""


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

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass
