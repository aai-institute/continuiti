"""
`continuity.trainer.callbacks`

Callbacks for Trainer in Continuity.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import matplotlib.pyplot as plt
from .logs import Logs


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
    Callback to print training loss.
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
            f"\rEpoch {logs.epoch}:  "
            f"loss/train = {logs.loss_train:.4e}  "
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
            keys = ["loss_train"]

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
