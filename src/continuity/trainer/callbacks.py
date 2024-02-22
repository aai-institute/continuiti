"""Callbacks for Trainer in Continuity."""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import matplotlib.pyplot as plt


class Callback(ABC):
    """
    Callback base class for `fit` method of `Trainer`.
    """

    @abstractmethod
    def __call__(self, epoch, logs: Dict[str, float]):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
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

    def __call__(self, epoch: int, logs: Dict[str, float]):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
        """
        loss_train = logs["loss/train"]
        seconds_per_epoch = logs["seconds_per_epoch"]

        print(
            f"\rEpoch {epoch}:  loss/train = {loss_train:.4e}  "
            f"({seconds_per_epoch:.3g} s/epoch)",
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
        keys: List of keys to plot. Default is ["loss/train"].
    """

    def __init__(self, keys: Optional[List[str]] = None):
        if keys is None:
            keys = ["loss/train"]

        self.keys = keys
        self.on_train_begin()
        super().__init__()

    def __call__(self, epoch: int, logs: Dict[str, float]):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
        """
        for key in self.keys:
            if key in logs:
                self.losses[key].append(logs[key])

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

    def __call__(self, epoch: int, logs: Dict[str, float]):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
        """
        self.trial.report(logs["loss/train"], step=epoch)

    def on_train_begin(self):
        """Called at the beginning of training."""

    def on_train_end(self):
        """Called at the end of training."""
