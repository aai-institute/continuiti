"""
`continuity.trainer.callbacks`

Callbacks for Trainer in Continuity.
"""

from abc import ABC, abstractmethod
from typing import Optional, List
from time import time
import matplotlib.pyplot as plt
from continuity.operators import Operator
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
    Callback to print training/test loss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        seconds_per_epoch = time() - self.time
        print(
            f"\rEpoch {logs.epoch}:  ",
            f"loss/train = {logs.loss_train:.4e}  ",
            f"loss/test = {logs.loss_test:.4e}  " if logs.loss_test is not None else "",
            f"({seconds_per_epoch:.3f} s/epoch)",
            end="",
        )
        self.time = time()

    def on_train_begin(self):
        """Called at the beginning of training."""
        self.time = time()

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
            keys = ["loss_train", "loss_test"]

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
                test = logs.__getattribute__(key)
                self.losses[key].append(test)
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

    def __init__(self, operator: Optional[Operator] = None):
        self.operator = operator
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        mlflow.log_metric("loss/train", logs.loss_train, step=logs.epoch)
        if logs.loss_test is not None:
            mlflow.log_metric("loss/test", logs.loss_test, step=logs.epoch)

        if logs.epoch % 100 == 0:
            self._save_model(f"ckpt_{logs.epoch:06d}")

    def on_train_begin(self):
        """Called at the beginning of training."""
        if not mlflow.active_run():
            mlflow.start_run()
        self._save_model("ckpt_000000")

    def on_train_end(self):
        """Called at the end of training."""
        self._save_model("operator")
        mlflow.end_run()

    def _save_model(self, name: str):
        """Save model checkpoint."""
        if self.operator:
            name = f"/tmp/{name}.pt"
            self.operator.save(name)
            mlflow.log_artifact(name)
