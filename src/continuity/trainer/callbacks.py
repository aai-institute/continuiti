"""
`continuity.trainer.callbacks`

Callbacks for Trainer in Continuity.
"""

from typing import Optional, List
from time import time
import math
import matplotlib.pyplot as plt
from continuity.operators import Operator
from .logs import Logs

try:
    import mlflow
except ImportError:
    pass


class Callback:
    """
    Callback base class for `fit` method of `Trainer`.
    """

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """

    def step(self, logs: Logs):
        """Called after every gradient step.

        Args:
            logs: Training logs.
        """

    def on_train_begin(self):
        """Called at the beginning of training."""

    def on_train_end(self):
        """Called at the end of training."""


class PrintTrainingLoss(Callback):
    """
    Callback to print training/test loss.

    Args:
        epochs: Number of epochs. Default is None.
        steps: Number of steps per epoch. Default is None.
    """

    def __init__(self, epochs: Optional[int] = None, steps: Optional[int] = None):
        self.epochs = epochs
        self.steps = steps
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        self.step(logs)
        self.time = time()

    def step(self, logs: Logs):
        """Called after every gradient step.

        Args:
            logs: Training logs.
        """
        ms_per_step = (time() - self.time) / logs.step * 1000

        s = ""
        s += f"Epoch {logs.epoch:2.0f}"
        if self.epochs is not None:
            s += f"/{self.epochs}"
        s += "  "

        s += f"Step {logs.step:2.0f}"
        if self.steps is not None:
            s += f"/{self.steps}"

            # Progress bar
            n = 20
            done = math.ceil(logs.step / self.steps * n)
            s += "  [" + "=" * done + " " * (n - done) + "]"
        s += "  "

        s += f"{ms_per_step:.0f}ms/step"

        if self.epochs is not None and self.steps is not None:
            remaining_steps = (self.epochs - logs.epoch) * self.steps
            remaining_steps += self.steps - logs.step
            eta = remaining_steps * ms_per_step / 1000
            eta_fmt = f"{eta/60:.0f}:{eta%60:02.0f}min"
            s += f"  ETA {eta_fmt}"
        s += " - "

        s += f"loss/train = {logs.loss_train:.4e}  "

        if logs.loss_test is not None:
            s += f"loss/test = {logs.loss_test:.4e} "

        print("\r" + s, end="")

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
