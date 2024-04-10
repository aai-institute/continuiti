"""
`continuiti.trainer.callbacks`

Callbacks for Trainer in continuiti.
"""

from typing import Optional, List
from time import time
import math
import matplotlib.pyplot as plt
from continuiti.operators import Operator
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
        self.steps_performed = 0
        self.start_time = time()
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        elapsed = time() - self.start_time
        sec_per_step = elapsed / self.steps_performed

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

        if sec_per_step < 1:
            s += f"{sec_per_step * 1000:.0f}ms/step"
        else:
            s += f"{sec_per_step:.2f}s/step"

        def to_min(t):
            min = t // 60
            sec = math.floor(t) % 60
            return f"{min:.0f}:{sec:02.0f}min"

        s += f"  [{to_min(elapsed)}"

        if self.epochs is not None and self.steps is not None:
            remaining_steps = (self.epochs - logs.epoch) * self.steps
            remaining_steps += self.steps - logs.step
            eta = remaining_steps * sec_per_step
            s += f"<{to_min(eta)}"
        s += "] - "

        s += f"loss/train = {logs.loss_train:.4e}  "

        if logs.loss_test is not None:
            s += f"loss/test = {logs.loss_test:.4e} "

        print("\r" + s, end="")

    def step(self, logs: Logs):
        """Called after every gradient step.

        Args:
            logs: Training logs.
        """
        self.steps_performed += 1
        self.__call__(logs)

    def on_train_begin(self):
        """Called at the beginning of training."""
        self.start_time = time()

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

    Args:
        operator: Operator to save snapshots. Default is None.
    """

    def __init__(self, operator: Optional[Operator] = None):
        self.operator = operator
        self.best_loss = math.inf
        super().__init__()

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.
        """
        mlflow.log_metric("loss/train", logs.loss_train, step=logs.epoch)
        loss = logs.loss_train
        if logs.loss_test is not None:
            mlflow.log_metric("loss/test", logs.loss_test, step=logs.epoch)
            loss = logs.loss_test

        # Save best model
        self.best_loss = min(self.best_loss, loss)
        if self.best_loss == loss:
            self._save_model("best")

    def on_train_begin(self):
        """Called at the beginning of training."""
        if not mlflow.active_run():
            mlflow.start_run()
        self._save_model("initial")

    def on_train_end(self):
        """Called at the end of training."""
        self._save_model("final")
        mlflow.end_run()

    def _save_model(self, name: str):
        """Save model checkpoint."""
        if self.operator:
            name = f"/tmp/{name}.pt"
            self.operator.save(name)
            mlflow.log_artifact(name)
