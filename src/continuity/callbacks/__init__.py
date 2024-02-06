"""
`continuity.callbacks`

Callbacks for training in Continuity.
"""

import math
from abc import ABC, abstractmethod


class Callback(ABC):
    """
    Callback base class for `fit` method of `Operator`.
    """

    @abstractmethod
    def __call__(self, epoch, logs=None):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
        """
        raise NotImplementedError


class PrintTrainingLoss(Callback):
    """
    Callback to print training loss.
    """

    def __init__(self):
        super().__init__()

    def __call__(self, epoch, logs=None):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
        """
        loss_train = logs["loss/train"]
        iter_per_second = logs["iter_per_second"]

        print(
            f"\rEpoch {epoch}:  loss/train = {loss_train:.4e}  "
            f"({iter_per_second:.2f} it/s)",
            end="",
        )


class LearningCurve(Callback):
    """
    Callback to plot learning curve.
    """

    def __init__(self):
        # Try to import lrcurve
        from lrcurve import PlotLearningCurve

        self.plot = PlotLearningCurve(
            line_config={
                "train": {"name": "Train", "color": "#000000"},
            },
            facet_config={"loss": {"name": "log(loss)", "limit": [None, None]}},
            xaxis_config={"name": "Epoch", "limit": [0, None]},
        )

        super().__init__()

    def __call__(self, epoch, logs=None):
        """Callback function.
        Called at the end of each epoch.

        Args:
            epoch: Current epoch.
            logs: Dictionary of logs.
        """
        vals = {"loss": {}}

        # Collect loss values
        for key in ["train", "val"]:
            loss_key = "loss/" + key
            if loss_key in logs:
                log_loss = math.log(logs[loss_key], 10)
                vals["loss"][key] = log_loss

        self.plot.append(epoch, vals)
        self.plot.draw()
