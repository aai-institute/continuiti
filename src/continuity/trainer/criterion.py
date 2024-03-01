"""
`continuity.trainer.criterion`

Stopping criterion for Trainer in Continuity.
"""

from abc import ABC, abstractmethod
from .logs import Logs


class Criterion(ABC):
    """
    Stopping criterion base class for `fit` method of `Trainer`.
    """

    @abstractmethod
    def __call__(self, logs: Logs):
        """Evaluate stopping criterion.
        Called at the end of each epoch.

        Args:
            logs: Training logs.

        Returns:
            bool: Whether to stop training.
        """
        raise NotImplementedError


class TrainingLossCriterion(Criterion):
    """
    Stopping criterion based on training loss.
    """

    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, logs: Logs):
        """Callback function.
        Called at the end of each epoch.

        Args:
            logs: Training logs.

        Returns:
            bool: True if training loss is below threshold.
        """
        return logs.loss_train < self.threshold