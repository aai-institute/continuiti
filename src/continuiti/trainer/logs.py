"""
`continuiti.trainer.logs`
"""

from dataclasses import dataclass


@dataclass
class Logs:
    """
    Logs for callbacks and criteria within Trainer in continuiti.

    Attributes:
        epoch: Current epoch.
        step: Current step.
        loss_train: Training loss.
        loss_test: Test loss.
    """

    epoch: int
    step: int
    loss_train: float
    loss_test: float
