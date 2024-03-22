"""
`continuity.trainer.logs`
"""

from dataclasses import dataclass


@dataclass
class Logs:
    """
    Logs for callbacks and criteria within Trainer in Continuity.

    Attributes:
        epoch: Current epoch.
        loss_train: Training loss.
        loss_test: Test loss.
    """

    epoch: int
    loss_train: float
    loss_test: float
