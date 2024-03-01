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
        time: Time taken for epoch.
    """

    epoch: int
    loss_train: float
    seconds_per_epoch: float
