"""
`continuity.pde`

PDEs in Continuity.

Every PDE is implemented using a physics-informed loss function.
"""

import torch
from abc import abstractmethod

from continuity.operators.operator import Operator
from .grad import Grad, grad, Div, div

__all__ = [
    "PDE",
    "PhysicsInformedLoss",
    "Grad",
    "Div",
    "grad",
    "div",
]


class PDE:
    """PDE base class."""

    @abstractmethod
    def __call__(
        self,
        op: Operator,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Computes PDE loss."""


class PhysicsInformedLoss:
    """Physics-informed loss function for training operators in Continuity.

    Args:
        pde: Maps evaluation coordinates $y$ and callable $v$ to PDE loss.
    """

    def __init__(self, pde: PDE):
        self.pde = pde

    def __call__(
        self,
        op: Operator,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        _: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate loss.

        Args:
            op: Operator object
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of evaluation coordinates of shape (batch_size, x_size, coordinate_dim)
            v: Ignored
        """
        # Call operator
        v_pred = op(x, u, y)

        # Get pde loss
        return self.pde(x, u, y, v_pred)
