"""Loss functions for physics-informed training."""

from torch import Tensor
from abc import abstractmethod

from continuity.operators.operator import Operator


class PDE:
    """PDE base class."""

    @abstractmethod
    def __call__(
        self, op: Operator, x: Tensor, u: Tensor, y: Tensor, v: Tensor
    ) -> Tensor:
        """Computes PDE loss."""


class PhysicsInformedLoss:
    """Physics-informed loss function for training operators in Continuity.

    Args:
        pde: Maps evaluation coordinates $y$ and callable $v$ to PDE loss.
    """

    def __init__(self, pde: PDE):
        self.pde = pde

    def __call__(
        self, op: Operator, x: Tensor, u: Tensor, y: Tensor, v: Tensor
    ) -> Tensor:
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
