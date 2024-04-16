"""
`continuiti.pde.physicsinformed`

PDEs and physics-informed loss functions.
"""

import torch
from abc import abstractmethod

from continuiti.operators.operator import Operator


class PDE:
    r"""PDE base class.

    Example:
        In general, we can implement a PDE like $\nabla v = u$
        as follows:

        ```python
        def pde(x, u, y, v):  # v = op(x, u, y)
            v_y = grad(y, v)
            return mse(v_y, u)

        loss_fn = PhysicsInformedLoss(pde)
        ```
    """

    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Computes PDE loss.

        Usually, we have `v = op(x, u, y)`, e.g., in the physics-informed loss.

        Args:
            x: Tensor of sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Tensor of sensor values of shape (batch_size, u_dim, num_sensors...).
            y: Tensor of evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).
            v: Tensor of _predicted_ values of shape (batch_size, v_dim, num_evaluations...).

        """


class PhysicsInformedLoss:
    """Physics-informed loss function for training operators in continuiti.

    ```python
    loss = pde(x, u, y, op(x, u, y))
    ```

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
            op: Operator object.
            x: Tensor of sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Tensor of sensor values of shape (batch_size, u_dim, num_sensors...).
            y: Tensor of evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).
            v: Ignored.
        """
        # Call operator
        v_pred = op(x, u, y)

        # Get pde loss
        return self.pde(x, u, y, v_pred)
