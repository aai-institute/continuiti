"""
`continuiti.operators.losses`

Loss functions for operator learning.

Every loss function takes an operator `op`, sensor positions `x`, sensor values `u`,
evaluation coordinates `y`, and labels `v` as input and returns a scalar loss:

```python
loss = loss_fn(op, x, u, y, v)
```
"""

import torch
from abc import abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from continuiti.operators.operator import Operator


class Loss:
    """Loss function for training operators in continuiti."""

    @abstractmethod
    def __call__(
        self,
        op: "Operator",
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate loss.

        Args:
            op: Operator object.
            x: Tensor of sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Tensor of sensor values of shape (batch_size, u_dim, num_sensors...).
            y: Tensor of evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).
            v: Tensor of labels of shape (batch_size, v_dim, num_evaluations...).
        """


class MSELoss(Loss):
    """Computes the mean-squared error between the predicted and true labels.

    ```python
    loss = mse(op(x, u, y), v)
    ```
    """

    def __init__(self):
        self.mse = torch.nn.MSELoss()

    def __call__(
        self,
        op: "Operator",
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate MSE loss.

        Args:
            op: Operator object.
            x: Tensor of sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Tensor of sensor values of shape (batch_size, u_dim, num_sensors...).
            y: Tensor of evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).
            v: Tensor of labels of shape (batch_size, v_dim, num_evaluations...).
        """
        # Call operator
        v_pred = op(x, u, y)

        # Align shapes
        v_pred = v_pred.reshape(v.shape)

        # Return MSE
        return self.mse(v_pred, v)


class RelativeL1Error(Loss):
    """Computes the relative L1 error between the predicted and true labels.

    ```python
    loss = l1(v, op(x, u, y)) / l1(v, 0)
    ```
    """

    def __init__(self):
        self.l1 = torch.nn.L1Loss()

    def __call__(
        self,
        op: "Operator",
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate relative L1 error.

        Args:
            op: Operator object.
            x: Tensor of sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Tensor of sensor values of shape (batch_size, u_dim, num_sensors...).
            y: Tensor of evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).
            v: Tensor of labels of shape (batch_size, v_dim, num_evaluations...).
        """
        # Call operator
        v_pred = op(x, u, y)

        # Align shapes
        v_pred = v_pred.reshape(v.shape)

        # Return relative L1 error
        return self.l1(v, v_pred) / self.l1(v, torch.zeros_like(v))
