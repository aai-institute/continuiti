"""
`continuity.operators.integralkernel`

Integral kernel operations.
"""

import torch
from typing import Callable, Union
from continuity.operators import Operator


class NaiveIntegralKernel(Operator):
    r"""Naive integral kernel operator.

    Maps continuous functions via integral kernel application to another
    continuous function and returns point-wise evaluations.

    In mathematical terms, for some given $y$, we obtain
    $$
    v(y) = \int u(x)~\kappa(x, y)~dx
        \approx \frac{1}{N} \sum_{i=1}^{N} u_i~\kappa(x_i, y)
    $$
    where $(x_i, u_i)$ are the $N$ sensors of the mapped observation.

    Note:
        This is a prototype implementation!

        It assumes that x is sampled from a uniform distribution and is not
        efficient for large numbers of sensors.

    Args:
        kernel: Kernel function $\kappa$ or network (if $d$ is the coordinate dimension, $\kappa: \R^d \times \R^d \to \R$)
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
    """

    def __init__(
        self,
        kernel: Union[Callable[[torch.Tensor], torch.Tensor], torch.nn.Module],
        coordinate_dim: int = 1,
        num_channels: int = 1,
    ):
        super().__init__()

        self.kernel = kernel
        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Input function values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, y_size, coordinate_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, y_size, num_channels)
        """
        # Apply the kernel function
        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)
        k = self.kernel(x_expanded, y_expanded)

        # Compute integral
        integral = torch.einsum("bsy,bsc->byc", k, u) / x.size(1)
        return integral
