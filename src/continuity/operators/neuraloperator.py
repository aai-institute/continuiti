"""Neural operator."""

import torch
from typing import Callable, Union
from torch import Tensor
from continuity.operators import Operator
from continuity.operators.common import NeuralNetworkKernel


class ContinuousConvolution(Operator):
    r"""Continuous convolution.

    Maps continuous functions via continuous convolution with a kernel function
    to another continuous functions and returns point-wise evaluations.

    In mathematical terms, for some given $y$, we obtain
    $$
    v(y) = \int u(x)~\kappa(x, y)~dx
        \approx \frac{1}{N} \sum_{i=1}^{N} u_i~\kappa(x_i, y)
    $$
    where $(x_i, u_i)$ are the $N$ sensors of the mapped observation.

    Args:
        kernel: Kernel function $\kappa$ or network (if $d$ is the coordinate dimension, $\kappa: \R^d \times \R^d \to \R$)
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
    """

    def __init__(
        self,
        kernel: Union[Callable[[Tensor], Tensor], torch.nn.Module],
        coordinate_dim: int = 1,
        num_channels: int = 1,
    ):
        super().__init__()

        self.kernel = kernel
        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels

    def forward(self, x: Tensor, u: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, y_size, coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, num_channels)
        """
        # Unsqueeze if no batch dim
        if len(u.shape) < 3:
            batch_size = 1
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            y = y.unsqueeze(0)

        # Patch for 1D coordinates and one channel
        if len(x.shape) < 3 and self.coordinate_dim == 1:
            x = x.unsqueeze(-1)

        if len(u.shape) < 3 and self.num_channels == 1:
            u = u.unsqueeze(-1)

        if len(y.shape) < 3 and self.coordinate_dim == 1:
            y = y.unsqueeze(-1)

        # Get batch size etc.
        batch_size = u.shape[0]
        num_sensors = u.shape[1]
        y_size = y.shape[1]

        # Check shapes
        assert x.shape[0] == batch_size
        assert y.shape[0] == batch_size
        assert x.shape[1] == num_sensors

        # Apply the kernel function
        x_expanded = x.unsqueeze(2)
        y_expanded = y.unsqueeze(1)
        k = self.kernel(x_expanded, y_expanded)
        assert k.shape == (batch_size, num_sensors, y_size)

        # Compute integral
        integral = torch.einsum("bsy,bsc->byc", k, u) / num_sensors
        assert integral.shape == (batch_size, y_size, self.num_channels)
        return integral


class NeuralOperator(Operator):
    r"""Neural operator architecture

    Maps continuous functions given as observation to another continuous
    functions and returns point-wise evaluations. The architecture is a
    stack of continuous convolutions with a lifting layer and a projection
    layer.

    *Reference:* N. Kovachki et al. Neural Operator: Learning Maps Between
    Function Spaces With Applications to PDEs. JMLR 24 1-97 (2023)

    For now, sensors positions are equal across all layers.

    Args:
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
        depth: Number of hidden layers
        kernel_width: Width of kernel network
        kernel_depth: Depth of kernel network
    """

    def __init__(
        self,
        coordinate_dim: int = 1,
        num_channels: int = 1,
        depth: int = 1,
        kernel_width: int = 32,
        kernel_depth: int = 3,
    ):
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels

        self.lifting = ContinuousConvolution(
            NeuralNetworkKernel(kernel_width, kernel_depth),
            coordinate_dim,
            num_channels,
        )

        self.hidden_layers = torch.nn.ModuleList(
            [
                ContinuousConvolution(
                    NeuralNetworkKernel(kernel_width, kernel_depth),
                    coordinate_dim,
                    num_channels,
                )
                for _ in range(depth)
            ]
        )

        self.projection = ContinuousConvolution(
            NeuralNetworkKernel(kernel_width, kernel_depth),
            coordinate_dim,
            num_channels,
        )

    def forward(self, x: Tensor, u: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, coordinate_dim).
            u: Tensor of sensor values of shape (batch_size, num_sensors, num_channels).
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, y_size, coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, num_channels)
        """
        # Unsqueeze if no batch dim
        if len(u.shape) < 3:
            batch_size = 1
            x = x.unsqueeze(0)
            u = u.unsqueeze(0)
            y = y.unsqueeze(0)

        # Patch for 1D coordinates and one channel
        if len(x.shape) < 3 and self.coordinate_dim == 1:
            x = x.unsqueeze(-1)

        if len(u.shape) < 3 and self.num_channels == 1:
            u = u.unsqueeze(-1)

        if len(y.shape) < 3 and self.coordinate_dim == 1:
            y = y.unsqueeze(-1)

        # Get batch size etc.
        batch_size = u.shape[0]
        num_sensors = u.shape[1]
        y_size = y.shape[1]

        # Check shapes
        assert x.shape[0] == batch_size
        assert y.shape[0] == batch_size
        assert x.shape[1] == num_sensors

        # Lifting layer (we use x as evaluation coordinates for now)
        v = self.lifting(x, u, x)
        assert v.shape == (batch_size, num_sensors, self.num_channels)

        # Hidden layers
        for layer in self.hidden_layers:
            # Layer operation (with residual connection)
            v = layer(x, v, x) + v
            assert v.shape == (batch_size, num_sensors, self.num_channels)

            # Activation
            v = torch.tanh(v)

        # Projection layer
        w = self.projection(x, v, y)
        assert w.shape == (batch_size, y_size, self.num_channels)
        return w
