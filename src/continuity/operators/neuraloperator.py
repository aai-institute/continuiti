"""Neural operator."""

import torch
from typing import Callable, Union
from continuity.operators import Operator
from continuity.operators.common import NeuralNetworkKernel
from continuity.data import DatasetShapes


class ContinuousConvolution(Operator):
    r"""Continuous convolution.

    Maps continuous functions via continuous convolution with a kernel function to another continuous function and
    returns point-wise evaluations.

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


class NeuralOperator(Operator):
    r"""Neural operator architecture

    Maps continuous functions given as observation to another continuous function and returns point-wise evaluations.
    The architecture is a stack of continuous convolutions with a lifting layer and a projection layer.

    *Reference:* N. Kovachki et al. Neural Operator: Learning Maps Between
    Function Spaces With Applications to PDEs. JMLR 24 1-97 (2023)

    For now, sensor positions are equal across all layers.

    Args:
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
        depth: Number of hidden layers
        kernel_width: Width of kernel network
        kernel_depth: Depth of kernel network
    """

    def __init__(
        self,
        dataset_shape: DatasetShapes,
        depth: int = 1,
        kernel_width: int = 32,
        kernel_depth: int = 3,
    ):
        super().__init__()

        self.dataset_shape = dataset_shape

        self.lifting = ContinuousConvolution(
            NeuralNetworkKernel(kernel_width, kernel_depth),
            dataset_shape.x.dim,
            dataset_shape.u.dim,
        )

        self.hidden_layers = torch.nn.ModuleList(
            [
                ContinuousConvolution(
                    NeuralNetworkKernel(kernel_width, kernel_depth),
                    dataset_shape.x.dim,
                    dataset_shape.u.dim,
                )
                for _ in range(depth)
            ]
        )

        self.projection = ContinuousConvolution(
            NeuralNetworkKernel(kernel_width, kernel_depth),
            dataset_shape.x.dim,
            dataset_shape.u.dim,
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, coordinate_dim).
            u: Input function values of shape (batch_size, num_sensors, num_channels).
            y: Coordinates where the mapped function is evaluated of shape (batch_size, y_size, coordinate_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, y_size, num_channels)
        """
        # Lifting layer (we use x as evaluation coordinates for now)
        v = self.lifting(x, u, x)
        assert v.shape[1:] == torch.Size(
            [self.dataset_shape.x.num, self.dataset_shape.u.dim]
        )

        # Hidden layers
        for layer in self.hidden_layers:
            # Layer operation (with residual connection)
            v = layer(x, v, x) + v
            assert v.shape[1:] == torch.Size(
                [self.dataset_shape.x.num, self.dataset_shape.u.dim]
            )

            # Activation
            v = torch.tanh(v)

        # Projection layer
        w = self.projection(x, v, y)
        assert w.shape[1:] == torch.Size([y.size(1), self.dataset_shape.u.dim])
        return w
