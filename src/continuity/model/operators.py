"""Various operator (and supporting network) implementations."""

import torch
from typing import Callable
from torch import Tensor
from continuity.model import device, Operator


class ResidualLayer(torch.nn.Module):
    """Residual layer.

    Args:
        width: Width of the layer.
    """

    def __init__(self, width: int):
        super().__init__()
        self.layer = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x: Tensor):
        """Forward pass."""
        return self.act(self.layer(x)) + x


class DeepResidualNetwork(torch.nn.Module):
    """Deep residual network.

    Args:
        input_size: Size of input tensor
        output_size: Size of output tensor
        width: Width of hidden layers
        depth: Number of hidden layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
    ):
        super().__init__()

        self.first_layer = torch.nn.Linear(input_size, width)
        self.hidden_layers = torch.nn.ModuleList(
            [ResidualLayer(width) for _ in range(depth)]
        )
        self.last_layer = torch.nn.Linear(width, output_size)

    def forward(self, x):
        """Forward pass."""
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)


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
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
        kernel: Kernel function $\kappa$ or network (if $d$ is the coordinate dimension, $\kappa: \R^d \times \R^d \to \R$)
    """

    def __init__(
        self,
        coordinate_dim: int,
        num_channels: int,
        kernel: Callable[[Tensor], Tensor] | torch.nn.Module,
    ):
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.kernel = kernel

    def forward(self, xu: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            xu: Tensor of observations of shape (batch_size, num_sensors, coordinate_dim + num_channels)
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, x_size, coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, x_size, num_channels)
        """
        batch_size = xu.shape[0]
        num_sensors = xu.shape[1]
        y_size = y.shape[1]

        # Check shapes
        assert y.shape[0] == batch_size
        assert y.shape[2] == self.coordinate_dim
        assert xu.shape[2] == self.coordinate_dim + self.num_channels

        # Sensors are (x, u)
        x = xu[:, :, : self.coordinate_dim]
        u = xu[:, :, -self.num_channels :]
        assert x.shape == (batch_size, num_sensors, self.coordinate_dim)
        assert u.shape == (batch_size, num_sensors, self.num_channels)

        # Compute radial coordinates
        assert y.shape == (batch_size, y_size, self.coordinate_dim)

        # Kernel operation (TODO: use tensor operation)
        k = torch.tensor(
            [
                [
                    [self.kernel(x[b][s], y[b][j]) for j in range(y_size)]
                    for s in range(num_sensors)
                ]
                for b in range(batch_size)
            ]
        )
        assert k.shape == (batch_size, num_sensors, y_size)

        # Compute integral
        integral = torch.einsum("bsy,bsc->byc", k, u) / num_sensors
        assert integral.shape == (batch_size, y_size, self.num_channels)
        return integral


class FullyConnected(Operator):
    """Fully connected architecture."""

    def __init__(
        self,
        coordinate_dim: int,
        num_channels: int,
        num_sensors: int,
        width: int,
        depth: int,
    ):
        """Maps observations and positions to evaluations.

        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            width: Width of network
            depth: Depth of network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.width = width
        self.depth = depth

        self.input_size = num_sensors * (coordinate_dim + num_channels) + coordinate_dim
        output_size = num_channels
        self.drn = DeepResidualNetwork(
            self.input_size,
            output_size,
            self.width,
            self.depth,
        )

    def forward(self, u, x):
        """Forward pass."""
        batch_size = u.shape[0]
        assert batch_size == x.shape[0]
        num_positions = x.shape[1]
        d = self.coordinate_dim

        ux = torch.empty(
            (batch_size, num_positions, self.input_size),
            device=device,
        )

        for i, u_tensor in enumerate(u):
            ux[i, :, :-d] = u_tensor.flatten()
        ux[:, :, -d:] = x

        ux = ux.reshape((batch_size * num_positions, -1))
        v = self.drn(ux)
        v = v.reshape((batch_size, num_positions, self.num_channels))
        return v


class DeepONet(Operator):
    """DeepONet architecture."""

    def __init__(
        self,
        coordinate_dim: int,
        num_channels: int,
        num_sensors: int,
        branch_width: int,
        branch_depth: int,
        trunk_width: int,
        trunk_depth: int,
        basis_functions: int,
    ):
        """A model maps observations to evaluations.

        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            num_sensors: Number of input sensors
            branch_width: Width of branch network
            branch_depth: Depth of branch network
            trunk_width: Width of trunk network
            trunk_depth: Depth of trunk network
            basis_functions: Number of basis functions
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.basis_functions = basis_functions

        branch_input = num_sensors * (coordinate_dim + num_channels)
        trunk_input = coordinate_dim

        self.branch = DeepResidualNetwork(
            branch_input,
            self.num_channels * basis_functions,
            branch_width,
            branch_depth,
        )

        self.trunk = DeepResidualNetwork(
            trunk_input,
            self.num_channels * basis_functions,
            trunk_width,
            trunk_depth,
        )

    def forward(self, u, x):
        """Forward pass."""
        batch_size_u = u.shape[0]
        batch_size_x = x.shape[1]

        u = u.reshape((batch_size_u, -1))
        x = x.reshape((-1, self.coordinate_dim))

        b = self.branch(u)
        t = self.trunk(x)

        b = b.reshape((batch_size_u, self.basis_functions, self.num_channels))
        t = t.reshape(
            (batch_size_u, batch_size_x, self.basis_functions, self.num_channels)
        )

        sum = torch.einsum("ubc,uxbc->uxc", b, t)
        assert sum.shape == (batch_size_u, batch_size_x, self.num_channels)
        return sum


class NeuralOperator(Operator):
    """Neural operator architecture."""

    def __init__(
        self,
        coordinate_dim: int,
        num_channels: int,
        depth: int,
        kernel_width: int,
        kernel_depth: int,
    ):
        """Maps observations and positions to evaluations.

        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            depth: Number of hidden layers
            kernel_width: Width of kernel network
            kernel_depth: Depth of kernel network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels

        self.lifting = ContinuousConvolution(
            coordinate_dim,
            num_channels,
            DeepResidualNetwork(1, 1, kernel_width, kernel_depth),
        )

        self.hidden_layers = torch.nn.ModuleList(
            [
                ContinuousConvolution(
                    coordinate_dim,
                    num_channels,
                    DeepResidualNetwork(1, 1, kernel_width, kernel_depth),
                )
                for _ in range(depth)
            ]
        )

        self.projection = ContinuousConvolution(
            coordinate_dim,
            num_channels,
            DeepResidualNetwork(1, 1, kernel_width, kernel_depth),
        )

    def forward(self, yu, x):
        """Forward pass."""
        batch_size = yu.shape[0]
        assert batch_size == x.shape[0]
        x_size = x.shape[1]
        num_sensors = yu.shape[1]

        sensor_dim = self.coordinate_dim + self.num_channels
        assert yu.shape == (batch_size, num_sensors, sensor_dim)

        # Sensors positions are equal across all layers (for now)
        y = yu[:, :, -self.coordinate_dim :]

        # Lifting layer
        v = self.lifting(yu, y)
        assert v.shape == (batch_size, num_sensors, self.num_channels)

        # First input to hidden layers is (y, v)
        yv = torch.cat((y, v), axis=-1)
        assert yv.shape == (batch_size, num_sensors, sensor_dim)

        # Layers
        for layer in self.hidden_layers:
            # Layer operation (with residual connection)
            v = layer(yv, y) + v
            assert v.shape == (batch_size, num_sensors, self.num_channels)

            # Activation
            v = torch.tanh(v)

            yv = torch.cat((y, v), axis=-1)
            assert yv.shape == (batch_size, num_sensors, sensor_dim)

        # Projection layer
        w = self.projection(yv, x)
        assert w.shape == (batch_size, x_size, self.num_channels)
        return w
