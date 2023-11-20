import torch
from continuity.model.residual import DeepResidualNetwork
from continuity.model.torchmodel import TorchModel


class ContinuousConvolutionLayer(torch.nn.Module):
    """Continuous convolution layer."""

    def __init__(
        self,
        coordinate_dim: int,
        num_channels: int,
        width: int,
        depth: int,
    ):
        """Maps continuous functions via convolution with a trainable kernel to another continuous functions using point-wise evaluations.

        Args:
            coordinate_dim: Dimension of coordinate space
            num_channels: Number of channels
            width: Width of kernel network
            depth: Depth of kernel network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels

        self.kernel = DeepResidualNetwork(1, 1, width, depth)

    def forward(self, yu, x):
        """Forward pass."""
        batch_size = yu.shape[0]
        num_sensors = yu.shape[1]
        assert batch_size == x.shape[0]
        x_size = x.shape[1]

        # Sensors are (x, u)
        y = yu[:, :, -self.coordinate_dim :]
        u = yu[:, :, : -self.coordinate_dim]
        assert y.shape == (batch_size, num_sensors, self.coordinate_dim)
        assert u.shape == (batch_size, num_sensors, self.num_channels)

        # Compute radial coordinates
        assert x.shape == (batch_size, x_size, self.coordinate_dim)
        x = x.unsqueeze(1)
        y = y.unsqueeze(2)

        r = torch.sum((x - y) ** 2, axis=-1)
        assert r.shape == (batch_size, num_sensors, x_size)

        # Flatten to 1D
        r = r.reshape((-1, 1))

        # Kernel operation
        k = self.kernel(r)

        # Reshape to 3D
        k = k.reshape((batch_size, num_sensors, x_size))

        # Compute integral
        integral = torch.einsum("bsx,bsc->bxc", k, u) / num_sensors
        assert integral.shape == (batch_size, x_size, self.num_channels)
        return integral


class NeuralOperator(TorchModel):
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
            width: Width of kernel network
            depth: Depth of kernel network
        """
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels

        self.lifting = ContinuousConvolutionLayer(
            coordinate_dim,
            num_channels,
            kernel_width,
            kernel_depth,
        )

        self.hidden_layers = torch.nn.ModuleList(
            [
                ContinuousConvolutionLayer(
                    coordinate_dim,
                    num_channels,
                    kernel_width,
                    kernel_depth,
                )
                for _ in range(depth)
            ]
        )

        self.projection = ContinuousConvolutionLayer(
            coordinate_dim,
            num_channels,
            kernel_width,
            kernel_depth,
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
