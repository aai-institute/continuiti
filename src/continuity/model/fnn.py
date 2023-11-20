import torch
from continuity.model import device
from continuity.model.residual import DeepResidualNetwork
from continuity.model.torchmodel import TorchModel


class FullyConnected(TorchModel):
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

        self.input_size = num_sensors * (num_channels + coordinate_dim) + coordinate_dim
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
