import torch
from continuity.model.residual import DeepResidualNetwork
from continuity.model.torchmodel import TorchModel


class DeepONet(TorchModel):
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

        branch_input = num_sensors * (num_channels + coordinate_dim)
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
