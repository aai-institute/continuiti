"""The DeepONet architecture."""

import torch
from torch import Tensor
from continuity.operators import Operator
from continuity.operators.common import DeepResidualNetwork


class DeepONet(Operator):
    r"""
    Maps continuous functions given as observation to another continuous
    functions and returns point-wise evaluations. The architecture is inspired
    by the universal approximation theorem for operators.

    *Reference:* Lu Lu et al. Learning nonlinear operators via DeepONet based
    on the universal approximation theorem of operators. Nat Mach Intell 3
    218-229 (2021)

    **Note:** This operator is not discretization invariant, i.e., it assumes
    that all observations were evaluated at the same positions.

    Args:
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
        num_sensors: Number of sensors (fixed!)
        branch_width: Width of branch network
        branch_depth: Depth of branch network
        trunk_width: Width of trunk network
        trunk_depth: Depth of trunk network
        basis_functions: Number of basis functions
    """

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
        super().__init__()

        self.coordinate_dim = coordinate_dim
        self.num_channels = num_channels
        self.num_sensors = num_sensors
        self.basis_functions = basis_functions

        branch_input = num_sensors * num_channels
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

    def forward(self, xu: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            xu: Tensor of observations of shape (batch_size, num_sensors, coordinate_dim + num_channels). Note that positions are ignored!
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, y_size, coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, num_channels)
        """
        batch_size = xu.shape[0]
        y_size = y.shape[1]

        # Check shapes
        assert y.shape[0] == batch_size
        assert y.shape[2] == self.coordinate_dim
        assert xu.shape[2] == self.coordinate_dim + self.num_channels

        # Sensors are (x, u), but here we only use u
        u = xu[:, :, -self.num_channels :]
        assert u.shape == (batch_size, self.num_sensors, self.num_channels)

        # Reshape branch and trunk inputs
        u = u.reshape((batch_size, -1))
        y = y.reshape((-1, self.coordinate_dim))

        # Pass trough branch and trunk networks
        b = self.branch(u)
        t = self.trunk(y)

        # Compute dot product
        b = b.reshape((batch_size, self.basis_functions, self.num_channels))
        t = t.reshape((batch_size, y_size, self.basis_functions, self.num_channels))
        sum = torch.einsum("ubc,uxbc->uxc", b, t)
        assert sum.shape == (batch_size, y_size, self.num_channels)
        return sum
