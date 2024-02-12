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
        num_sensors: Number of sensors (fixed!)
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
        branch_width: Width of branch network
        branch_depth: Depth of branch network
        trunk_width: Width of trunk network
        trunk_depth: Depth of trunk network
        basis_functions: Number of basis functions
    """

    def __init__(
        self,
        num_sensors: int,
        coordinate_dim: int = 1,
        num_channels: int = 1,
        branch_width: int = 32,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        basis_functions: int = 8,
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

    def forward(self, x: Tensor, u: Tensor, y: Tensor) -> Tensor:
        """Forward pass through the operator.

        Args:
            x: Ignored.
            u: Tensor of sensor values of shape (batch_size, num_sensors, [num_channels]). If len(u.shape) < 3, a batch dimension will be added.
            y: Tensor of coordinates where the mapped function is evaluated of shape (batch_size, y_size, [coordinate_dim]). If len(y.shape) < 3, a batch dimension will be added.

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, num_channels)
        """
        # Get batch size
        batch_size = u.shape[0]

        # Get number of evaluations
        assert len(y.shape) >= 2
        y_size = y.shape[1]

        # Check shapes
        assert y.shape[0] == batch_size
        assert u.shape[1] == self.num_sensors
        if len(u.shape) > 2:
            assert u.shape[2] == self.num_channels

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
