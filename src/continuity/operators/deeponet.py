"""
`continuity.operators.deeponet`

The DeepONet architecture.
"""

import torch
from continuity.operators import Operator
from continuity.operators.common import DeepResidualNetwork
from continuity.data import DatasetShapes


class DeepONet(Operator):
    r"""
    Maps continuous functions given as observation to another continuous function and returns point-wise evaluations.
    The architecture is inspired by the universal approximation theorem for operators.

    *Reference:* Lu Lu et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of
    operators. Nat Mach Intell 3 218-229 (2021)

    **Note:** This operator is not discretization invariant, i.e., it assumes that all observations were evaluated at
    the same positions.

    Args:
        shapes: Shape variable of the dataset
        branch_width: Width of branch network
        branch_depth: Depth of branch network
        trunk_width: Width of trunk network
        trunk_depth: Depth of trunk network
        basis_functions: Number of basis functions
    """

    def __init__(
        self,
        shapes: DatasetShapes,
        branch_width: int = 32,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        basis_functions: int = 8,
    ):
        super().__init__()

        self.shapes = shapes
        self.branch_width = branch_width
        self.branch_depth = branch_depth
        self.trunk_width = trunk_width
        self.trunk_depth = trunk_depth
        self.basis_functions = basis_functions

        self.basis_functions = basis_functions
        self.dot_dim = shapes.v.dim * basis_functions
        # trunk network
        self.trunk = DeepResidualNetwork(
            input_size=shapes.y.dim,
            output_size=self.dot_dim,
            width=trunk_width,
            depth=trunk_depth,
        )
        # branch network
        branch_input_dim = shapes.u.num * shapes.u.dim
        self.branch = DeepResidualNetwork(
            input_size=branch_input_dim,
            output_size=self.dot_dim,
            width=branch_width,
            depth=branch_depth,
        )

    def __str__(self):
        """String representation of the operator."""
        s = f"DeepONet(branch=({self.branch_width}, {self.branch_depth}), "
        s += f"trunk=({self.trunk_width}, {self.trunk_depth}), "
        s += f"basis={self.basis_functions})"
        return s

    def forward(
        self, _: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            _: Ignored.
            u: Input function values of shape (batch_size, #sensors, u_dim)
            y: Evaluation coordinates of shape (batch_size, #evaluations, y_dim)

        Returns:
            Operator output (batch_size, #evaluations, v_dim)
        """
        assert u.size(0) == y.size(0)

        # flatten inputs for both trunk and branch network
        u = u.flatten(1, -1)
        assert u.shape[1:] == torch.Size([self.shapes.u.num * self.shapes.u.dim])

        y = y.flatten(0, 1)
        assert y.shape[-1:] == torch.Size([self.shapes.y.dim])

        # Pass through branch and trunk networks
        b = self.branch(u)
        t = self.trunk(y)

        # dot product
        b = b.reshape(-1, self.shapes.v.dim, self.basis_functions)
        t = t.reshape(
            b.size(0),
            -1,
            self.shapes.v.dim,
            self.basis_functions,
        )
        dot_prod = torch.einsum("abcd,acd->abc", t, b)

        return dot_prod
