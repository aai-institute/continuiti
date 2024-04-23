"""
`continuiti.operators.deeponet`

The DeepONet architecture.
"""

import math
import torch
from typing import Optional
from continuiti.operators import Operator
from continuiti.networks import DeepResidualNetwork
from continuiti.operators.shape import OperatorShapes


class DeepONet(Operator):
    r"""
    Maps continuous functions given as observation to another continuous function and returns point-wise evaluations.
    The architecture is inspired by the universal approximation theorem for operators.

    *Reference:* Lu Lu et al. Learning nonlinear operators via DeepONet based on the universal approximation theorem of
    operators. Nat Mach Intell 3 218-229 (2021)

    **Note:** This operator is not discretization invariant, i.e., it assumes that all observations were evaluated at
    the same positions.

    Args:
        shapes: Shapes of the operator.
        branch_width: Width of branch network.
        branch_depth: Depth of branch network.
        trunk_width: Width of trunk network.
        trunk_depth: Depth of trunk network.
        basis_functions: Number of basis functions.
        act: Activation function.
        device: Device.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        branch_width: int = 32,
        branch_depth: int = 3,
        trunk_width: int = 32,
        trunk_depth: int = 3,
        basis_functions: int = 8,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)

        self.basis_functions = basis_functions
        self.dot_dim = shapes.v.dim * basis_functions
        # trunk network
        self.trunk = DeepResidualNetwork(
            input_size=shapes.y.dim,
            output_size=self.dot_dim,
            width=trunk_width,
            depth=trunk_depth,
            act=act,
            device=device,
        )
        # branch network
        self.branch_input_dim = math.prod(shapes.u.size) * shapes.u.dim
        self.branch = DeepResidualNetwork(
            input_size=self.branch_input_dim,
            output_size=self.dot_dim,
            width=branch_width,
            depth=branch_depth,
            act=act,
            device=device,
        )

    def forward(
        self, _: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            _: Ignored.
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Operator output (batch_size, v_dim, num_evaluations...).
        """
        assert u.size(0) == y.size(0)
        y_num = y.shape[2:]

        # flatten inputs for both trunk and branch network
        u = u.flatten(1, -1)
        assert u.shape[1:] == torch.Size([self.branch_input_dim])

        y = y.swapaxes(1, -1).flatten(0, -2)
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
        dot_prod = torch.einsum("abcd,acd->acb", t, b)
        dot_prod = dot_prod.reshape(-1, self.shapes.v.dim, *y_num)

        return dot_prod
