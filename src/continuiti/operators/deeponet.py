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
        act: Activation function used in default trunk and branch networks.
        device: Device.
        branch_network: Custom branch network that maps input function
            evaluations to `basis_functions` many coefficients (if set,
            branch_width and branch_depth will be ignored).
        trunk_network: Custom trunk network that maps `shapes.y.dim`-dimensional
            evaluation coordinates to `basis_functions` many basis function
            evaluations (if set, trunk_width and trunk_depth will be ignored).
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
        branch_network: Optional[torch.nn.Module] = None,
        trunk_network: Optional[torch.nn.Module] = None,
    ):
        super().__init__(shapes, device)

        # trunk network
        if trunk_network is not None:
            self.trunk = trunk_network
            self.trunk.to(device)
        else:
            self.trunk = DeepResidualNetwork(
                input_size=shapes.y.dim,
                output_size=shapes.v.dim * basis_functions,
                width=trunk_width,
                depth=trunk_depth,
                act=act,
                device=device,
            )

        # branch network
        if branch_network is not None:
            self.branch = branch_network
            self.branch.to(device)
        else:
            branch_input_dim = math.prod(shapes.u.size) * shapes.u.dim
            self.branch = torch.nn.Sequential(
                torch.nn.Flatten(),
                DeepResidualNetwork(
                    input_size=branch_input_dim,
                    output_size=shapes.v.dim * basis_functions,
                    width=branch_width,
                    depth=branch_depth,
                    act=act,
                    device=device,
                ),
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

        # flatten inputs for trunk network
        y = y.swapaxes(1, -1).flatten(0, -2)
        assert y.shape[-1:] == torch.Size([self.shapes.y.dim])

        # Pass through branch network
        b = self.branch(u)

        # Pass through trunk network
        t = self.trunk(y)

        assert b.shape[1:] == t.shape[1:], (
            f"Branch network output of shape {b.shape[1:]} does not match "
            f"trunk network output of shape {t.shape[1:]}"
        )

        # determine basis functions dynamically
        basis_functions = b.shape[1] // self.shapes.v.dim

        # dot product
        b = b.reshape(-1, self.shapes.v.dim, basis_functions)
        t = t.reshape(
            b.size(0),
            -1,
            self.shapes.v.dim,
            basis_functions,
        )
        dot_prod = torch.einsum("abcd,acd->acb", t, b)
        dot_prod = dot_prod.reshape(-1, self.shapes.v.dim, *y_num)

        return dot_prod
