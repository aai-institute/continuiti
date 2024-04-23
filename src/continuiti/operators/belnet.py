"""
`continuiti.operators.belnet`

The BelNet architecture.
"""

import math
import torch
from typing import Optional
from continuiti.operators import Operator
from continuiti.networks import DeepResidualNetwork
from continuiti.operators.shape import OperatorShapes


class BelNet(Operator):
    r"""
    The BelNet architecture is an extension of the DeepONet architecture that
    adds a learnable projection basis network to interpolate the sensor inputs.
    Therefore, it supports changing sensor positions, or in other terms, is
    *discretization invariant*.

    *Reference:* Z. Zhang et al. BelNet: basis enhanced learning, a mesh-free
    neural operator. Proceedings of the royal society A (2023).

    **Note:** In the paper, you can use Figure 6 for reference, but we swapped
    the notation of `x` and `y` to comply with the convention in continuiti,
    where `x` is the collocation points and `y` is the evaluation points. We
    also replace the single layer projection and construction networks by more
    expressive deep residual networks.

    Args:
        shapes: Shapes of the operator
        K: Number of basis functions
        N_1: Width of the projection basis network
        D_1: Depth of the projection basis network
        N_2: Width of the construction network
        D_2: Depth of the construction network
        a_x: Activation function of projection networks. Default: Tanh
        a_u: Activation function applied after the projection. Default: Tanh
        a_y: Activation function of the construction network. Default: Tanh
        device: Device.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        K: int = 8,
        N_1: int = 32,
        D_1: int = 3,
        N_2: int = 32,
        D_2: int = 3,
        a_x: Optional[torch.nn.Module] = None,
        a_u: Optional[torch.nn.Module] = None,
        a_y: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)

        self.K = K
        self.a_x = a_x or torch.nn.Tanh()
        self.a_u = a_u or torch.nn.Tanh()
        self.a_y = a_y or torch.nn.Tanh()

        self.Nx = math.prod(self.shapes.x.size) * self.shapes.x.dim
        self.Nu = math.prod(self.shapes.u.size) * self.shapes.u.dim
        self.Kv = K * self.shapes.v.dim

        # K projection nets
        self.p = DeepResidualNetwork(
            input_size=self.Nx,
            output_size=self.Nu * K,
            width=N_1,
            depth=D_1,
            act=self.a_x,
            device=device,
        )

        # construction net
        self.q = DeepResidualNetwork(
            input_size=shapes.y.dim,
            output_size=self.Kv,
            width=N_2,
            depth=D_2,
            act=self.a_y,
            device=device,
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, x_dim, num_sensors...)
            u: Input function values of shape (batch_size, u_dim, num_sensors...)
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...)

        Returns:
            Operator output (batch_size, v_dim, num_evaluations...)
        """
        assert x.size(0) == u.size(0) == y.size(0)
        y_size = y.size()[2:]
        num_evaluations = math.prod(y.size()[2:])

        # flatten inputs
        x = x.reshape(-1, self.Nx)
        u = u.reshape(-1, self.Nu)
        y = y.reshape(-1, self.shapes.y.dim)

        # build projection matrix
        P = self.p(x)
        P = P.reshape(-1, self.K, self.Nu)

        # perform the projection
        aPu = self.a_u(torch.einsum("bkn,bn->bk", P, u))
        assert aPu.shape[1:] == torch.Size([self.K])

        # construction net
        Q = self.q(y)
        assert Q.shape[1:] == torch.Size([self.Kv])

        # dot product
        Q = Q.reshape(-1, num_evaluations, self.K, self.shapes.v.dim)
        output = torch.einsum("bk,bckv->bvc", aPu, Q)
        output = output.reshape(-1, self.shapes.v.dim, *y_size)

        return output
