"""
`continuiti.operators.mino`

The mesh-independent neural operator (MINO).
"""

import math
import torch
from typing import Optional
from continuiti.operators import Operator
from continuiti.operators.shape import OperatorShapes, TensorShape


class AttentionKernel(Operator):
    def __init__(
        self,
        shapes: OperatorShapes,
        query_dim: int = 1,
        bias: bool = False,
        device: Optional[torch.device] = None,
    ):
        """

        Args:
            shapes: Shapes of the operator.
            query_dim: Dimension of the query vectors.
            bias: Whether to include bias in the linear layers.
            device: Device.
        """
        super().__init__(shapes, device)
        self.query_dim = query_dim

        # W^q in \mathbb{R}^{d_y \times d_q}
        # W^k in \mathbb{R}^{d_x \times d_q}
        # W^v in \mathbb{R}^{d_x \times d_v}
        self.W_q = torch.nn.Linear(shapes.y.dim, query_dim, bias=bias, device=device)
        self.W_k = torch.nn.Linear(shapes.x.dim, query_dim, bias=bias, device=device)
        self.W_v = torch.nn.Linear(shapes.x.dim, shapes.v.dim, bias=bias, device=device)

    def forward(
        self,
        x: torch.Tensor,
        _: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        r"""Forward pass through the attention kernel.

        For input vectors $X \in \mathbb{R}^{n_x \times d_x}$ and query vectors
        $Y \in \mathbb{R}^{n_y \times d_y}$, the attention kernel is defined as

        $$
        Att(Y, X, X) = \sigma (Q K^T) V,
        $$

        where
        $Q = Y W^q \in \mathbb{R}^{n_y \times d_q}$,
        $K = X W^k \in \mathbb{R}^{n_x \times d_q}$,
        $V = X W^v \in \mathbb{R}^{n_x \times d_v}$
        are the query, key, and value matrices, respectively.

        Args:
            x: Evaluation coordinates of shape (batch_size, x_dim, num_sensors...).
            _: Ignored input (to match the operator interface)
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Attention kernel output (batch_size, v_dim..., num_sensors...).
        """
        # n_x = num_sensors
        # d_x = x_dim
        # n_y = num_evaluations
        # d_y = y_dim
        # d_v = v_dim
        # d_q = width

        batch_size = y.size(0)
        assert x.size(0) == batch_size
        num_evaluations = math.prod(y.shape[2:])
        num_sensors = math.prod(x.shape[2:])
        y_dim = self.shapes.y.dim
        x_dim = self.shapes.x.dim
        v_dim = self.shapes.v.dim

        # flatten inputs
        y_flat = y.flatten(2, -1).transpose(1, -1)
        x_flat = x.flatten(2, -1).transpose(1, -1)
        assert y_flat.shape == (batch_size, num_evaluations, y_dim)
        assert x_flat.shape == (batch_size, num_sensors, x_dim)

        # query, key, and value matrices
        Q = self.W_q(y_flat)
        K = self.W_k(x_flat)
        V = self.W_v(x_flat)
        assert Q.shape == (batch_size, num_evaluations, self.query_dim)
        assert K.shape == (batch_size, num_sensors, self.query_dim)
        assert V.shape == (batch_size, num_sensors, v_dim)

        # attention kernel
        dot_prod = torch.einsum("byd, bxd -> byx", Q, K)
        dot_prod = torch.nn.functional.softmax(dot_prod, dim=-1)
        att = torch.einsum("byx, bxd -> byd", dot_prod, V)

        # reshape output
        assert att.shape == (batch_size, num_evaluations, v_dim)
        att = att.transpose(1, -1).reshape(batch_size, v_dim, *y.shape[2:])

        return att


class MINO(Operator):
    r"""
    The mesh-independent neural operator (MINO) is an attention-based neural
    operator that treats the input functions evaluations as an unordered set.

    *Reference:* Seungjun Lee. Mesh-Independent Operator Learning for Partial
    Differential Equations. 2nd AI4Science Workshop, ICML (2022)

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
        query_dim: int = 32,
        depth: int = 3,
        n_z: int = 16,
        d_z: int = 8,
        d_h: int = 4,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)
        self.query_dim = query_dim
        self.depth = depth
        self.n_z = n_z
        self.d_z = d_z
        self.d_h = d_h

        self.Z_0 = torch.nn.Parameter(
            torch.randn(d_z, n_z, device=device)  # TODO: how to initialize?
        )
        z_shape = TensorShape(d_z, n_z)
        a_shape = TensorShape(shapes.x.dim + shapes.u.dim, shapes.u.size)
        h_shape = TensorShape(d_h, n_z)

        encoder_shape = OperatorShapes(
            x=a_shape,
            y=z_shape,
            u=a_shape,
            v=h_shape,
        )
        self.encoder = AttentionKernel(
            shapes=encoder_shape,
            query_dim=query_dim,
            device=device,
        )

        processor_shape = OperatorShapes(
            x=h_shape,
            y=h_shape,
            u=h_shape,
            v=h_shape,
        )
        self.layers = torch.nn.ModuleList(
            [
                AttentionKernel(
                    shapes=processor_shape,
                    query_dim=query_dim,
                    device=device,
                )
                for _ in range(depth)
            ]
        )

        decoder_shape = OperatorShapes(
            x=h_shape,
            y=shapes.y,
            u=h_shape,
            v=shapes.v,
        )
        self.decoder = AttentionKernel(
            shapes=decoder_shape,
            query_dim=query_dim,
            device=device,
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor coordinates of shape (batch_size, x_dim, num_sensors...).
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Operator output (batch_size, v_dim, num_evaluations...).
        """
        batch_size = y.size(0)
        assert x.size(0) == batch_size
        num_evaluations = math.prod(y.shape[2:])
        num_sensors = math.prod(x.shape[2:])
        x_dim = self.shapes.x.dim
        u_dim = self.shapes.u.dim
        v_dim = self.shapes.v.dim

        # flatten inputs
        x_flat = x.flatten(2, -1)
        u_flat = u.flatten(2, -1)
        assert x_flat.shape == (batch_size, x_dim, num_sensors)
        assert u_flat.shape == (batch_size, u_dim, num_sensors)

        # encoder
        a = torch.cat([x_flat, u_flat], dim=1)
        assert a.shape == (batch_size, x_dim + u_dim, num_sensors)

        z0 = self.Z_0.expand(batch_size, -1, -1)
        assert z0.shape == (batch_size, self.d_z, self.n_z)

        z1 = self.encoder(a, None, z0)
        assert z1.shape == (batch_size, self.d_h, self.n_z)

        # processor
        z = z1
        for layer in self.layers:
            z = layer(z, z, z)
        assert z.shape == (batch_size, self.d_h, self.n_z)

        # decoder
        y_flat = y.flatten(2, -1)
        assert y_flat.shape == (batch_size, self.shapes.y.dim, num_evaluations)
        output = self.decoder(z, None, y_flat)
        assert output.shape == (batch_size, v_dim, num_evaluations)

        # reshape output
        output = output.reshape(batch_size, v_dim, *y.shape[2:])
        return output
