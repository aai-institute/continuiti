"""
`continuiti.operators.modulus.fno`

The Fourier Neural Operator from NVIDIA Modulus wrapped in continuiti.
"""

import torch
from typing import Optional
from continuiti.operators import Operator, OperatorShapes
from modulus.models.fno import FNO as FNOModulus


class FNO(Operator):
    r"""FNO architecture from NVIDIA Modulus.

    The `in_channels` and `out_channels` arguments are determined by the
    `shapes` argument. The `dimension` is set to the dimension of the input
    coordinates, assuming that the grid dimension is the same as the coordinate
    dimension of `x`.

    All other keyword arguments are passed to the Fourier Neural Operator, please refer
    to the documentation of the `modulus.model.fno.FNO` class for more information.

    Args:
        shapes: Shapes of the input and output data.
        device: Device.
        **kwargs: Additional arguments for the Fourier layers.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        device: Optional[torch.device] = None,
        dimension: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(shapes, device)

        if dimension is None:
            # Per default, use coordinate dimension
            dimension = shapes.x.dim

        self.fno = FNOModulus(
            in_channels=shapes.u.dim,
            out_channels=shapes.v.dim,
            dimension=dimension,
            **kwargs,
        )
        self.fno.to(device)

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward pass of the Fourier Neural Operator.

        Args:
            x: Ignored.
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Ignored.
        """
        return self.fno(u)
