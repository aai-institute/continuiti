"""
`continuiti.operators.cnn`

The ConvolutionalNeuralNetwork (CNN) architecture.
"""

import torch
from typing import Optional, Tuple
from continuiti.operators import Operator
from continuiti.operators.shape import OperatorShapes


class ConvolutionalNeuralNetwork(Operator):
    """
    The `ConvolutionalNeuralNetwork` class is a convolutional neural network
    that can be viewed at as an operator on a fixed grid.

    Args:
        shapes: An instance of `OperatorShapes`.
        width: The number hidden channels.
        depth: The number of hidden layers.
        kernel_size: The size of the convolutional kernel.
        grid_shape: x and y have to be sampled on a grid. If grid dimensions
            are not specified, a grid with equal sizes is assumed.
        act: Activation function.
        device: Device.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        width: int = 16,
        depth: int = 3,
        kernel_size: int = 3,
        grid_shape: Optional[Tuple[int]] = None,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        assert depth >= 1, "Depth is at least one."
        super().__init__(shapes, device)

        self.act = torch.nn.Tanh() if act is None else act
        self.grid_shape = grid_shape
        padding = kernel_size // 2

        assert shapes.x.dim in [1, 2, 3], "Only 1D, 2D, and 3D grids supported."
        Conv = [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d][shapes.x.dim - 1]

        self.first_layer = Conv(
            shapes.u.dim, width, kernel_size=kernel_size, padding=padding, device=device
        )
        self.hidden_layers = torch.nn.ModuleList(
            Conv(width, width, kernel_size=kernel_size, padding=padding, device=device)
            for _ in range(depth - 1)
        )
        self.last_layer = Conv(
            width, shapes.v.dim, kernel_size=kernel_size, padding=padding, device=device
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Performs the forward pass through the operator.
        This assumes `x == y` and that `x` is a quadratic grid of positions.

        Args:
            x: Ignored.
            u: Input function values of shape (batch_size, #sensors, u_dim).
            y: Ignored.

        Returns:
            The output of the operator, of shape (batch_size, #evaluations, v_dim).
        """
        # Transform input to (batch_size, u_dim, ux, uy, ...)
        if self.grid_shape is not None:
            u = u.reshape(-1, *self.grid_shape, self.shapes.u.dim)
        else:
            per_dim = int(self.shapes.u.num ** (1 / self.shapes.x.dim))
            u = u.reshape([-1] + [per_dim] * self.shapes.x.dim + [self.shapes.u.dim])
        u = u.swapaxes(1, -1)

        # Convolutional layers
        residual = u
        u = self.act(self.first_layer(u))
        for layer in self.hidden_layers:
            u = self.act(layer(u))
        u = self.last_layer(u) + residual

        # Transform back to (batch_size, v_num, v_dim)
        u = u.swapaxes(1, -1)
        u = u.reshape(-1, self.shapes.v.num, self.shapes.v.dim)

        return u
