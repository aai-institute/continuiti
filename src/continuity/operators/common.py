"""
`continuity.operators.common`

Common functionality for operators in Continuity.
"""

import torch


class ResidualLayer(torch.nn.Module):
    """Residual layer.

    Args:
        width: Width of the layer.
    """

    def __init__(self, width: int):
        super().__init__()
        self.layer = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return self.act(self.layer(x)) + x


class DeepResidualNetwork(torch.nn.Module):
    """Deep residual network.

    Args:
        input_size: Size of input tensor
        output_size: Size of output tensor
        width: Width of hidden layers
        depth: Number of hidden layers
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
    ):
        super().__init__()

        self.first_layer = torch.nn.Linear(input_size, width)
        self.hidden_layers = torch.nn.ModuleList(
            [ResidualLayer(width) for _ in range(depth)]
        )
        self.last_layer = torch.nn.Linear(width, output_size)

    def forward(self, x):
        """Forward pass."""
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)


class NeuralNetworkKernel(torch.nn.Module):
    """Neural network kernel.

    Args:
        kernel_width: Width of kernel network
        kernel_depth: Depth of kernel network

    TODO: As it is implemented, this is a convolution `k(x - y)`, but it should be more general.
    """

    def __init__(
        self,
        kernel_width: int,
        kernel_depth: int,
    ):
        super().__init__()
        self.net = DeepResidualNetwork(
            1,
            1,
            kernel_width,
            kernel_depth,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute kernel value.

        Args:
            x: Tensor of shape (..., coordinate_dim)
            y: Tensor of shape (..., coordinate_dim)

        Returns:
            Tensor of shape (...)
        """
        r = ((x - y) ** 2).sum(dim=-1)
        output_shape = r.shape
        r = r.reshape((-1, 1))
        k = self.net(r)
        k = k.reshape(output_shape)
        return k
