"""
`continuity.networks.res_net`

ResNet in Continuity.
"""

import torch
from typing import Optional


class ResidualLayer(torch.nn.Module):
    """Residual layer.

    Args:
        width: Width of the layer.
        act: Activation function.
    """

    def __init__(self, width: int, act: Optional[torch.nn.Module] = None):
        super().__init__()
        self.layer = torch.nn.Linear(width, width)
        self.act = act or torch.nn.Tanh()

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
        act: Activation function
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        act: Optional[torch.nn.Module] = None,
    ):
        assert depth >= 1, "DeepResidualNetwork has at least depth 1."
        super().__init__()

        self.act = act or torch.nn.Tanh()
        self.first_layer = torch.nn.Linear(input_size, width)
        self.hidden_layers = torch.nn.ModuleList(
            [ResidualLayer(width, act=self.act) for _ in range(1, depth)]
        )
        self.last_layer = torch.nn.Linear(width, output_size)

    def forward(self, x):
        """Forward pass."""
        x = self.first_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)
