"""
`continuity.networks.fully_connected`

Fully connected neural network in Continuity.
"""

import torch
from typing import Optional


class FullyConnected(torch.nn.Module):
    """Fully connected network.

    Args:
        input_size: Input dimension.
        output_size: Output dimension.
        width: Width of the hidden layer.
        act: Activation function.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        act: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.inner_layer = torch.nn.Linear(input_size, width)
        self.outer_layer = torch.nn.Linear(width, output_size)
        self.act = act or torch.nn.Tanh()

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        x = self.inner_layer(x)
        x = self.act(x)
        x = self.outer_layer(x)
        return x
