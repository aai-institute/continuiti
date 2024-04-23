"""
`continuiti.networks.fully_connected`

Fully connected neural network in continuiti.
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
        device: Device.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.inner_layer = torch.nn.Linear(input_size, width, device=device)
        self.outer_layer = torch.nn.Linear(width, output_size, device=device)
        self.act = act or torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(width, device=device)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        x = self.inner_layer(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.outer_layer(x)
        return x
