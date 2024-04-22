"""
`continuiti.operators.common`

Common functionality for operators in continuiti.
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


class ResidualLayer(torch.nn.Module):
    """Residual layer.

    Args:
        width: Width of the layer.
        act: Activation function.
        device: Device.
    """

    def __init__(
        self,
        width: int,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.layer = torch.nn.Linear(width, width, device=device)
        self.act = act or torch.nn.GELU()
        self.norm = torch.nn.LayerNorm(width, device=device)

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        return self.norm(self.act(self.layer(x))) + x


class DeepResidualNetwork(torch.nn.Module):
    """Deep residual network.

    Args:
        input_size: Size of input tensor
        output_size: Size of output tensor
        width: Width of hidden layers
        depth: Number of hidden layers
        act: Activation function
        device: Device.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        width: int,
        depth: int,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        assert depth >= 1, "DeepResidualNetwork has at least depth 1."
        super().__init__()

        self.act = act or torch.nn.GELU()
        self.first_layer = torch.nn.Linear(input_size, width, device=device)
        self.hidden_layers = torch.nn.ModuleList(
            [
                ResidualLayer(
                    width,
                    act=self.act,
                    device=device,
                )
                for _ in range(1, depth)
            ]
        )
        self.last_layer = torch.nn.Linear(width, output_size, device=device)

    def forward(self, x):
        """Forward pass."""
        x = self.first_layer(x)
        x = self.act(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)
