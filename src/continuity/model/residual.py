import torch


class ResidualLayer(torch.nn.Module):
    """Residual layer."""

    def __init__(self, width: int):
        super().__init__()
        self.layer = torch.nn.Linear(width, width)
        self.act = torch.nn.Tanh()

    def forward(self, x):
        return self.act(self.layer(x)) + x


class DeepResidualNetwork(torch.nn.Module):
    """Deep residual network."""

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
        x = self.first_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.last_layer(x)
