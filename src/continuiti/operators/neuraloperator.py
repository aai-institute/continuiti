"""
`continuiti.operators.neuraloperator`

Operators can be stacked into a `NeuralOperator` architecture, which is a
stack of continuous convolutions with a lifting layer and a projection layer.
"""

import torch
from typing import List, Optional
from continuiti.operators import Operator
from continuiti.operators.shape import OperatorShapes


class NeuralOperator(Operator):
    r"""Neural operator architecture

    Maps continuous functions given as observation to another continuous function and returns point-wise evaluations.
    The architecture is a stack of continuous kernel integrations with a lifting layer and a projection layer.

    *Reference:* N. Kovachki et al. Neural Operator: Learning Maps Between
    Function Spaces With Applications to PDEs. JMLR 24 1-97 (2023)

    For now, sensor positions are equal across all layers.

    Args:
        shapes: Shapes of the input and output data.
        layers: List of operator layers.
        act: Activation function.
        device: Device.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        layers: List[Operator],
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)

        self.layers = torch.nn.ModuleList(layers)
        self.act = act or torch.nn.GELU()

        self.first_dim = layers[0].shapes.u.dim
        self.last_dim = layers[-1].shapes.v.dim

        assert self.shapes.x == layers[0].shapes.x
        assert self.shapes.u.size == layers[0].shapes.u.size
        assert self.shapes.y == layers[-1].shapes.y
        assert self.shapes.v.size == layers[-1].shapes.v.size

        self.lifting = torch.nn.Linear(self.shapes.u.dim, self.first_dim, device=device)
        self.projection = torch.nn.Linear(
            self.last_dim, self.shapes.v.dim, device=device
        )

        self.W = torch.nn.ModuleList(
            [
                torch.nn.Linear(layer.shapes.u.dim, layer.shapes.v.dim, device=device)
                for layer in layers[:-1]
            ]
        )
        self.norms = torch.nn.ModuleList(
            [
                torch.nn.LayerNorm(layer.shapes.v.dim, device=device)
                for layer in layers[:-1]
            ]
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, coordinate_dim, num_sensors, ...).
            u: Input function values of shape (batch_size, num_channels, num_sensors, ...).
            y: Coordinates where the mapped function is evaluated of shape (batch_size, coordinate_dim, y_size, ...)

        Returns:
            Evaluations of the mapped function with shape (batch_size, num_channels, y_size, ...)
        """
        assert u.shape[1:] == torch.Size([self.shapes.u.dim, *self.shapes.u.size])

        # Lifting
        u = u.reshape(-1, self.shapes.u.dim)
        v = self.lifting(u)
        v = v.reshape(-1, self.first_dim, *self.shapes.u.size)

        # Hidden layers
        for layer, W, norm in zip(self.layers[:-1], self.W, self.norms):
            v1 = layer(x, v, x)

            v = v1.transpose(1, -1) + W(v.transpose(1, -1))

            v = self.act(v)
            v = norm(v)

            v = v.transpose(1, -1)

        # Last layer (evaluates y)
        v = self.layers[-1](x, v, y)

        # Projection
        v = v.reshape(-1, self.last_dim)
        v = self.projection(v)
        w = v.reshape(-1, self.shapes.v.dim, *self.shapes.v.size)

        assert w.shape[1:] == torch.Size([self.shapes.v.dim, *y.size()[2:]])
        return w
