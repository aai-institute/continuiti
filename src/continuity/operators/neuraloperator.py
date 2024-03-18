"""
`continuity.operators.neuraloperator`

Operators can be stacked into a `NeuralOperator` architecture, which is a
stack of continuous convolutions with a lifting layer and a projection layer.
"""

import torch
from typing import List, Optional
from continuity.operators import Operator
from continuity.operators.shape import OperatorShapes


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
        act: Activation function. Default is tanh.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        layers: List[Operator],
        act: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        self.shapes = shapes
        self.layers = torch.nn.ModuleList(layers)
        self.act = act or torch.nn.Tanh()

        self.first_dim = layers[0].shapes.u.dim
        self.last_dim = layers[-1].shapes.v.dim

        assert self.shapes.u.num == layers[0].shapes.u.num
        assert self.shapes.v.num == layers[-1].shapes.v.num

        self.lifting = torch.nn.Linear(self.shapes.u.dim, self.first_dim)
        self.projection = torch.nn.Linear(self.last_dim, self.shapes.v.dim)

        self.W = torch.nn.ModuleList(
            [
                torch.nn.Linear(layer.shapes.u.dim, layer.shapes.v.dim)
                for layer in layers
            ]
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, coordinate_dim).
            u: Input function values of shape (batch_size, num_sensors, num_channels).
            y: Coordinates where the mapped function is evaluated of shape (batch_size, y_size, coordinate_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, y_size, num_channels)
        """
        assert u.shape[1:] == torch.Size([self.shapes.u.num, self.shapes.u.dim])

        # Lifting
        u = u.reshape(-1, self.shapes.u.dim)
        v = self.lifting(u)
        v = v.reshape(-1, self.shapes.u.num, self.first_dim)

        # Hidden layers
        for i, layer in enumerate(self.layers):
            v = self.act(layer(x, v, x)) + self.W[i](v)

        # Projection
        v = v.reshape(-1, self.last_dim)
        v = self.projection(v)
        w = v.reshape(-1, self.shapes.v.num, self.shapes.v.dim)

        assert w.shape[1:] == torch.Size([y.size(1), self.shapes.v.dim])
        return w
