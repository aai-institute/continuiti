"""
`continuity.operators.operator`

Operators can be stacked into a `NeuralOperator` architecture, which is a
stack of continuous convolutions with a lifting layer and a projection layer.
"""

import torch
from typing import List, Optional
from continuity.operators import Operator
from continuity.data import DatasetShapes


class NeuralOperator(Operator):
    r"""Neural operator architecture

    Maps continuous functions given as observation to another continuous function and returns point-wise evaluations.
    The architecture is a stack of continuous kernel integrations with a lifting layer and a projection layer.

    *Reference:* N. Kovachki et al. Neural Operator: Learning Maps Between
    Function Spaces With Applications to PDEs. JMLR 24 1-97 (2023)

    For now, sensor positions are equal across all layers.

    Args:
        shapes: Shapes of the input and output data.
        layers: List of operator layers. First layer is used as lifting,
                last layer as projection operator.
        act: Activation function. Default is tanh.
    """

    def __init__(
        self,
        shapes: DatasetShapes,
        layers: List[Operator],
        act: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        # Check shapes
        assert len(layers) >= 2
        assert layers[0].shapes.x == shapes.x
        assert layers[0].shapes.u == shapes.u
        for i in range(1, len(layers)):
            assert layers[i].shapes.x == layers[i - 1].shapes.y
            assert layers[i].shapes.u == layers[i - 1].shapes.v
        assert layers[-1].shapes.y == shapes.y
        assert layers[-1].shapes.v == shapes.v

        self.shapes = shapes
        self.layers = torch.nn.ModuleList(layers)
        self.act = act or torch.nn.Tanh()

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
        v = self.layers[0](x, u, x)

        # Hidden layers
        for layer in self.layers[1:-1]:
            v = self.act(layer(x, v, x)) + v

        # Projection
        w = self.layers[-1](x, v, y)

        assert w.shape[1:] == torch.Size([y.size(1), self.shapes.v.dim])
        return w
