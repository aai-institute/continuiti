"""
`continuity.operators.fusion`

The Fusion architecture.
"""

import torch
from continuity.operators import Operator
from continuity.operators.common import DeepResidualNetwork
from continuity.data import DatasetShapes


class FusionOperator(Operator):
    """
    The `FusionOperator` class integrates a Deep Residual Network within a neural operator framework, designed for
     effectively processing and fusing input function values and evaluation coordinates.

    Args:
        shapes: An instance of `DatasetShapes`.
        width: The width of the Deep Residual Network, defining the number of neurons in each hidden layer.
        depth: The depth of the Deep Residual Network, indicating the number of hidden layers in the network.

    """

    def __init__(self, shapes: DatasetShapes, width: int = 32, depth: int = 3):
        super().__init__()
        self.shapes = shapes

        self.width = width
        self.depth = depth

        self.net = DeepResidualNetwork(
            input_size=shapes.y.dim + shapes.u.num * shapes.u.dim,
            output_size=shapes.v.dim,
            width=width,
            depth=depth,
        )

    def forward(
        self, _: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Performs the forward pass through the operator, processing the input function values `u` and evaluation
        coordinates `y` to produce the operator output. The method ignores the first argument, utilizes `u` and `y`
        to create a combined input for the deep residual network, and reshapes the network's output to match the
        expected dimensions.

        Args:
            _: Ignored. Placeholder for compatibility with other operators.
            u: Input function values of shape (batch_size, #sensors, u_dim), representing the values of the input
                functions at different sensor locations.
            y: Evaluation coordinates of shape (batch_size, #evaluations, y_dim), representing the points in space at
                which the output function values are to be computed.

        Returns:
            The output of the operator, of shape (batch_size, #evaluations, v_dim), representing the computed function
                values at the specified evaluation coordinates.
        """
        assert u.size(0) == y.size(0)
        batch_size = u.size(0)

        # Get all combinations
        u = u.unsqueeze(1).expand(-1, self.shapes.y.num, -1, -1)
        y = y.unsqueeze(2).expand(-1, -1, self.shapes.u.num, -1)
        net_input = torch.cat([u, y], dim=-1)

        v = self.net(net_input)
        v = v.view(batch_size, self.shapes.y.num, self.shapes.v.dim)

        return v
