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
            input_size=(
                shapes.y.dim + shapes.u.dim * shapes.u.num + shapes.x.dim * shapes.x.num
            ),
            output_size=shapes.v.dim,
            width=width,
            depth=depth,
        )

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Performs the forward pass through the operator, processing the input function values `u` and input function
        probe locations `x` by flattening them. They are then expanded to match the dimensions of the evaluation
        coordinates y. The preprocessed x, preprocessed u, and y are stacked and passed through a deep residual network.


        Args:
            x: Input coordinates of shape (batch_size, #sensors, x_dim), representing the points in space at
                which the input function values are probed.
            u: Input function values of shape (batch_size, #sensors, u_dim), representing the values of the input
                functions at different sensor locations.
            y: Evaluation coordinates of shape (batch_size, #evaluations, y_dim), representing the points in space at
                which the output function values are to be computed.

        Returns:
            The output of the operator, of shape (batch_size, #evaluations, v_dim), representing the computed function
                values at the specified evaluation coordinates.
        """
        # u repeated shape (batch_size, #evaluations, #sensors * u_dim)
        u_repeated = u.flatten(1, 2).unsqueeze(1).expand(-1, y.size(1), -1)
        # x repeated shape (batch_size, #evaluations, #sensors * x_dim)
        x_repeated = x.flatten(1, 2).unsqueeze(1).expand(-1, y.size(1), -1)

        net_input = torch.cat([x_repeated, u_repeated, y], dim=-1)

        return self.net(net_input)
