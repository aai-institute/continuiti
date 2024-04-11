"""
`continuiti.operators.dno`

The Deep Neural Operator (DNO) architecture.
"""

import torch
from typing import Optional
from continuiti.operators import Operator
from continuiti.operators.common import DeepResidualNetwork
from continuiti.operators.shape import OperatorShapes


class DeepNeuralOperator(Operator):
    """
    The `DeepNeuralOperator` class integrates a deep residual network within a neural operator framework. It uses all
    input locations, input values, and the evaluation point as input for a deep residual network.

    Args:
        shapes: An instance of `OperatorShapes`.
        width: The width of the `DeepResidualNetwork`.
        depth: The depth of the `DeepResidualNetwork`.
        act: Activation function of the `DeepResidualNetwork`.
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        width: int = 32,
        depth: int = 3,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)

        self.width = width
        self.depth = depth

        self.net = DeepResidualNetwork(
            input_size=(
                shapes.x.dim * shapes.x.num + shapes.u.dim * shapes.u.num + shapes.y.dim
            ),
            output_size=shapes.v.dim,
            width=width,
            depth=depth,
            act=act,
            device=device,
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
