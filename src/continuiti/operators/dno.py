"""
`continuiti.operators.dno`

The Deep Neural Operator (DNO) architecture.
"""

import math
import torch
from typing import Optional
from continuiti.operators import Operator
from continuiti.networks import DeepResidualNetwork
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

        self.x_num = math.prod(shapes.x.size)
        self.u_num = math.prod(shapes.u.size)
        self.net_input_size = (
            shapes.x.dim * self.x_num + shapes.u.dim * self.u_num + shapes.y.dim
        )

        self.net = DeepResidualNetwork(
            input_size=self.net_input_size,
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
            x: Input coordinates of shape (batch_size, x_dim, num_sensors...), representing the points in space at
                which the input function values are probed.
            u: Input function values of shape (batch_size, u_dim, num_sensors...), representing the values of the input
                functions at different sensor locations.
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...), representing the points in space at
                which the output function values are to be computed.

        Returns:
            The output of the operator, of shape (batch_size, v_dim, num_evaluations...), representing the computed function
                values at the specified evaluation coordinates.
        """
        batch_size = u.size(0)
        y_num = math.prod(y.size()[2:])

        u_repeated = u.flatten(1, -1).unsqueeze(1).expand(-1, y_num, -1)
        assert u_repeated.shape == (batch_size, y_num, self.shapes.u.dim * self.u_num)

        x_repeated = x.flatten(1, -1).unsqueeze(1).expand(-1, y_num, -1)
        assert x_repeated.shape == (batch_size, y_num, self.shapes.x.dim * self.x_num)

        y_flatten = y.flatten(2, -1).transpose(1, 2)
        assert y_flatten.shape == (batch_size, y_num, self.shapes.y.dim)

        net_input = torch.cat([x_repeated, u_repeated, y_flatten], dim=-1)
        assert net_input.shape == (batch_size, y_num, self.net_input_size)

        net_output = self.net(net_input)
        assert net_output.shape == (batch_size, y_num, self.shapes.v.dim)

        net_output = net_output.transpose(1, 2)
        assert net_output.shape == (batch_size, self.shapes.v.dim, y_num)

        net_output = net_output.reshape(batch_size, self.shapes.v.dim, *y.size()[2:])

        return net_output
