"""
`continuity.operators.operator`

Operators can be stacked into a `NeuralOperator` architecture, which is a
stack of continuous convolutions with a lifting layer and a projection layer.
"""

import torch
from continuity.operators import Operator
from continuity.data import DatasetShapes
from continuity.operators.integralkernel import NaiveIntegralKernel, NeuralNetworkKernel


class NeuralOperator(Operator):
    r"""Neural operator architecture

    Maps continuous functions given as observation to another continuous function and returns point-wise evaluations.
    The architecture is a stack of continuous kernel integrations with a lifting layer and a projection layer.

    *Reference:* N. Kovachki et al. Neural Operator: Learning Maps Between
    Function Spaces With Applications to PDEs. JMLR 24 1-97 (2023)

    For now, sensor positions are equal across all layers.

    TODO: This implementation has to be generalized to arbitrary `Operator` layers.

    Args:
        coordinate_dim: Dimension of coordinate space
        num_channels: Number of channels
        depth: Number of hidden layers
        kernel_width: Width of kernel network
        kernel_depth: Depth of kernel network
    """

    def __init__(
        self,
        shapes: DatasetShapes,
        depth: int = 1,
        kernel_width: int = 32,
        kernel_depth: int = 3,
    ):
        super().__init__()

        self.shapes = shapes
        self.lifting = NaiveIntegralKernel(
            NeuralNetworkKernel(shapes, kernel_width, kernel_depth),
        )

        self.hidden_layers = torch.nn.ModuleList(
            [
                NaiveIntegralKernel(
                    NeuralNetworkKernel(shapes, kernel_width, kernel_depth),
                )
                for _ in range(depth)
            ]
        )

        self.projection = NaiveIntegralKernel(
            NeuralNetworkKernel(shapes, kernel_width, kernel_depth),
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
        # Lifting layer (we use x as evaluation coordinates for now)
        v = self.lifting(x, u, x)
        assert v.shape[1:] == torch.Size([self.shapes.x.num, self.shapes.u.dim])

        # Hidden layers
        for layer in self.hidden_layers:
            # Layer operation (with residual connection)
            v = layer(x, v, x) + v
            assert v.shape[1:] == torch.Size([self.shapes.x.num, self.shapes.u.dim])

            # Activation
            v = torch.tanh(v)

        # Projection layer
        w = self.projection(x, v, y)
        assert w.shape[1:] == torch.Size([y.size(1), self.shapes.u.dim])
        return w
