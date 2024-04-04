"""
`continuity.operators.integralkernel`

Integral kernel operations.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional
from continuity.operators import Operator
from continuity.operators.common import DeepResidualNetwork
from continuity.operators.shape import OperatorShapes


class Kernel(torch.nn.Module, ABC):
    r"""Kernel abstract base class.

    In general, a kernel is a function

    \begin{align*}
    \kappa:\ \mathbb{R}^{d_x} \times \mathbb{R}^{d_y} &\to \mathbb{R}^{{d_u}\times{d_v}}, \\
            (x, y) &\mapsto \kappa(x, y).
    \end{align*}

    In Continuity, we add a batch dimension and the number of sensor points for
    $x$ and $y$ to enable efficient implementation of the kernel, such that the
    shapes of the input tensors are

    ```python
        x: (batch_size, x_num, x_dim)
        y: (batch_size, y_num, y_dim)
    ```
    and the kernel output is of shape

    ```python
        (batch_size, x_num, y_num, u_dim, v_dim)
    ```

    Args:
        shapes: Shapes of the operator.
        device: Device.
    """

    def __init__(self, shapes: OperatorShapes, device: Optional[torch.device] = None):
        super().__init__()
        self.shapes = shapes
        self.device = device

    @abstractmethod
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of coordinates of shape `(batch_size, x_num, shapes.x.dim)`.
            y: Tensor of coordinates of shape `(batch_size, y_num, shapes.y.dim)`.

        Returns:
            Tensor of shape `(batch_size, x_num, y_num, shapes.u.dim, shapes.v.dim)`.
        """


class NeuralNetworkKernel(Kernel):
    r"""Neural network kernel.

    The neural network kernel is a kernel where $\kappa(x, y)$ is parameterized
    by a simple feed forward neural network with skip connections.

    Args:
        shapes: Shapes of the operator
        kernel_width: Width of kernel network
        kernel_depth: Depth of kernel network
        act: Activation function
        device: Device
    """

    def __init__(
        self,
        shapes: OperatorShapes,
        kernel_width: int,
        kernel_depth: int,
        act: Optional[torch.nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__(shapes, device)

        self.shapes = shapes
        self.net = DeepResidualNetwork(
            input_size=shapes.y.dim + shapes.x.dim,
            output_size=shapes.u.dim * shapes.v.dim,
            width=kernel_width,
            depth=kernel_depth,
            act=act,
            device=device,
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of coordinates of shape `(batch_size, x_num, shapes.x.dim)`.
            y: Tensor of coordinates of shape `(batch_size, y_num, shapes.y.dim)`.

        Returns:
            Tensor of shape `(batch_size, x_num, y_num, shapes.u.dim, shapes.v.dim)`.
        """

        # shapes that can change for different forward passes
        batch_size = x.shape[0]
        assert batch_size == y.shape[0]
        x_num, y_num = x.shape[1], y.shape[1]

        # shapes that are fixed
        x_dim, y_dim = self.shapes.x.dim, self.shapes.y.dim
        u_dim, v_dim = self.shapes.u.dim, self.shapes.v.dim

        # In order to evaluate all kernel values k(x_i, y_j), we need every x_i and y_j combination.
        network_input = torch.concat(
            [
                x.unsqueeze(2).repeat(1, 1, y_num, 1),  # repeat tensor in dim=2
                y.unsqueeze(1).repeat(1, x_num, 1, 1),  # repeat tensor in dim=1
            ],
            dim=-1,
        )
        assert network_input.shape == torch.Size(
            [batch_size, x_num, y_num, x_dim + y_dim]
        )

        output = self.net(network_input)
        assert output.shape == torch.Size([batch_size, x_num, y_num, u_dim * v_dim])

        output = output.reshape(batch_size, x_num, y_num, u_dim, v_dim)

        return output


class NaiveIntegralKernel(Operator):
    r"""Naive integral kernel operator.

    Maps continuous functions via integral kernel application to another
    continuous function and returns point-wise evaluations.

    In mathematical terms, for some given $y$, we obtain
    $$
    v(y) = \int u(x)~\kappa(x, y)~dx
        \approx \frac{1}{N} \sum_{i=1}^{N} u_i~\kappa(x_i, y)
    $$
    where $(x_i, u_i)$ are the $N$ sensors where $u$ is evaluated.

    Note:
        This implementation is not efficient for a large number of sensors and
        only serves as a proof of concept. Please refer to other integral kernel
        operators for more efficient implementations (e.g. Fourier layers).

    Args:
        kernel: Kernel function.
    """

    def __init__(
        self,
        kernel: Kernel,
    ):
        super().__init__(kernel.shapes, kernel.device)
        self.kernel = kernel

    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, x_dim)
            u: Input function values of shape (batch_size, num_sensors, u_dim)
            y: Evaluation coordinates of shape (batch_size, num_evaluations, y_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, num_evaluations, v_dim)
        """
        # shapes that can change for different forward passes
        batch_size = x.shape[0]
        assert batch_size == y.shape[0]
        x_num, y_num = x.shape[1], y.shape[1]

        # shapes that are fixed
        u_dim, v_dim = self.shapes.u.dim, self.shapes.v.dim

        # Apply the kernel function
        k = self.kernel(x, y)
        assert k.shape == torch.Size([batch_size, x_num, y_num, u_dim, v_dim])

        # Compute integral
        assert u.shape == torch.Size([batch_size, x_num, u_dim])
        integral = torch.einsum("bxyuv,bxu->byv", k, u) / x_num
        assert integral.shape == torch.Size([batch_size, y_num, v_dim])

        return integral
