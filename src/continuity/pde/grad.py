"""
`continuity.pde.grad`

Functional gradients in Continuity.

Derivatives are function operators, so it is natural to define them as operators
within Continuity.

The following gradients define several derivation operators (e.g., grad, div)
that simplify the definition of PDEs in physics-informed losses.
"""

import torch
from torch import Tensor
from typing import Optional, Callable
from continuity.operators.operator import Operator


class Grad(Operator):
    """Gradient operator.

    The gradient is a function operator that maps a function to its gradient.
    """

    def forward(self, x: Tensor, u: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the operator.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, input_coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, input_channels)
            y: Tensor of evaluation positions of shape (batch_size, y_size, output_coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, output_channels)
        """
        if y is not None:
            assert torch.equal(x, y), "x and y must be equal for gradient operator"

        assert x.requires_grad, "x must require gradients for gradient operator"

        # Compute gradients
        gradients = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        return gradients


def grad(u: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Compute the gradient of a function.

    Args:
        u: Function to compute the gradient of.

    Returns:
        Function that computes the gradient of the input function.
    """
    return lambda x: Grad()(x, u(x))


class Div(Operator):
    """Divergence operator.

    The divergence is a function operator that maps a function to its divergence.
    """

    def forward(self, x: Tensor, u: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """Forward pass through the operator.

        Args:
            x: Tensor of sensor positions of shape (batch_size, num_sensors, input_coordinate_dim)
            u: Tensor of sensor values of shape (batch_size, num_sensors, input_channels)
            y: Tensor of evaluation positions of shape (batch_size, y_size, output_coordinate_dim)

        Returns:
            Tensor of evaluations of the mapped function of shape (batch_size, y_size, output_channels)
        """
        if y is not None:
            assert torch.equal(x, y), "x and y must be equal for divergence operator"

        assert x.requires_grad, "x must require gradients for divergence operator"

        # Compute gradients
        gradients = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]

        return torch.sum(gradients, dim=-1, keepdim=True)


def div(u: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
    """Compute the divergence of a function.

    Args:
        u: Function to compute the divergence of.

    Returns:
        Function that computes the divergence of the input function.
    """
    return lambda x: Div()(x, u(x))
