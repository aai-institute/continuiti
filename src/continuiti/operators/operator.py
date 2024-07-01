"""
`continuiti.operators.operator`

In continuiti, all models for operator learning are based on the `Operator` base class.
"""

import torch
from typing import Optional
from abc import ABC, abstractmethod
from continuiti.operators.shape import OperatorShapes


class Operator(torch.nn.Module, ABC):
    """Operator base class.

    An operator is a neural network model that maps functions by mapping an
    observation to the evaluations of the mapped function at given coordinates.

    Args:
        shapes: Operator shapes.
        device: Device.

    Attributes:
        shapes: Operator shapes.
    """

    device: Optional[torch.device]
    shapes: OperatorShapes

    def __init__(
        self,
        shapes: Optional[OperatorShapes] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.shapes = shapes
        self.device = device

    def save(self, path: str):
        """Save the operator to a file.

        Args:
            path: Path to the file.
        """
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load the operator from a file.

        Args:
            path: Path to the file.
        """
        self.load_state_dict(torch.load(path))

    def num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def __str__(self):
        """Return string representation of the operator."""
        return self.__class__.__name__


class UnmaskedOperator(Operator, ABC):
    """Masked operator base class.

    An unmasked operator processes all values in the input without applying any masks during the forward pass. This
    means that all input values are considered during the forward pass, no values are ignored.

    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).

        Returns:
            Evaluations of the mapped function with shape (batch_size, v_dim, num_evaluations...).
        """


class MaskedOperator(Operator, ABC):
    """Masked operator base class.

    A masked operator can apply masks during the forward pass to selectively use or ignore parts of the input. Masked
    operators allow for different numbers of sensors in addition to the common property of being able to handle
    variing numbers of evaluations.

    """

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        u: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, x_dim, num_sensors...).
            u: Input function values of shape (batch_size, u_dim, num_sensors...).
            y: Evaluation coordinates of shape (batch_size, y_dim, num_evaluations...).
            mask: A boolean mask where a value of True indicates that the element should be taken into consideration
                during the forward pass of shape (batch_size, num_evaluations..., num_sensors...).

        Returns:
            Evaluations of the mapped function with shape (batch_size, v_dim, num_evaluations...).
        """
