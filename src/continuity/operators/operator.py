"""
`continuity.operators.operator`

In Continuity, all models for operator learning are based on the `Operator` base class.
"""

import torch
from abc import abstractmethod


class Operator(torch.nn.Module):
    """Operator base class.

    An operator is a neural network model that maps functions by mapping an
    observation to the evaluations of the mapped function at given coordinates.

    This class implements default `compile` and `fit` methods.
    """

    @abstractmethod
    def forward(
        self, x: torch.Tensor, u: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the operator.

        Args:
            x: Sensor positions of shape (batch_size, num_sensors, x_dim)
            u: Input function values of shape (batch_size, num_sensors, u_dim)
            y: Evaluation coordinates of shape (batch_size, num_evaluations, y_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, num_evaluations, v_dim)
        """

    def __str__(self):
        """String representation of the operator."""
        return self.__class__.__name__

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
