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
            x: Sensor positions of shape (batch_size, num_sensors, input_coordinate_dim)
            u: Input function values of shape (batch_size, num_sensors, input_channels)
            y: Coordinates where the mapped function is evaluated of shape (batch_size, y_size, output_coordinate_dim)

        Returns:
            Evaluations of the mapped function with shape (batch_size, y_size, output_channels)
        """

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
