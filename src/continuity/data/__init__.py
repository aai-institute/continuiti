"""
In Continuity, data is given by *observations*. Every observation is a set of
function evaluations, so-called *sensors*. Every data set is a set of
observations, evaluation coordinates and labels.
"""

import torch
from torch import Tensor
from numpy import ndarray
from typing import List, Tuple
from abc import abstractmethod


def tensor(x):
    """Default conversion for tensors."""
    return torch.tensor(x, dtype=torch.float32)


class Sensor:
    """
    A sensor is a function evaluation.

    Args:
        x: spatial coordinate of shape (coordinate_dim)
        u: function value of shape (num_channels)
    """

    def __init__(self, x: ndarray, u: ndarray):
        self.x = x
        self.u = u

        self.coordinate_dim = x.shape[0]
        self.num_channels = u.shape[0]

    def __str__(self) -> str:
        return f"Sensor(x={self.x}, u={self.u})"


class Observation:
    """
    An observation is a set of sensors.

    Args:
        sensors: List of sensors. Used to derive 'num_sensors', 'coordinate_dim' and 'num_channels'.
    """

    def __init__(self, sensors: List[Sensor]):
        self.sensors = sensors

        self.num_sensors = len(sensors)
        assert self.num_sensors > 0

        self.coordinate_dim = self.sensors[0].coordinate_dim
        self.num_channels = self.sensors[0].num_channels

        # Check consistency across sensors
        for sensor in self.sensors:
            assert (
                sensor.coordinate_dim == self.coordinate_dim
            ), "Inconsistent coordinate dimension."
            assert (
                sensor.num_channels == self.num_channels
            ), "Inconsistent number of channels."

    def __str__(self) -> str:
        s = "Observation(sensors=\n"
        for sensor in self.sensors:
            s += f"  {sensor}, \n"
        s += ")"
        return s

    def to_tensor(self) -> torch.Tensor:
        """Convert observation to tensor.

        Returns:
            Tensor of shape (num_sensors, coordinate_dim + num_channels)

        """
        u = torch.zeros((self.num_sensors, self.coordinate_dim + self.num_channels))
        for i, sensor in enumerate(self.sensors):
            u[i] = torch.concat([tensor(sensor.x), tensor(sensor.u)])
        return u


class DataSet:
    """Data set base class."""

    @abstractmethod
    def __len__(self) -> int:
        """Return number of batches.

        Returns:
            Number of batches.
        """

    @abstractmethod
    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return i-th batch as a tuple `(u, x, v)` with tensors for
        observations `u`, coordinates `x` and labels `v`.

        Args:
            i: Index of batch.

        Returns:
            Batch tuple `(u, x, v)`.
        """
