"""Provide base classes for data sets."""

import math
from numpy import ndarray
from typing import List, Optional
import torch
from continuity.model import device


def tensor(x):
    return torch.tensor(x, dtype=torch.float32)


class Sensor:
    """Sensor

    A sensor is tuple (x, u).
    """

    def __init__(self, x: ndarray, u: ndarray):
        self.x = x
        self.u = u

        self.coordinate_dim = x.shape[0]
        self.num_channels = u.shape[0]

    def __str__(self):
        return f"Sensor(x={self.x}, u={self.u})"


class Observation:
    """Observation

    An observation is a list of N sensors.
    """

    def __init__(self, sensors: List[Optional[Sensor]]):
        self.sensors = sensors
        self.num_sensors = len(sensors)
        assert self.num_sensors > 0
        self.coordinate_dim = self.sensors[0].coordinate_dim
        self.num_channels = self.sensors[0].num_channels

    def __str__(self):
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
    """DataSet

    A data set is constructed from a set of observations and exports batches of observations and labels for self-supervised learning.
    """

    def __init__(
        self,
        observations: List[Observation],
        batch_size: int,
    ):
        self.observations = observations
        self.batch_size = batch_size

        num_sensors = observations[0].num_sensors
        coordinate_dim = observations[0].sensors[0].coordinate_dim
        num_channels = observations[0].sensors[0].num_channels

        self.u = []
        self.v = []
        self.x = []

        for observation in self.observations:
            u = observation.to_tensor()
            v = torch.zeros((num_sensors, num_channels))
            x = torch.zeros((num_sensors, coordinate_dim))

            for i, sensor in enumerate(observation.sensors):
                v[i] = tensor(sensor.u)  # v = u
                x[i] = tensor(sensor.x)  # y = x

            self.u.append(u)
            self.v.append(v)
            self.x.append(x)

        self.u = torch.stack(self.u)
        self.v = torch.stack(self.v)
        self.x = torch.stack(self.x)

        # Randomize
        idx = torch.randperm(len(self.u))
        self.u = self.u[idx]
        self.v = self.v[idx]
        self.x = self.x[idx]

        # Move to device
        self.to(device=device)

    def get_observation(self, idx: int) -> Observation:
        """Return observation at index"""
        return self.observations[idx]

    def __len__(self) -> int:
        """Return number of batches"""
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx: int):
        """Return batch of observations"""
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        return self.u[low:high], self.v[low:high], self.x[low:high]

    def to(self, device):
        """Move data set to device"""
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.x = self.x.to(device)
