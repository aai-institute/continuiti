"""Provide base classes for data sets."""

import math
import torch
from typing import List, Tuple
from continuity.model import device
from continuity.data import tensor, Observation


class SelfSupervisedDataSet:
    """SelfSupervisedDataSet

    A self-supervised data set is constructed from a set of observations and
    exports batches of observations and labels for self-supervised learning.
    Every data point is created by taking one observation as label.

    It returns data points as tuple (u, v, x), where u is a tensor of the
    observation, v is the label and x the label's coordinate.
    """

    def __init__(
        self,
        observations: List[Observation],
        batch_size: int,
        randomize: bool = True,
    ):
        self.observations = observations
        self.batch_size = batch_size

        num_sensors = observations[0].num_sensors
        coordinate_dim = observations[0].sensors[0].coordinate_dim
        num_channels = observations[0].sensors[0].num_channels

        # Check consistency across observations
        for observation in self.observations:
            assert (
                observation.num_sensors == num_sensors
            ), "Inconsistent number of sensors."
            assert (
                observation.coordinate_dim == coordinate_dim
            ), "Inconsistent coordinate dimension."
            assert (
                observation.num_channels == num_channels
            ), "Inconsistent number of channels."

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

                # Add data point for every sensor
                self.u.append(u)
                self.v.append(v)
                self.x.append(x)

        self.u = torch.stack(self.u)
        self.v = torch.stack(self.v)
        self.x = torch.stack(self.x)

        if randomize:
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

    def __getitem__(self, idx: int) -> Tuple[tensor, tensor, tensor]:
        """Return batch of observations"""
        low = idx * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        return self.u[low:high], self.v[low:high], self.x[low:high]

    def to(self, device):
        """Move data set to device"""
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.x = self.x.to(device)
