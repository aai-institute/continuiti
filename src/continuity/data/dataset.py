"""Base classes for data sets."""

from abc import abstractmethod
import math
import torch
from torch import Tensor
from typing import List, Tuple
from continuity.model import device
from continuity.data import tensor, Observation


class DataSet:
    """Base class."""

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


class SelfSupervisedDataSet(DataSet):
    """
    A `SelfSupervisedDataSet` is a data set constructed from a set of
    observations that exports batches of observations and labels for
    self-supervised learning. Every data point is created by taking one
    sensor as label.

    Every batch consists of tuples `(u, x, v)`, where `u` is the observation
    tensor, `x` is the label's coordinate and `v` is the label.

    Args:
        observations: List of observations.
        batch_size: Batch size.
        shuffle: Shuffle dataset.
    """

    def __init__(
        self,
        observations: List[Observation],
        batch_size: int,
        shuffle: bool = True,
    ):
        self.observations = observations
        self.batch_size = batch_size

        self.num_sensors = observations[0].num_sensors
        self.coordinate_dim = observations[0].sensors[0].coordinate_dim
        self.num_channels = observations[0].sensors[0].num_channels

        # Check consistency across observations
        for observation in self.observations:
            assert (
                observation.num_sensors == self.num_sensors
            ), "Inconsistent number of sensors."
            assert (
                observation.coordinate_dim == self.coordinate_dim
            ), "Inconsistent coordinate dimension."
            assert (
                observation.num_channels == self.num_channels
            ), "Inconsistent number of channels."

        self.u = []
        self.x = []
        self.v = []

        for observation in self.observations:
            u = observation.to_tensor()

            for sensor in observation.sensors:
                x = tensor(sensor.x)
                v = tensor(sensor.u)

                # Add data point for every sensor
                self.u.append(u)
                self.x.append(x)
                self.v.append(v)

        self.u = torch.stack(self.u)
        self.x = torch.stack(self.x)
        self.v = torch.stack(self.v)

        if shuffle:
            idx = torch.randperm(len(self.u))
            self.u = self.u[idx]
            self.x = self.x[idx]
            self.v = self.v[idx]

        # Move to device
        self.to(device=device)

    def get_observation(self, i: int) -> Observation:
        """Return i-th original observation object.

        Args:
            i: Index of observation.

        Returns:
            Observation object.
        """
        return self.observations[i]

    def __len__(self) -> int:
        """Return number of batches.

        Returns:
            Number of batches.
        """
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Return i-th batch as a tuple `(u, x, v)`, where

        - Observations `u` is a tensor of shape `(batch_size, num_sensors, coordinate_dim + num_channels)`
        - Coordinates `x` is a tensor of shape `(batch_size, coordinate_dim)`
        - Labels `v` is a tensor  of shape `(batch_size, num_channels)`

        Args:
            i: Index of batch.

        Returns:
            Batch tuple `(u, x, v)`.
        """
        low = i * self.batch_size
        high = min(low + self.batch_size, len(self.x))
        return self.u[low:high], self.x[low:high], self.v[low:high]

    def to(self, device):
        """Move data set to device.

        Args:
            device: Torch device dataset is moved to.
        """
        self.u = self.u.to(device)
        self.v = self.v.to(device)
        self.x = self.x.to(device)
