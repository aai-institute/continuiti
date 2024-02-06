"""
`continuity.data`

Data sets in Continuity.
Every data set is a list of `(x, u, y, v)` tuples.
"""

import os
import math
import torch
from torch import Tensor
from typing import Tuple


def get_device() -> torch.device:
    """Get torch device.

    Defaults to `cuda` or `mps` if available, otherwise to `cpu`.

    Use the environment variable `USE_MPS_BACKEND` to disable the `mps` backend.

    Returns:
        Device.
    """
    device = torch.device("cpu")
    use_mps_backend = os.environ.get("USE_MPS_BACKEND", True).lower() in ("true", "1")

    if use_mps_backend and torch.backends.mps.is_available():
        device = torch.device("mps")

    if torch.cuda.is_available():
        device = torch.device("cuda")

    return device


device = get_device()


def tensor(x):
    """Default conversion for tensors."""
    return torch.tensor(x, device=device, dtype=torch.float32)


class DataSet:
    """Data set base class.

    Args:
        x: Tensor of shape (num_observations, num_sensors, coordinate_dim) with sensor positions.
        u: Tensor of shape (num_observations, num_sensors, num_channels)
        y: Tensor of shape (num_observations, num_sensors, coordinate_dim) with evaluation coordinates.
        v: Tensor of shape (num_observations, num_sensors, num_channels) with target labels.
        batch_size: Batch size.
        shuffle: Shuffle data set.

    Attributes:
        num_sensors: Number of sensors.
        coordinate_dim: Coordinate dimension.
        num_channels: Number of channels.
    """

    def __init__(
        self,
        x: Tensor,
        u: Tensor,
        y: Tensor,
        v: Tensor,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.x = x
        self.u = u
        self.y = y
        self.v = v
        self.batch_size = batch_size

        self.num_sensors = u.shape[1]
        self.coordinate_dim = x.shape[-1]
        self.num_channels = u.shape[-1]

        if shuffle:
            idx = torch.randperm(len(self.u))
            self.x = self.x[idx]
            self.u = self.u[idx]
            self.y = self.y[idx]
            self.v = self.v[idx]

        self.to(device=device)

    def __len__(self) -> int:
        """Return number of batches.

        Returns:
            Number of batches.
        """
        return math.ceil(len(self.u) / self.batch_size)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """Return i-th batch as a tuple `(x, u, y, v)`, where

        - Sensor positions `x` is a tensor of shape `(batch_size, num_sensors, coordinate_dim)`
        - Sensor values `u` is a tensor of shape `(batch_size, num_sensors, num_channels)`
        - Evaluation coordinates `y` is a tensor of shape `(batch_size, 1, coordinate_dim)`
        - Labels `v` is a tensor  of shape `(batch_size, 1, num_channels)`

        Args:
            i: Index of batch.

        Returns:
            Batch tuple `(x, u, y, v)`.
        """
        while i < 0:
            i += len(self)
        low = i * self.batch_size
        high = min(low + self.batch_size, len(self.u))
        return self.x[low:high], self.u[low:high], self.y[low:high], self.v[low:high]

    def to(self, device: torch.device):
        """Move data set to device.

        Args:
            device: Torch device dataset is moved to.
        """
        self.x = self.x.to(device)
        self.u = self.u.to(device)
        self.y = self.y.to(device)
        self.v = self.v.to(device)


class SelfSupervisedDataSet(DataSet):
    """
    A `SelfSupervisedDataSet` is a data set that exports batches of observations
    and labels for self-supervised learning.
    Every data point is created by taking one sensor as label.

    Every batch consists of tuples `(x, u, y, v)`, where `x` contains the sensor
    positions, `u` the sensor values, and `y = x_i` and `v = u_i` are
    the label's coordinate its value for all `i`.

    Args:
        x: Sensor positions of shape (num_observations, num_sensors, coordinate_dim)
        u: Sensor values of shape (num_observations, num_sensors, num_channels)
        batch_size: Batch size.
        shuffle: Shuffle dataset.
    """

    def __init__(
        self,
        x: Tensor,
        u: Tensor,
        batch_size: int,
        shuffle: bool = True,
    ):
        self.num_observations = u.shape[0]
        self.num_sensors = u.shape[1]
        self.coordinate_dim = x.shape[-1]
        self.num_channels = u.shape[-1]

        # Check consistency across observations
        for i in range(self.num_observations):
            assert (
                x[i].shape[-1] == self.coordinate_dim
            ), "Inconsistent coordinate dimension."
            assert (
                u[i].shape[-1] == self.num_channels
            ), "Inconsistent number of channels."

        xs, us, ys, vs = [], [], [], []

        for i in range(self.num_observations):
            # Add one data point for every sensor
            for j in range(self.num_sensors):
                y = x[i][j].unsqueeze(0)
                v = u[i][j].unsqueeze(0)

                xs.append(x[i])
                us.append(u[i])
                ys.append(y)
                vs.append(v)

        xs = torch.stack(xs)
        us = torch.stack(us)
        ys = torch.stack(ys)
        vs = torch.stack(vs)

        super().__init__(xs, us, ys, vs, batch_size, shuffle)
