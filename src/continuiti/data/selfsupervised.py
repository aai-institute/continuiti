"""
`continuiti.data.selfsupervised`

Self-supervised data set.
"""

import math
import torch

from .dataset import OperatorDataset


class SelfSupervisedOperatorDataset(OperatorDataset):
    """
    A `SelfSupervisedOperatorDataset` is a data set that contains data for self-supervised learning.
    Every data point is created by taking one sensor as a label.

    Every observation consists of tuples `(x, u, y, v)`, where `x` contains the sensor
    positions, `u` the sensor values, and `y = x_i` and `v = u_i` are
    the label's coordinate its value for all `i`.

    Args:
        x: Sensor positions of shape (num_observations, coordinate_dim, num_sensors...)
        u: Sensor values of shape (num_observations, num_channels, num_sensors...)
    """

    def __init__(self, x: torch.Tensor, u: torch.Tensor):
        self.num_observations = u.shape[0]
        self.coordinate_dim = x.shape[1]
        self.num_channels = u.shape[1]

        # Check consistency across observations
        for i in range(self.num_observations):
            assert (
                x[i].shape[0] == self.coordinate_dim
            ), "Inconsistent coordinate dimension."
            assert (
                u[i].shape[0] == self.num_channels
            ), "Inconsistent number of channels."

        xs, us, ys, vs = [], [], [], []

        x_flat = x.view(self.num_observations, self.coordinate_dim, -1)
        u_flat = u.view(self.num_observations, self.num_channels, -1)
        self.num_sensors = math.prod(u.shape[2:])

        for i in range(self.num_observations):
            # Add one data point for every sensor
            for j in range(self.num_sensors):
                y = x_flat[i, :, j].unsqueeze(0)
                v = u_flat[i, :, j].unsqueeze(0)

                xs.append(x[i])
                us.append(u[i])
                ys.append(y)
                vs.append(v)

        xs = torch.stack(xs)
        us = torch.stack(us)
        ys = torch.stack(ys)
        vs = torch.stack(vs)

        super().__init__(xs, us, ys, vs)
