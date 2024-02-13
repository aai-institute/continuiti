"""
`continuity.data`

Data sets in Continuity.
Every data set is a list of `(x, u, y, v)` tuples.
"""

import torch
import torch.utils.data as td
from typing import Tuple

from .shape import DatasetShape, TensorShape


class OperatorDataset(td.Dataset):
    """A dataset for operator training.

    In operator training, at least one function is mapped onto a second one. To fulfill the properties discretization
    invariance, domain independence and learn operators with physics-based loss access to at least four different
    discretized spaces is necessary. One on which the input is sampled (x), the input function sampled on these points
    (u), the discretization of the output space (y), and the output of the operator (v) sampled on these points. Not
    all loss functions and/or operators need access to all of these attributes.

    Args:
        x: Tensor of shape (#observations, #sensors, x-dim) with sensor positions.
        u: Tensor of shape (#observations, #sensors, u-dim) with evaluations of the input functions at sensor positions.
        y: Tensor of shape (#observations, #evaluations, y-dim) with evaluation positions.
        v: Tensor of shape (#observations, #evaluations, v-dim) with ground truth operator mappings.

    Attributes:
        shape (dataclass): Shape of all tensors.
        transform (dict): Transformations for each tensor.
    """

    def __init__(
            self,
            x: torch.Tensor,
            u: torch.Tensor,
            y: torch.Tensor,
            v: torch.Tensor,
            x_transform=None,
            u_transform=None,
            y_transform=None,
            v_transform=None,
    ):
        assert x.ndim == u.ndim == y.ndim == v.ndim == 3, "Wrong number of dimensions."
        assert (
                x.size(0) == u.size(0) == y.size(0) == v.size(0)
        ), "Inconsistent number of observations."
        assert x.size(1) == u.size(1), "Inconsistent number of sensors."
        assert y.size(1) == v.size(1), "Inconsistent number of evaluations."

        super().__init__()

        self.x = x
        self.u = u
        self.y = y
        self.v = v

        # used to initialize architectures
        self.shape = DatasetShape(
            x=TensorShape(*x.size()[1:]),
            u=TensorShape(*u.size()[1:]),
            y=TensorShape(*y.size()[1:]),
            v=TensorShape(*v.size()[1:]),
        )

        self.transform = {
            dim: tf
            for dim, tf in [
                ("x", x_transform),
                ("u", u_transform),
                ("y", y_transform),
                ("v", v_transform),
            ]
            if tf is not None
        }

    def __len__(self) -> int:
        """Return the number of samples.

        Returns:
            number of samples in the entire set.
        """
        return len(self.u)

    def __getitem__(
            self, idx
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieves the input-output pair at the specified index and applies transformations.

        Parameters:
            - idx: The index of the sample to retrieve.

        Returns:
            A tuple containing the three input tensors and the output tensor for the given index.
        """
        sample = {
            "x": self.x[idx],
            "u": self.u[idx],
            "y": self.y[idx],
            "v": self.v[idx],
        }

        # transform
        for dim, val in sample.items():
            if dim in self.transform:
                sample[dim] = self.transform[dim](val)

        return sample["x"], sample["u"], sample["y"], sample["v"]


class SelfSupervisedOperatorDataset(OperatorDataset):
    """
    A `SelfSupervisedOperatorDataset` is a data set that exports batches of observations
    and labels for self-supervised learning.
    Every data point is created by taking one sensor as label.

    Every batch consists of tuples `(x, u, y, v)`, where `x` contains the sensor
    positions, `u` the sensor values, and `y = x_i` and `v = u_i` are
    the label's coordinate its value for all `i`.

    Args:
        x: Sensor positions of shape (num_observations, num_sensors, coordinate_dim)
        u: Sensor values of shape (num_observations, num_sensors, num_channels)
    """

    def __init__(self, x: torch.Tensor, u: torch.Tensor):
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

        super().__init__(xs, us, ys, vs)
