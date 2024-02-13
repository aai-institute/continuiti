"""Various data set implementations."""

import torch
import numpy as np
from typing import Tuple

from continuity.data import OperatorDataset


class Sine(OperatorDataset):
    r"""Creates a data set of sine waves.

    The data set is generated by sampling sine waves at the given number of
    sensors placed evenly in the interval $[-1, 1]$. The wave length of the
    sine waves is evenly distributed between $\pi$ for the first observation
    and $2\pi$ for the last observation, respectively.

    The `Sine` dataset generates $N$ sine waves
    $$
    f(x) = \sin(w_k x), \quad w_k = 1 + \frac{k}{N-1}, \quad k = 0, \dots, N-1.
    $$
    It exports batches of samples
    $$
    \left(\mathbf{x}, f(\mathbf{x}), \mathbf{x}, f(\mathbf{x})\right),
    $$
    $$
    where $\mathbf{x} = (x_i)_{i=1 \dots M}$ are the $M$ equidistantly
    distributed sensor positions.

    - coordinate_dim: 1
    - num_channels: 1

    Args:
        num_sensors: Number of sensors.
        size: Size of data set.
    """

    def __init__(self, num_sensors: int, size: int):
        self.num_sensors = num_sensors
        self.size = size

        self.coordinate_dim = 1
        self.num_channels = 1

        # Generate observations
        observations = [self.generate_observation(i) for i in range(self.size)]

        # observations used as labels y, v
        x = torch.stack([x for x, _ in observations])
        u = torch.stack([u for _, u in observations])

        super().__init__(x, u, x, u)

    def generate_observation(self, i: float) -> Tuple[np.array, np.array]:
        """Generate observation

        Args:
            i: Index of observation (0 <= i <= size).
        """
        # Create x of shape (n, 1)
        x = np.linspace(-1, 1, self.num_sensors).reshape(-1, 1)

        if self.size == 1:
            w = 1
        else:
            w = 1 + i / (self.size - 1)

        u = np.sin(w * np.pi * x)

        return torch.tensor(x), torch.tensor(u)
