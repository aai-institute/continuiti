"""
`continuiti.discrete.box_sampler`

Abstract base class for sampling from n-dimensional boxes.
"""

from abc import ABC
from typing import Union

import torch

from .sampler import Sampler


class BoxSampler(Sampler, ABC):
    r"""Abstract base class for sampling from n-dimensional boxes.

    Given two points in $\mathbb{R}^n$ with coordinates
    $$
    (x_1^{(min)},x_2^{(min)},\dots,x_n^{(min)}), \quad
    (x_1^{(max)},x_2^{(max)},\dots,x_n^{(max)}),
    $$
    an n-dimensional box $B \subset \mathbb{R}^n$ is given by the
    Cartesian product
    $$
    B = [x_1^{(min)}, x_1^{(max)}] \times \dots \times [x_n^{(min)}, x_n^{(max)}].
    $$

    Args:
        x_min: Minimum corner point of the n-dimensional box.
        x_max: Maximum corner point of the n-dimensional box.
    """

    def __init__(
        self, x_min: Union[torch.Tensor, list], x_max: Union[torch.Tensor, list]
    ):
        # Convert lists to tensors
        if isinstance(x_min, list):
            x_min = torch.tensor(x_min)
        if isinstance(x_max, list):
            x_max = torch.tensor(x_max)

        assert x_min.shape == x_max.shape
        assert x_min.ndim == x_max.ndim == 1
        super().__init__(len(x_min))
        self.x_min = x_min
        self.x_max = x_max
        self.x_delta = x_max - x_min
