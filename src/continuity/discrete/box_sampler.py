"""
`continuity.discrete.box_sampler`

Abstract base class for sampling from n-dimensional boxes.
"""

from abc import ABC

import torch

from .sampler import Sampler


class BoxSampler(Sampler, ABC):
    r"""Abstract base class for sampling from n-dimensional boxes.

    An n-dimensional box (hyper-rectangle), is described as a Cartesian product of intervals. Each of these intervals
    corresponds to one of the dimensions in an n-dimensional space. Given an n dimensional space $\mathbb{R}^n$, an
    n-dimensional box $\mathcal{B}$ is definded by two points in this space.
    $P_{min}=(x_1^{(min)},x_2^{(min)},\dots,x_n^{(min)})$ and $P_{max}=(x_1^{(max)},x_2^{(max)},\dots,x_n^{(max)})$ are
    the minimum and maximum coordinates of the box along the $i$--th dimension, respectively.
    $\mathcal{B}$ can be represented as:
    $$B = [x_1^{(min)}, x_1^{(max)}] \times [x_2^{(min)}, x_2^{(max)}] \times \dots \times [x_n^{(min)}, x_n^{(max)}]$$,
    where $[x_i^{(min)}, x_i^{(max)}]$$ denotes the closed interval on the $i$-th dimension and $\times$ the Cartesian
    product of the intervals.

    Args:
        x_min: Minimum corner point of the n-dimensional box.
        x_max: Maximum corner point of the n-dimensional box.
    """

    def __init__(self, x_min: torch.Tensor, x_max: torch.Tensor):
        assert x_min.shape == x_max.shape
        assert x_min.ndim == x_max.ndim == 1
        super().__init__(len(x_min))
        self.x_min = x_min
        self.x_max = x_max
        self.x_delta = x_max - x_min
