"""
`continuity.discrete.uniform`

Samplers sampling uniformly from n-dimensional boxes.
"""

import torch

from .box_sampler import BoxSampler


class UniformBoxSampler(BoxSampler):
    r"""Class for sampling uniformly from a n-dimensional box.

    A sampler for uniformly sampling points from a given n-dimensional box, where the sampling interval for each
    dimension is inclusive of both the lower and upper bounds.

    This class extends BoxSampler and provides functionality to sample points uniformly from an n-dimensional
    hyperrectangle (or box). The sampling considers an inclusive range for each dimension, meaning that the edge values
    of the box can also be sampled.

    Example:
        >>> sampler = UniformBoxSampler(torch.tensor([0, 0]), torch.tensor([1, 1]))
        >>> samples = sampler(100)
    """

    def __call__(self, n_samples: int) -> torch.Tensor:
        """Generates a sample within the n-dimensional box. The sample is uniformly distributed, and every point within
        the specified box, including the edges, has an equal chance of being sampled.

        Args:
            n_samples: number of samples to draw.

        Returns:
            samples as tensor of shape (n_samples, ndim)
        """
        sample = torch.rand((n_samples, self.ndim))
        sample = sample * (
            1 + torch.finfo(torch.get_default_dtype()).tiny
        )  # inclusive upper interval bound
        return sample * self.x_delta + self.x_min
