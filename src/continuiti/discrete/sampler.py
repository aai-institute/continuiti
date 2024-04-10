"""
`continuiti.discrete.sampler`

Abstract base class for sampling from domains.
"""

from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    """Abstract base class for sampling discrete points from a domain.

    A sampler is a mechanism or process that samples discrete points from a
    domain based on a specific criterion or distribution.

    Args:
        ndim: Dimension of the domain.
    """

    def __init__(self, ndim: int):
        self.ndim = ndim

    @abstractmethod
    def __call__(self, n: int) -> torch.Tensor:
        """Draws samples from a domain.

        Args:
            n: Number of samples drawn by the sampler from the domain.

        Returns:
            Samples as tensor of shape (n, ndim).
        """
