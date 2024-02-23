"""
`continuity.discrete.sampler`

Abstract base class for sampling from domains.
"""

from abc import ABC, abstractmethod

import torch


class Sampler(ABC):
    """Abstract base class for sampling discrete points from a domain.

    A sampler $S$ is a mechanism or process that samples discrete points from a domain $D$ based on a specific
    criterion or distribution.

    Args:
        ndim: number of dimensions of the domain.
    """

    def __init__(self, ndim: int):
        self.ndim = ndim

    @abstractmethod
    def __call__(self, n_samples: int) -> torch.Tensor:
        """Draws samples from a domain.

        Args:
            n_samples: number of samples drawn by the sampler from the domain.

        Returns:
            samples as tensor of shape (n_samples, ndim).
        """
