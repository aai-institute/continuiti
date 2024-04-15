"""
`continuiti.discrete.uniform`

Uniform samplers.
"""

import torch

from .box_sampler import BoxSampler


class UniformBoxSampler(BoxSampler):
    r"""Sample uniformly from an n-dimensional box.

    Example:
        ```python
        sampler = UniformBoxSampler(torch.tensor([0, 0]), torch.tensor([1, 1]))
        samples = sampler(100)
        samples.shape
        ```
        Output:
        ```
        torch.Size([2, 100])
        ```

    Note:
        Using `torch.rand` the UniformBoxSampler samples from a right-open
        interval in every dimension.
    """

    def __call__(self, n: int) -> torch.Tensor:
        """Generates `n` uniformly distributed samples within the n-dimensional box.

        Args:
            n: Number of samples to draw.

        Returns:
            Samples as tensor of shape (dim, n).
        """
        sample = torch.rand((n, self.ndim))
        x = sample * self.x_delta + self.x_min
        return x.permute(1, 0)
