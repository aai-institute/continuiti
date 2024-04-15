"""
`continuiti.discrete.regular_grid`

Samplers sampling on a regular grid from n-dimensional boxes.
"""

import torch
from typing import Union

from .box_sampler import BoxSampler


class RegularGridSampler(BoxSampler):
    """Regular Grid sampler class.

    A class for generating regularly spaced samples within an n-dimensional box
    defined by its minimum and maximum corner points. This sampler creates a
    regular grid evenly spaced in each dimension, trying to create sub-boxes
    that are as close to cubical as possible.

    If cubical sub-boxes are not possible (e.g., drawing 8 samples from a unit
    square as 8 is not a power of 2), the `prefer_more_samples` flag determines
    of we either over or undersample the domain: If the product of the number of
    samples in each dimension does not equal the requested number of samples,
    the most under/over-sampled dimension will gain/lose one sample.

    Args:
        x_min: The minimum corner point of the n-dimensional box, specifying the start of each dimension.
        x_max: The maximum corner point of the n-dimensional box, specifying the end of each dimension.
        prefer_more_samples: Flag indicating whether to prefer a sample count slightly above (True) or below (False) the
            desired total if an exact match isn't possible due to the properties of the regular grid. Defaults to True.


    Example:
        ```
        min_corner = torch.tensor([0, 0, 0])  # Define the minimum corner of the box
        max_corner = torch.tensor([1, 1, 1])  # Define the maximum corner of the box
        sampler = RegularGridSampler(min_corner, max_corner, prefer_more_samples=True)
        samples = sampler(100)
        print(samples.shape)
        ```
        Output:
        ```
        torch.Size([125, 3])
        ```
    """

    def __init__(
        self,
        x_min: Union[torch.Tensor, list],
        x_max: Union[torch.Tensor, list],
        prefer_more_samples: bool = True,
    ):
        super().__init__(x_min, x_max)
        self.prefer_more_samples = prefer_more_samples
        if torch.allclose(self.x_delta, torch.zeros(self.x_delta.shape)):
            # all samples are drawn from the same point
            self.x_aspect = torch.zeros(self.x_delta.shape)
            self.x_aspect[0] = 1.0
        else:
            abs_x_delta = torch.abs(self.x_delta)
            self.x_aspect = abs_x_delta / torch.sum(abs_x_delta)

    def __call__(self, n_samples: int) -> torch.Tensor:
        """Generate a uniformly spaced grid of samples within an n-dimensional box.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            Tensor containing the samples of shape (ndim, ~n_samples, ...) as a grid.
        """
        samples_per_dim = self.__calculate_samples_per_dim(n_samples)
        samples_per_dim = self.__adjust_samples_to_fit(n_samples, samples_per_dim)

        # Generate grid
        grids = [
            torch.linspace(start, end, n_samples_dim)
            for start, end, n_samples_dim in zip(
                self.x_min, self.x_max, samples_per_dim
            )
        ]
        mesh = torch.meshgrid(*grids, indexing="ij")

        return torch.stack(mesh, dim=-1).permute(-1, *range(self.ndim))

    def __calculate_samples_per_dim(self, n_samples: int) -> torch.Tensor:
        """Calculate the (floating point) number of samples in each dimension to
        obtain an evenly spaced grid. This method also ensures that there is at
        least one sample in each dimension. The implemented method is best
        understood by the following example.

        Example:
            For `x_min = [0, 0, 1]`, `xmax = [1, 2, 1]` and `n_samples = 200`, this method computes:

            ```
            x_aspect = [1/3, 2/3, 0]
            mask = [1, 1, 0]
            scale_fac = 2/9
            relevant_ndim = 2
            samples_per_dim = (200 / (2/9))^(1 / 2) = sqrt(900) = 30
            samples_per_dim = x_aspect*30 = [10, 20, 0]
            samples_per_dim = max(samples_per_dim, 1) = [10, 20, 1]
            ```
            Output:
            ```
            tensor([10, 20, 1])
            ```

        Args:
            n_samples: Desired total number of samples.

        Returns:
            Approximate number of samples for each dimension as a float vector.
        """
        mask = ~torch.isclose(self.x_aspect, torch.zeros(self.x_aspect.shape))
        scale_fac = torch.prod(self.x_aspect[mask])
        relevant_ndim = torch.sum(mask)
        samples_per_dim = torch.pow(n_samples / scale_fac, 1 / relevant_ndim)
        samples_per_dim = self.x_aspect * samples_per_dim
        samples_per_dim = torch.max(
            samples_per_dim, torch.ones(samples_per_dim.shape)
        )  # ensure every dimension is sampled
        return samples_per_dim

    def __adjust_samples_to_fit(
        self, n_samples: int, samples_per_dim: torch.Tensor
    ) -> torch.Tensor:
        """Round and adjust the `samples_per_dim` to fit the `n_samples` requirement.

        The result of `__calculate_samples_per_dim` is a floating point
        representation, which is rounded by this method to the next integer value.
        If the product of the rounded samples equals the required number of samples,
        we return this number. Otherwise, the most under-sampled dimension or
        the most over-sampled dimension will gain or lose one sample, according
        to the `prefer_more_samples` flag.

        Args:
            n_samples: Desired total number of samples.
            samples_per_dim: Initial distribution of samples across dimensions.

        Returns:
            Adjusted number of samples for each dimension as an integer vector.
        """
        samples_per_dim_int = torch.round(samples_per_dim).to(dtype=torch.int)
        current_total = torch.prod(samples_per_dim_int)

        if current_total == n_samples:
            # no need to adjust anymore
            return samples_per_dim_int

        sample_diff = samples_per_dim - samples_per_dim_int

        if current_total > n_samples and not self.prefer_more_samples:
            # decrease samples in most over-sampled dimension
            dim = torch.argmin(sample_diff)
            samples_per_dim_int[dim] -= 1

        elif current_total < n_samples and self.prefer_more_samples:
            # increase samples in most under-sampled dimension
            dim = torch.argmax(sample_diff)
            samples_per_dim_int[dim] += 1

        return samples_per_dim_int
