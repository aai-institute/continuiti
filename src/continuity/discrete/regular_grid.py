"""
`continuity.discrete.regular_grid`

Samplers sampling on a regular grid from n-dimensional boxes.
"""

import torch

from .box_sampler import BoxSampler


class RegularGridSampler(BoxSampler):
    """Regular Grid sampler class.

    A class for generating regularly spaced samples within an n-dimensional box defined by its minimum and maximum
    corner points. The goal of this sampler is to create a regular grid evenly spaced in all dimensions. Evenly spaced
    does mean that the sub-boxes created by the grid are as close to cubical as possible. To achieve this, the number of
    samples in each is adjusted dimension based on the box's size and the total desired number of samples. However, as
    this is not always possible (e.g. drawing 8 samples from a 2d unit square as 8 is not a power of 2) there is an
    option to either over or undersample the domain to achieve a regular mesh. If more samples are preferred, the
    sampler will oversample the domain, exceeding the requested number of samples if necessary. If more samples are not
    preferred, the sampler will undersample the domain if necessary. For this, the necessary number of samples is
    distributed to all dimensions according to the aspect ratio of the box. If the product the number of samples in each
    dimension does not equal the requested number of samples, the most under/over-sampled dimension will gain/lose one
    sample.

    Args:
        x_min: The minimum corner point of the n-dimensional box, specifying the start of each dimension.
        x_max: The maximum corner point of the n-dimensional box, specifying the end of each dimension.
        prefer_more_samples: Flag indicating whether to prefer a sample count slightly above (True) or below (False) the
            desired total if an exact match isn't possible due to the properties of the regular grid. Defaults to True.


    Example usage:
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
        self, x_min: torch.Tensor, x_max: torch.Tensor, prefer_more_samples: bool = True
    ):
        super().__init__(x_min, x_max)
        self.prefer_more_samples = prefer_more_samples
        if torch.allclose(self.x_delta, torch.zeros(self.x_delta.shape)):
            # all samples are drawn from the same point
            self.x_aspect = torch.zeros(self.x_delta.shape)
            self.x_aspect[0] = 1.0
        else:
            self.x_aspect = self.x_delta / torch.sum(self.x_delta)

    def __call__(self, n_samples: int) -> torch.Tensor:
        """Generate a uniformly spaced grid of samples within an n-dimensional box.

        Args:
            n_samples: The number of samples to generate.

        Returns:
            Tensor containing the samples of shape (~n_samples, ndim)
        """
        samples_per_dim = self.calculate_samples_per_dim(n_samples)
        samples_per_dim = self.adjust_samples_to_fit(n_samples, samples_per_dim)

        # Generate grid
        grids = [
            torch.linspace(start, end, n_samples_dim)
            for start, end, n_samples_dim in zip(
                self.x_min, self.x_max, samples_per_dim
            )
        ]
        mesh = torch.meshgrid(*grids, indexing="ij")

        return torch.stack(mesh, dim=-1).reshape(-1, self.ndim)

    def calculate_samples_per_dim(self, n_samples: int) -> torch.Tensor:
        """Calculate an approximate distribution of samples across each dimension.

        This is done by taking the product of all non-zero aspect ratios. This scaling factor is used to adjust the
        sample distribution across different dimensions according to their aspect ratios. A higher scale factor
        indicates that the cumulative aspect ratio product is large, suggesting that more samples are needed to
        adequately represent the multidimensional space in proportion to its dimensions' scales. It effectively
        normalizes the sampling process. This normalized aspect ratio is then used to calculate the samples required in
        each dimension to get the exact number of required samples in each dimension. This method also ensures that
        there is at least one sample in each dimension.

        Examples:
            ```python
            xmin=[0,0]
            xmax=[1,2]
            x_aspect = [1/3, 2/3]
            n_samples = 200
            mask = [1,1]
            scale_fac = 2/9
            relevant_ndim = 2
            samples_per_dim = (200 / 2/9)^(1 / 2) = sqrt(900) = 30
            samples_per_dim = [1/3, 2/3]*30 = [10,20]
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

    def adjust_samples_to_fit(
        self, n_samples: int, samples_per_dim: torch.Tensor
    ) -> torch.Tensor:
        """Adjust the samples distribution to better fit the total_samples requirement.

        The first guess drawn in `calculate_samples_per_dim` is in a floating point representation. This method rounds
        this guess to the next integer. If the product of all samples is the required number of samples, this guess is
        returned. Otherwise, the most under-sampled dimension or the most over-sampled dimension are adjusted according
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
