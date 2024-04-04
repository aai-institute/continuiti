"""
`continuity.transforms.quantile_scaler`

Quantile Scaler class.
"""

import torch
from .transform import Transform
from typing import Union


class QuantileScaler(Transform):
    """Quantile Scaler Class.

    A transform for scaling input data to a specified target distribution using quantiles. This is
    particularly useful for normalizing data in a way that is more robust to outliers than standard
    z-score normalization.

    The transformation maps the quantiles of the input data to the quantiles of the target distribution,
    effectively performing a non-linear scaling that preserves the relative distribution of the data.

    Args:
        src: tensor from which the source distribution is drawn.
        n_quantile_intervals: Number of individual bins into which the data is categorized.
        target_mean: Mean of the target Gaussian distribution. Can be float (all dimensions use the same mean), or
            tensor (allows for different means along different dimensions).
        target_std: Std of the target Gaussian distribution. Can be float (all dimensions use the same std), or
            tensor (allows for different stds along different dimensions).
        eps: Small value to bound the target distribution to a finite interval.

    """

    def __init__(
        self,
        src: torch.Tensor,
        n_quantile_intervals: int = 1000,
        target_mean: Union[float, torch.Tensor] = 0.0,
        target_std: Union[float, torch.Tensor] = 1.0,
        eps: float = 1e-3,
    ):
        assert eps <= 0.5
        assert eps >= 0

        if isinstance(target_mean, float):
            target_mean = target_mean * torch.ones(1)
        if isinstance(target_std, float):
            target_std = target_std * torch.ones(1)
        self.target_mean = target_mean
        self.target_std = target_std

        assert n_quantile_intervals > 0
        self.n_quantile_intervals = n_quantile_intervals
        self.n_q_points = n_quantile_intervals + 2  # n intervals have n + 2 edges

        self.n_dim = src.size(-1)

        # source "distribution"
        self.quantile_fractions = torch.linspace(0, 1, self.n_q_points)
        self.quantile_points = torch.quantile(
            src.view(-1, self.n_dim),
            self.quantile_fractions,
            dim=0,
            interpolation="linear",
        )
        self.deltas = self.quantile_points[1:] - self.quantile_points[:-1]
        self.src_max = torch.max(src)
        self.src_min = torch.min(src)

        # target distribution
        self.target_distribution = torch.distributions.normal.Normal(
            target_mean, target_std
        )
        self.target_quantile_fractions = torch.linspace(
            0 + eps, 1 - eps, self.n_q_points
        )  # bounded domain
        target_quantile_points = self.target_distribution.icdf(
            self.target_quantile_fractions
        )
        self.target_quantile_points = target_quantile_points.unsqueeze(1).repeat(
            1, self.n_dim
        )
        self.target_deltas = (
            self.target_quantile_points[1:] - self.target_quantile_points[:-1]
        )

        super().__init__()

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """Transforms the input tensor to match the target distribution using quantile scaling .

        Args:
            tensor: The input tensor to transform.

        Returns:
            The transformed tensor, scaled to the target distribution.
        """
        work_dim = 2  # dimension alon which calculations are performed
        bcs = tensor.size(0)
        n_elements = tensor.size(1)

        # find left boundary inside quantile intervals
        v1 = tensor.unsqueeze(work_dim).expand(-1, -1, self.n_q_points, -1)
        v2 = (
            self.quantile_points.unsqueeze(0)
            .unsqueeze(0)
            .expand(bcs, v1.size(1), self.n_q_points, -1)
        )
        diff = v1 - v2
        diff[diff > 0] = -torch.inf  # left boundary (either negative or zero)
        indices = diff.argmax(dim=work_dim)  # defaults to zero when all values are -inf

        # prepare for indexing
        indices = (indices.view(-1), torch.arange(self.n_dim).repeat(bcs * n_elements))

        # Scale input tensor to the unit interval based on source quantiles
        p_min = self.quantile_points[indices].view(tensor.shape)
        delta = self.deltas[indices].view(tensor.shape)
        out = tensor - p_min
        out = out / delta

        # Scale and shift to match the target distribution
        p_t_min = self.target_quantile_points[indices].view(tensor.shape)
        delta_t = self.target_deltas[indices].view(tensor.shape)
        out = out * delta_t
        out = out + p_t_min

        return out

    def undo(self, tensor: torch.Tensor) -> torch.Tensor:
        """Reverses the transformation applied by the forward method, mapping the tensor back to its original
        distribution.

        Args:
            tensor: The tensor to reverse the transformation on.

        Returns:
            The tensor with the quantile scaling transformation reversed according to the src distribution.
        """
        work_dim = 2
        bcs = tensor.size(0)
        n_elements = tensor.size(1)

        # find left boundary inside quantile intervals
        v1 = tensor.unsqueeze(work_dim).expand(-1, -1, self.n_q_points, -1)
        v2 = (
            self.target_quantile_points.unsqueeze(0)
            .unsqueeze(0)
            .expand(bcs, v1.size(1), self.n_q_points, -1)
        )
        diff = v1 - v2
        diff[diff > 0] = -torch.inf  # left boundary (either negative or zero)
        indices = diff.argmax(dim=work_dim)  # defaults to zero when all values are -inf

        # prepare for indexing
        indices = (indices.view(-1), torch.arange(self.n_dim).repeat(bcs * n_elements))

        # Scale input tensor to the unit interval based on the target distribution
        p_t_min = self.target_quantile_points[indices].view(tensor.shape)
        delta_t = self.target_deltas[indices].view(tensor.shape)
        out = tensor - p_t_min
        out = out / delta_t

        # Scale and shift to match the src distribution
        p_min = self.quantile_points[indices].view(tensor.shape)
        delta = self.deltas[indices].view(tensor.shape)
        out = out * delta
        out = out + p_min

        return out
